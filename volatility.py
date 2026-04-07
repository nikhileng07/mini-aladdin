"""
volatility.py
-------------
Volatility analytics for portfolio and individual assets:
- Realized volatility (rolling windows)
- EWMA volatility (RiskMetrics model)
- Volatility regime detection (low / medium / high)
- Volatility term structure
- Asset-level volatility contribution
Reads from S3/MinIO with DuckDB fallback. Outputs to CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from analytics.portfolio_returns import load_price_data


TRADING_DAYS = 252
OUTPUT_DIR   = Path("data/processed/analytics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Volatility Engine ─────────────────────────────────────────────────────────

class VolatilityEngine:

    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS", "BAC"]

    def __init__(self, portfolio_id: str, weights: dict = None):
        self.portfolio_id = portfolio_id
        self.weights = weights or {s: 1 / len(self.SYMBOLS) for s in self.SYMBOLS}

    def _portfolio_returns(self, prices: pd.DataFrame) -> pd.Series:
        returns = prices[self.SYMBOLS].pct_change().dropna()
        w = np.array([self.weights.get(s, 0) for s in self.SYMBOLS if s in returns.columns])
        w = w / w.sum()
        cols = [s for s in self.SYMBOLS if s in returns.columns]
        return (returns[cols] * w).sum(axis=1)

    # ── Realized Volatility ───────────────────────────────────────────────────

    def realized_volatility(
        self,
        returns: pd.Series,
        windows: list[int] = [21, 63, 126, 252],
    ) -> pd.DataFrame:
        """
        Rolling realized volatility over multiple windows.
        Annualized: vol_daily * sqrt(252)
        """
        df = pd.DataFrame({"date": returns.index, "daily_return": returns.values})
        df["date"] = pd.to_datetime(df["date"]).dt.date

        for w in windows:
            col = f"vol_{w}d"
            df[col] = (
                returns.rolling(w)
                .std()
                .values * np.sqrt(TRADING_DAYS)
            )

        df["portfolio_id"] = self.portfolio_id
        return df

    # ── EWMA Volatility ───────────────────────────────────────────────────────

    def ewma_volatility(
        self,
        returns: pd.Series,
        lambda_: float = 0.94,          # RiskMetrics standard decay factor
    ) -> pd.DataFrame:
        """
        Exponentially Weighted Moving Average volatility.
        More responsive to recent shocks than simple rolling vol.
        lambda=0.94 is JP Morgan RiskMetrics standard.
        lambda=0.97 is smoother (used for monthly horizons).
        """
        squared = returns ** 2
        ewma_var = squared.ewm(alpha=1 - lambda_, adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var) * np.sqrt(TRADING_DAYS)

        df = pd.DataFrame({
            "date": returns.index,
            "portfolio_id": self.portfolio_id,
            "ewma_vol_annualized": ewma_vol.values,
            "lambda": lambda_,
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # ── Volatility Regime ─────────────────────────────────────────────────────

    def volatility_regime(self, realized_vol_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each day into a volatility regime based on 21d vol:
          LOW    < 10th percentile
          MEDIUM  10th–75th percentile
          HIGH   > 75th percentile
        Useful for risk budgeting — reduce exposure in HIGH regime.
        """
        df = realized_vol_df.copy().dropna(subset=["vol_21d"])
        p10 = df["vol_21d"].quantile(0.10)
        p75 = df["vol_21d"].quantile(0.75)

        df["vol_regime"] = pd.cut(
            df["vol_21d"],
            bins=[-np.inf, p10, p75, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        )
        df["p10_threshold"] = round(p10, 4)
        df["p75_threshold"] = round(p75, 4)
        return df[["date", "portfolio_id", "vol_21d", "vol_regime", "p10_threshold", "p75_threshold"]]

    # ── Volatility Term Structure ─────────────────────────────────────────────

    def vol_term_structure(self, returns: pd.Series) -> pd.DataFrame:
        """
        Vol term structure: how volatility changes across horizons.
        Under random walk: vol(T) = vol(1) * sqrt(T)
        Deviation from sqrt(T) scaling = mean reversion or clustering signal.
        """
        horizons = [1, 5, 10, 21, 63, 126, 252]
        rows = []
        for h in horizons:
            compounded = returns.rolling(h).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
            realized = float(compounded.std() * np.sqrt(TRADING_DAYS / h))
            theoretical = float(returns.std() * np.sqrt(TRADING_DAYS) * np.sqrt(h / 1) / np.sqrt(h))
            rows.append({
                "portfolio_id": self.portfolio_id,
                "horizon_days": h,
                "realized_vol": round(realized, 4),
                "theoretical_vol_sqrt_t": round(theoretical, 4),
                "vol_ratio": round(realized / theoretical, 4) if theoretical > 0 else None,
            })
        return pd.DataFrame(rows)

    # ── Asset Volatility Contribution ────────────────────────────────────────

    def asset_vol_contribution(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Marginal risk contribution per asset.
        MRC_i = w_i * (Sigma * w)_i / portfolio_vol
        Tells you which stock is actually driving your risk.
        """
        cols = [s for s in self.SYMBOLS if s in prices.columns]
        returns = prices[cols].pct_change().dropna()

        w = np.array([self.weights.get(s, 0) for s in cols])
        w = w / w.sum()

        cov = returns.cov().values * TRADING_DAYS          # Annualized covariance
        port_var = w @ cov @ w
        port_vol = np.sqrt(port_var)

        marginal = cov @ w                                  # Marginal contribution vector
        component_vol = w * marginal                        # Absolute contribution
        pct_contribution = component_vol / port_var         # % of total variance

        rows = []
        for i, symbol in enumerate(cols):
            rows.append({
                "portfolio_id": self.portfolio_id,
                "symbol": symbol,
                "weight": round(float(w[i]), 4),
                "asset_vol": round(float(returns[symbol].std() * np.sqrt(TRADING_DAYS)), 4),
                "marginal_contribution": round(float(marginal[i]), 6),
                "component_vol": round(float(component_vol[i]), 6),
                "pct_risk_contribution": round(float(pct_contribution[i]), 4),
            })

        df = pd.DataFrame(rows).sort_values("pct_risk_contribution", ascending=False)
        df["portfolio_vol"] = round(port_vol, 4)
        return df

    # ── Full Run + CSV Export ─────────────────────────────────────────────────

    def run(self, period: str = "2y") -> dict[str, Path]:
        logger.info("Running volatility analysis | portfolio={}", self.portfolio_id)

        prices   = load_price_data(period=period)
        port_ret = self._portfolio_returns(prices)

        realized  = self.realized_volatility(port_ret)
        ewma      = self.ewma_volatility(port_ret)
        regime    = self.volatility_regime(realized)
        term      = self.vol_term_structure(port_ret)
        contrib   = self.asset_vol_contribution(prices)

        today = datetime.today().strftime("%Y%m%d")
        pid   = self.portfolio_id.replace("-", "_").lower()

        exports = {
            "realized_volatility":      realized,
            "ewma_volatility":          ewma,
            "volatility_regime":        regime,
            "vol_term_structure":       term,
            "asset_vol_contribution":   contrib,
        }

        outputs = {}
        for name, df in exports.items():
            if df.empty:
                logger.warning("Skipping empty: {}", name)
                continue
            path = OUTPUT_DIR / f"{pid}_{name}_{today}.csv"
            df.to_csv(path, index=False)
            outputs[name] = path
            logger.info("Exported {} → {} | rows={}", name, path, len(df))

        logger.info("Volatility analysis complete | {} files written", len(outputs))
        return outputs


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for portfolio_id in ["PORT-001", "PORT-002", "PORT-003"]:
        engine = VolatilityEngine(portfolio_id=portfolio_id)
        outputs = engine.run(period="2y")
        for name, path in outputs.items():
            print(f"  {name}: {path}")