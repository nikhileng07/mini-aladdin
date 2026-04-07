"""
portfolio_returns.py
--------------------
Computes portfolio-level return analytics:
- Daily / monthly / annual returns
- Cumulative return curve
- Benchmark-relative (active) returns
- Attribution by asset
Reads from S3/MinIO with DuckDB fallback.
Outputs to CSV.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime

# ── Storage Reader ────────────────────────────────────────────────────────────

def load_price_data(period: str = "2y") -> pd.DataFrame:
    """
    Try S3/MinIO first, fall back to DuckDB, fall back to yfinance live pull.
    Returns a DataFrame of daily close prices, columns = symbols.
    """
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS", "BAC"]
    BENCHMARK = "SPY"

    # ── Try S3 ────────────────────────────────────────────────────────────────
    try:
        import awswrangler as wr
        from config.settings import settings
        wr.config.s3_endpoint_url = settings.MINIO_ENDPOINT
        s3_path = f"s3://{settings.MINIO_BUCKET}/processed/returns/"
        df = wr.s3.read_parquet(path=s3_path, dataset=True)
        logger.info("Loaded price data from S3 | rows={}", len(df))
        return df
    except Exception as e:
        logger.warning("S3 unavailable ({}), falling back to DuckDB", str(e)[:60])

    # ── Try DuckDB ────────────────────────────────────────────────────────────
    try:
        import duckdb
        from config.settings import settings
        conn = duckdb.connect(settings.DUCKDB_PATH)
        df = conn.execute("""
            SELECT symbol, DATE(timestamp) as date, price
            FROM stock_ticks
            ORDER BY symbol, date
        """).df()
        conn.close()
        if not df.empty:
            df = df.pivot(index="date", columns="symbol", values="price")
            df.index = pd.to_datetime(df.index)
            logger.info("Loaded price data from DuckDB | rows={}", len(df))
            return df
        raise ValueError("DuckDB returned empty data")
    except Exception as e:
        logger.warning("DuckDB unavailable ({}), falling back to yfinance", str(e)[:60])

    # ── Fallback: yfinance ────────────────────────────────────────────────────
    import yfinance as yf
    logger.info("Pulling price data from yfinance | period={}", period)
    prices = yf.download(SYMBOLS + [BENCHMARK], period=period, progress=False)["Close"]
    return prices.dropna()


def load_transactions(portfolio_id: str) -> pd.DataFrame:
    """Load transaction history. S3 → DuckDB → empty fallback."""
    try:
        import duckdb
        from config.settings import settings
        conn = duckdb.connect(settings.DUCKDB_PATH)
        df = conn.execute("""
            SELECT * FROM transactions
            WHERE portfolio_id = ?
            ORDER BY timestamp
        """, [portfolio_id]).df()
        conn.close()
        if not df.empty:
            return df
    except Exception as e:
        logger.warning("Could not load transactions: {}", str(e)[:60])
    return pd.DataFrame()


# ── Returns Engine ────────────────────────────────────────────────────────────

class PortfolioReturnsEngine:

    SYMBOLS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS", "BAC"]
    BENCHMARK  = "SPY"
    OUTPUT_DIR = Path("data/processed/analytics")

    def __init__(self, portfolio_id: str, weights: dict = None):
        self.portfolio_id = portfolio_id
        self.weights = weights or {s: 1 / len(self.SYMBOLS) for s in self.SYMBOLS}
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _weighted_returns(self, returns: pd.DataFrame) -> pd.Series:
        w = np.array([self.weights.get(s, 0) for s in self.SYMBOLS])
        w = w / w.sum()
        cols = [s for s in self.SYMBOLS if s in returns.columns]
        return (returns[cols] * w[:len(cols)]).sum(axis=1)

    # ── Daily Returns ─────────────────────────────────────────────────────────

    def daily_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Simple daily returns for portfolio and benchmark."""
        returns = prices.pct_change().dropna()
        port_ret = self._weighted_returns(returns)
        bench_ret = returns[self.BENCHMARK] if self.BENCHMARK in returns.columns else pd.Series(dtype=float)

        df = pd.DataFrame({
            "date": port_ret.index,
            "portfolio_id": self.portfolio_id,
            "portfolio_return": port_ret.values,
            "benchmark_return": bench_ret.reindex(port_ret.index).values,
            "active_return": (port_ret - bench_ret.reindex(port_ret.index)).values,
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # ── Monthly Returns ───────────────────────────────────────────────────────

    def monthly_returns(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Compound daily returns into monthly buckets."""
        df = daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)

        monthly = (
            df.groupby("month")
            .agg(
                portfolio_return=("portfolio_return", lambda x: (1 + x).prod() - 1),
                benchmark_return=("benchmark_return", lambda x: (1 + x).prod() - 1),
                active_return=("active_return",   lambda x: (1 + x).prod() - 1),
                trading_days=("portfolio_return", "count"),
            )
            .reset_index()
        )
        monthly["portfolio_id"] = self.portfolio_id
        return monthly

    # ── Annual Returns ────────────────────────────────────────────────────────

    def annual_returns(self, daily: pd.DataFrame) -> pd.DataFrame:
        df = daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        annual = (
            df.groupby("year")
            .agg(
                portfolio_return=("portfolio_return", lambda x: (1 + x).prod() - 1),
                benchmark_return=("benchmark_return", lambda x: (1 + x).prod() - 1),
                active_return=("active_return",   lambda x: (1 + x).prod() - 1),
                trading_days=("portfolio_return", "count"),
            )
            .reset_index()
        )
        annual["portfolio_id"] = self.portfolio_id
        return annual

    # ── Cumulative Returns ────────────────────────────────────────────────────

    def cumulative_returns(self, daily: pd.DataFrame) -> pd.DataFrame:
        df = daily.copy().sort_values("date")
        df["cum_portfolio"] = (1 + df["portfolio_return"]).cumprod() - 1
        df["cum_benchmark"] = (1 + df["benchmark_return"]).cumprod() - 1
        df["cum_active"]    = df["cum_portfolio"] - df["cum_benchmark"]
        return df[["date", "portfolio_id", "cum_portfolio", "cum_benchmark", "cum_active"]]

    # ── Attribution ───────────────────────────────────────────────────────────

    def return_attribution(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Brinson attribution — how much each asset contributed to total return.
        Contribution = weight * asset_return
        """
        returns = prices[self.SYMBOLS].pct_change().dropna() if all(
            s in prices.columns for s in self.SYMBOLS
        ) else pd.DataFrame()

        if returns.empty:
            logger.warning("Cannot compute attribution — missing symbol data")
            return pd.DataFrame()

        total_return = (1 + returns).prod() - 1
        rows = []
        for symbol in self.SYMBOLS:
            if symbol not in total_return.index:
                continue
            w = self.weights.get(symbol, 0)
            asset_ret = float(total_return[symbol])
            contribution = w * asset_ret
            rows.append({
                "portfolio_id": self.portfolio_id,
                "symbol": symbol,
                "weight": round(w, 4),
                "asset_total_return": round(asset_ret, 4),
                "contribution": round(contribution, 4),
            })

        df = pd.DataFrame(rows).sort_values("contribution", ascending=False)
        df["contribution_pct"] = (df["contribution"] / df["contribution"].abs().sum()).round(4)
        return df

    # ── Full Run + CSV Export ─────────────────────────────────────────────────

    def run(self, period: str = "2y") -> dict[str, Path]:
        """Compute all return metrics and export to CSV. Returns dict of output paths."""
        logger.info("Running returns analysis | portfolio={} | period={}", self.portfolio_id, period)

        prices  = load_price_data(period=period)
        daily   = self.daily_returns(prices)
        monthly = self.monthly_returns(daily)
        annual  = self.annual_returns(daily)
        cumul   = self.cumulative_returns(daily)
        attrib  = self.return_attribution(prices)

        today   = datetime.today().strftime("%Y%m%d")
        pid     = self.portfolio_id.replace("-", "_").lower()
        outputs = {}

        exports = {
            "daily_returns":        daily,
            "monthly_returns":      monthly,
            "annual_returns":       annual,
            "cumulative_returns":   cumul,
            "return_attribution":   attrib,
        }

        for name, df in exports.items():
            if df.empty:
                logger.warning("Skipping empty export: {}", name)
                continue
            path = self.OUTPUT_DIR / f"{pid}_{name}_{today}.csv"
            df.to_csv(path, index=False)
            outputs[name] = path
            logger.info("Exported {} → {} | rows={}", name, path, len(df))

        logger.info("Returns analysis complete | {} files written", len(outputs))
        return outputs


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for portfolio_id in ["PORT-001", "PORT-002", "PORT-003"]:
        engine = PortfolioReturnsEngine(portfolio_id=portfolio_id)
        outputs = engine.run(period="2y")
        for name, path in outputs.items():
            print(f"  {name}: {path}")