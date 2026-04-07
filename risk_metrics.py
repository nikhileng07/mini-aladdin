"""
risk_metrics.py
---------------
Core portfolio risk analytics — zero empyrical dependency.
Pure numpy + scipy only.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252

@dataclass
class RiskReport:
    portfolio_id: str
    as_of_date: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_volatility: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95_historical: float = 0.0
    var_99_historical: float = 0.0
    var_95_parametric: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation_to_benchmark: float = 0.0
    herfindahl_index: float = 0.0
    effective_n_stocks: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__


class RiskEngine:

    TRADING_DAYS   = 252
    RISK_FREE_RATE = 0.05

    def __init__(self, risk_free_rate: float = None):
        self.rf       = risk_free_rate or self.RISK_FREE_RATE
        self.rf_daily = (1 + self.rf) ** (1 / self.TRADING_DAYS) - 1

    def total_return(self, returns: pd.Series) -> float:
        return float((1 + returns).prod() - 1)

    def annualized_return(self, returns: pd.Series) -> float:
        n_days = len(returns)
        total  = (1 + returns).prod()
        return float(total ** (self.TRADING_DAYS / n_days) - 1)

    def annualized_volatility(self, returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(self.TRADING_DAYS))

    def sharpe_ratio(self, returns: pd.Series) -> float:
        excess = returns - self.rf_daily
        if returns.std() == 0:
            return 0.0
        return float((excess.mean() / returns.std()) * np.sqrt(self.TRADING_DAYS))

    def sortino_ratio(self, returns: pd.Series) -> float:
        excess   = returns - self.rf_daily
        downside = returns[returns < self.rf_daily]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        downside_std = downside.std() * np.sqrt(self.TRADING_DAYS)
        ann_excess   = excess.mean() * self.TRADING_DAYS
        return float(ann_excess / downside_std)

    def calmar_ratio(self, returns: pd.Series) -> float:
        ann_ret = self.annualized_return(returns)
        max_dd  = abs(self.max_drawdown(returns))
        if max_dd == 0:
            return 0.0
        return float(ann_ret / max_dd)

    def information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        active = returns - benchmark_returns.reindex(returns.index).fillna(0)
        if active.std() == 0:
            return 0.0
        return float((active.mean() / active.std()) * np.sqrt(self.TRADING_DAYS))

    def max_drawdown(self, returns: pd.Series) -> float:
        cum      = (1 + returns).cumprod()
        peak     = cum.cummax()
        drawdown = (cum - peak) / peak
        return float(drawdown.min())

    def drawdown_series(self, returns: pd.Series) -> pd.Series:
        cum  = (1 + returns).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    def max_drawdown_duration(self, returns: pd.Series) -> int:
        dd      = self.drawdown_series(returns)
        in_dd   = (dd < 0).astype(int)
        max_dur = cur_dur = 0
        for v in in_dd:
            if v:
                cur_dur += 1
                max_dur  = max(max_dur, cur_dur)
            else:
                cur_dur  = 0
        return max_dur

    def var_historical(self, returns: pd.Series, confidence: float = 0.95) -> float:
        return float(np.percentile(returns, (1 - confidence) * 100))

    def var_parametric(self, returns: pd.Series, confidence: float = 0.95) -> float:
        mu    = returns.mean()
        sigma = returns.std()
        z     = stats.norm.ppf(1 - confidence)
        return float(mu + z * sigma)

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        var  = self.var_historical(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def beta_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> tuple:
        aligned         = pd.concat([returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["port", "bench"]
        if len(aligned) < 10:
            return 0.0, 0.0
        slope, intercept, _, _, _ = stats.linregress(aligned["bench"], aligned["port"])
        return float(slope), float(intercept * self.TRADING_DAYS)

    def correlation_to_benchmark(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 10:
            return 0.0
        return float(aligned.corr().iloc[0, 1])

    def herfindahl_index(self, weights: pd.Series) -> float:
        return float((weights ** 2).sum())

    def effective_n_stocks(self, weights: pd.Series) -> float:
        hhi = self.herfindahl_index(weights)
        return float(1 / hhi) if hhi > 0 else 0.0

    def covariance_matrix(self, returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
        cov = returns_df.cov()
        return cov * self.TRADING_DAYS if annualize else cov

    def compute_full_report(
        self,
        portfolio_id: str,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        weights: pd.Series,
        as_of_date: str = None,
    ) -> RiskReport:
        from datetime import date
        as_of_date = as_of_date or str(date.today())
        logger.info("Computing risk report | portfolio={} | as_of={}", portfolio_id, as_of_date)
        beta, alpha = self.beta_alpha(portfolio_returns, benchmark_returns)
        report = RiskReport(
            portfolio_id             = portfolio_id,
            as_of_date               = as_of_date,
            total_return             = self.total_return(portfolio_returns),
            annualized_return        = self.annualized_return(portfolio_returns),
            daily_volatility         = float(portfolio_returns.std()),
            annualized_volatility    = self.annualized_volatility(portfolio_returns),
            sharpe_ratio             = self.sharpe_ratio(portfolio_returns),
            sortino_ratio            = self.sortino_ratio(portfolio_returns),
            calmar_ratio             = self.calmar_ratio(portfolio_returns),
            information_ratio        = self.information_ratio(portfolio_returns, benchmark_returns),
            max_drawdown             = self.max_drawdown(portfolio_returns),
            max_drawdown_duration    = self.max_drawdown_duration(portfolio_returns),
            var_95_historical        = self.var_historical(portfolio_returns, 0.95),
            var_99_historical        = self.var_historical(portfolio_returns, 0.99),
            var_95_parametric        = self.var_parametric(portfolio_returns, 0.95),
            cvar_95                  = self.cvar(portfolio_returns, 0.95),
            beta                     = beta,
            alpha                    = alpha,
            correlation_to_benchmark = self.correlation_to_benchmark(portfolio_returns, benchmark_returns),
            herfindahl_index         = self.herfindahl_index(weights),
            effective_n_stocks       = self.effective_n_stocks(weights),
        )
        logger.info(
            "Risk report complete | sharpe={:.2f} | max_dd={:.2%} | var95={:.2%}",
            report.sharpe_ratio, report.max_drawdown, report.var_95_historical
        )
        return report


if __name__ == "__main__":
    import yfinance as yf
    symbols   = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    benchmark = "SPY"
    prices    = yf.download(symbols + [benchmark], period="2y", progress=False)["Close"]
    returns   = prices.pct_change().dropna()
    weights          = pd.Series([0.25, 0.25, 0.20, 0.15, 0.15], index=symbols)
    portfolio_returns = (returns[symbols] * weights.values).sum(axis=1)
    benchmark_returns = returns[benchmark]
    engine = RiskEngine()
    report = engine.compute_full_report("PORT-001", portfolio_returns, benchmark_returns, weights)
    import json
    print(json.dumps(report.to_dict(), indent=2, default=str))