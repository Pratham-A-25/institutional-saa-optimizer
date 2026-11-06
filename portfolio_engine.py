import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from typing import Optional, Dict, Tuple, List


class DataManager:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_log_returns(self) -> pd.DataFrame:
        prices = pd.DataFrame()
        # Download assets one by one to bypass yfinance's MultiIndex formatting bug
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            # Modern yfinance (>=0.2.31) auto-adjusts Close by default;
            # 'Adj Close' was removed in later versions.
            if 'Close' in df.columns:
                prices[ticker] = df['Close'].squeeze()
            else:
                prices[ticker] = df.iloc[:, 0].squeeze()

        prices = prices.ffill().dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns


class RiskEstimator:
    @staticmethod
    def compute_ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_ * 252  # Annualized
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)


class BlackLittermanModel:
    def __init__(self, cov_matrix: pd.DataFrame, market_caps: Dict[str, float], risk_aversion: float = 2.5):
        self.cov_matrix = cov_matrix.values
        self.tickers = list(cov_matrix.index)
        weights = np.array([market_caps[t] for t in self.tickers])
        self.market_weights = weights / np.sum(weights)
        self.delta = risk_aversion
        self.pi = self.delta * (self.cov_matrix @ self.market_weights)

    def compute_posterior_returns(self, P: np.ndarray, Q: np.ndarray, confidences: List[float]) -> pd.Series:
        tau = 0.05
        omega = np.diag([tau * (P[i] @ self.cov_matrix @ P[i].T) / confidences[i] for i in range(len(Q))])

        M_inverse = np.linalg.inv(np.linalg.inv(tau * self.cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
        expected_returns = M_inverse @ (np.linalg.inv(tau * self.cov_matrix) @ self.pi + P.T @ np.linalg.inv(omega) @ Q)

        return pd.Series(expected_returns, index=self.tickers)


class InstitutionalOptimizer:
    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame):
        self.mu = expected_returns.values
        self.sigma = cov_matrix.values
        self.tickers = list(expected_returns.index)
        self.n_assets = len(self.tickers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_weights(self, raw_weights: np.ndarray) -> np.ndarray:
        """Round tiny weights to zero and re-normalise to sum to 1."""
        w = np.round(raw_weights, 4)
        w[np.abs(w) < 1e-4] = 0.0
        total = np.sum(w)
        if total > 0:
            w /= total
        else:
            w = np.ones(self.n_assets) / self.n_assets
        return w

    def _compute_results(
        self,
        clean_weights: np.ndarray,
        carbon_intensities: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Shared post-solve: weights series, risk attribution, metrics dict."""
        weights_series = pd.Series(clean_weights, index=self.tickers)

        # Euler risk decomposition:
        #   MCR_i  = (Σw)_i / σ_p          (marginal contribution to volatility)
        #   CCR_i  = w_i × MCR_i            (component contribution; sums to σ_p)
        #   PRC_i  = CCR_i / σ_p            (percentage risk contribution; sums to 1)
        port_vol = np.sqrt(clean_weights.T @ self.sigma @ clean_weights)
        if port_vol > 0:
            mcr = (self.sigma @ clean_weights) / port_vol
            ccr = clean_weights * mcr
            pct_risk = ccr / port_vol
        else:
            pct_risk = np.zeros(self.n_assets)

        port_return = float(clean_weights @ self.mu)
        metrics = {
            "Expected Return": port_return,
            "Annualized Volatility": float(port_vol),
            "Sharpe Ratio": float((port_return - risk_free_rate) / port_vol) if port_vol > 0 else 0.0,
        }

        if carbon_intensities is not None:
            c_vec = np.array([carbon_intensities[t] for t in self.tickers])
            metrics["Portfolio Carbon Intensity"] = float(clean_weights @ c_vec)

        return weights_series, pd.Series(pct_risk, index=self.tickers), metrics

    # ------------------------------------------------------------------
    # Optimisation objectives
    # ------------------------------------------------------------------

    def optimize_min_variance(
        self,
        target_return: Optional[float] = None,
        max_weight: float = 0.50,
        carbon_intensities: Optional[Dict[str, float]] = None,
        max_portfolio_carbon: Optional[float] = None,
        risk_free_rate: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Minimum-variance portfolio (optionally with a return floor)."""
        w = cp.Variable(self.n_assets)
        constraints = [cp.sum(w) == 1.0, w >= 0.0, w <= max_weight]
        if target_return is not None:
            constraints.append(self.mu @ w >= target_return)
        if carbon_intensities is not None and max_portfolio_carbon is not None:
            c_vec = np.array([carbon_intensities[t] for t in self.tickers])
            constraints.append(c_vec @ w <= max_portfolio_carbon)

        problem = cp.Problem(cp.Minimize(cp.quad_form(w, self.sigma, assume_PSD=True)), constraints)
        problem.solve(solver=cp.OSQP)

        if w.value is None:
            raise ValueError(f"Min-variance optimisation failed. Status: {problem.status}")

        return self._compute_results(self._clean_weights(w.value), carbon_intensities, risk_free_rate)

    def optimize_max_sharpe(
        self,
        risk_free_rate: float = 0.0,
        max_weight: float = 0.50,
        carbon_intensities: Optional[Dict[str, float]] = None,
        max_portfolio_carbon: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Maximum Sharpe-ratio portfolio (Cornuéjols–Tütüncü transformation).

        Transforms the fractional objective into a convex QP by setting
        y = w / ((μ−r_f)'w) and minimising y'Σy subject to (μ−r_f)'y = 1.
        Constraints on normalised weights w translate linearly to y.
        Falls back to min-variance when no asset has positive excess return.
        """
        excess_mu = self.mu - risk_free_rate

        if np.max(excess_mu) <= 0:
            return self.optimize_min_variance(
                max_weight=max_weight,
                carbon_intensities=carbon_intensities,
                max_portfolio_carbon=max_portfolio_carbon,
                risk_free_rate=risk_free_rate,
            )

        y = cp.Variable(self.n_assets)
        k = cp.sum(y)

        constraints = [excess_mu @ y == 1, y >= 0, y <= max_weight * k]

        if carbon_intensities is not None and max_portfolio_carbon is not None:
            c_vec = np.array([carbon_intensities[t] for t in self.tickers])
            constraints.append(c_vec @ y <= max_portfolio_carbon * k)

        problem = cp.Problem(cp.Minimize(cp.quad_form(y, self.sigma, assume_PSD=True)), constraints)
        problem.solve(solver=cp.OSQP)

        if y.value is None:
            # Infeasible under constraints — fall back to min-variance
            return self.optimize_min_variance(
                max_weight=max_weight,
                carbon_intensities=carbon_intensities,
                max_portfolio_carbon=max_portfolio_carbon,
                risk_free_rate=risk_free_rate,
            )

        normalised = y.value / np.sum(y.value)
        return self._compute_results(self._clean_weights(normalised), carbon_intensities, risk_free_rate)

    def optimize_risk_parity(
        self,
        max_weight: float = 0.50,
        risk_free_rate: float = 0.0,
        carbon_intensities: Optional[Dict[str, float]] = None,
        max_portfolio_carbon: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Risk-parity portfolio (Spinu 2013 log-barrier formulation).

        Minimises  ½ w'Σw − (1/n) Σ log(wᵢ)  then normalises to sum-to-one.
        The log barrier drives the solution toward equal risk contribution.
        Constraints on normalised weights are expressed relative to sum(w).
        """
        w = cp.Variable(self.n_assets)
        budget = 1.0 / self.n_assets
        k = cp.sum(w)

        objective = 0.5 * cp.quad_form(w, self.sigma, assume_PSD=True) - budget * cp.sum(cp.log(w))
        constraints = [w >= 1e-6, w <= max_weight * k]

        if carbon_intensities is not None and max_portfolio_carbon is not None:
            c_vec = np.array([carbon_intensities[t] for t in self.tickers])
            constraints.append(c_vec @ w <= max_portfolio_carbon * k)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.SCS, max_iters=10000)

        if w.value is None:
            raise ValueError(f"Risk-parity optimisation failed. Status: {problem.status}")

        normalised = w.value / np.sum(w.value)
        return self._compute_results(self._clean_weights(normalised), carbon_intensities, risk_free_rate)

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def compute_efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: float = 0.50,
        risk_free_rate: float = 0.0,
        carbon_intensities: Optional[Dict[str, float]] = None,
        max_portfolio_carbon: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of (Return, Volatility, Sharpe) tracing the frontier."""
        w = cp.Variable(self.n_assets)
        base = [cp.sum(w) == 1, w >= 0, w <= max_weight]
        if carbon_intensities is not None and max_portfolio_carbon is not None:
            c_vec = np.array([carbon_intensities[t] for t in self.tickers])
            base.append(c_vec @ w <= max_portfolio_carbon)

        # Minimum-variance return
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, self.sigma, assume_PSD=True)), base)
        prob.solve(solver=cp.OSQP)
        min_ret = float(self.mu @ w.value) if w.value is not None else float(np.min(self.mu))

        # Maximum achievable return
        prob = cp.Problem(cp.Maximize(self.mu @ w), base)
        prob.solve(solver=cp.OSQP)
        max_ret = float(self.mu @ w.value) if w.value is not None else float(np.max(self.mu))

        targets = np.linspace(min_ret, max_ret * 0.995, n_points)
        frontier: List[Dict[str, float]] = []

        for target in targets:
            try:
                _, _, m = self.optimize_min_variance(
                    target_return=target,
                    max_weight=max_weight,
                    carbon_intensities=carbon_intensities,
                    max_portfolio_carbon=max_portfolio_carbon,
                    risk_free_rate=risk_free_rate,
                )
                frontier.append(m)
            except ValueError:
                continue

        return pd.DataFrame(frontier)


# ======================================================================
# Walk-forward backtester
# ======================================================================

class Backtester:
    """Out-of-sample walk-forward backtest with periodic rebalancing.

    Uses *equilibrium returns only* (no investor views) so that the backtest
    is free of look-ahead bias.  The same Ledoit-Wolf + BL-equilibrium +
    optimiser pipeline is re-run at every rebalance date using a rolling
    lookback window of historical returns.
    """

    @staticmethod
    def run_walkforward(
        returns: pd.DataFrame,
        benchmark_weights: Dict[str, float],
        optimization_mode: str = "min_variance",
        risk_free_rate: float = 0.0,
        max_weight: float = 0.50,
        lookback_days: int = 756,
        carbon_intensities: Optional[Dict[str, float]] = None,
        max_portfolio_carbon: Optional[float] = None,
    ) -> pd.DataFrame:
        tickers = list(returns.columns)
        bmk_w = np.array([benchmark_weights[t] for t in tickers])
        bmk_w = bmk_w / bmk_w.sum()

        # Monthly rebalance dates — last actual trading day of each month
        monthly_groups = returns.groupby([returns.index.year, returns.index.month])
        rebalance_dates = pd.DatetimeIndex([group.index[-1] for _, group in monthly_groups])

        first_valid_idx = min(lookback_days, len(returns) - 1)
        valid_dates = rebalance_dates[rebalance_dates >= returns.index[first_valid_idx]]

        if len(valid_dates) < 2:
            raise ValueError("Insufficient data for backtesting. Need more history or a shorter lookback.")

        all_dates = [valid_dates[0]]
        port_values = [1.0]
        bmk_values = [1.0]
        current_weights = bmk_w.copy()

        for i in range(len(valid_dates) - 1):
            rebal_date = valid_dates[i]
            next_rebal = valid_dates[i + 1]

            rebal_idx = returns.index.get_loc(rebal_date)
            lb_start = max(0, rebal_idx - lookback_days)
            lookback = returns.iloc[lb_start : rebal_idx + 1]

            if len(lookback) < 60:
                continue

            # Re-estimate and re-optimise
            try:
                cov = RiskEstimator.compute_ledoit_wolf_cov(lookback)
                bl = BlackLittermanModel(cov_matrix=cov, market_caps=benchmark_weights)
                eq_returns = pd.Series(bl.pi, index=tickers)
                opt = InstitutionalOptimizer(expected_returns=eq_returns, cov_matrix=cov)

                if optimization_mode == "max_sharpe":
                    w, _, _ = opt.optimize_max_sharpe(
                        risk_free_rate=risk_free_rate,
                        max_weight=max_weight,
                        carbon_intensities=carbon_intensities,
                        max_portfolio_carbon=max_portfolio_carbon,
                    )
                elif optimization_mode == "risk_parity":
                    w, _, _ = opt.optimize_risk_parity(
                        max_weight=max_weight,
                        risk_free_rate=risk_free_rate,
                        carbon_intensities=carbon_intensities,
                        max_portfolio_carbon=max_portfolio_carbon,
                    )
                else:
                    w, _, _ = opt.optimize_min_variance(
                        max_weight=max_weight,
                        risk_free_rate=risk_free_rate,
                        carbon_intensities=carbon_intensities,
                        max_portfolio_carbon=max_portfolio_carbon,
                    )

                current_weights = w.values
            except Exception:
                pass  # Keep previous weights on solver failure

            # Simulate daily P&L over the holding period
            hold_mask = (returns.index > rebal_date) & (returns.index <= next_rebal)
            for _, daily_log_ret in returns.loc[hold_mask].iterrows():
                daily_simple = np.exp(daily_log_ret.values) - 1  # cross-sectional aggregation needs simple returns
                port_values.append(port_values[-1] * (1 + current_weights @ daily_simple))
                bmk_values.append(bmk_values[-1] * (1 + bmk_w @ daily_simple))
                all_dates.append(daily_log_ret.name)

        return pd.DataFrame(
            {"Optimized Portfolio": port_values, "Benchmark": bmk_values},
            index=all_dates,
        )

    @staticmethod
    def compute_metrics(equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
        """Compute annualised performance statistics for each column."""
        daily_returns = equity_curve.pct_change().dropna()
        results = {}

        for col in equity_curve.columns:
            r = daily_returns[col]
            total_return = equity_curve[col].iloc[-1] / equity_curve[col].iloc[0] - 1
            n_years = len(r) / 252
            if n_years <= 0:
                continue
            cagr = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1.0
            vol = r.std() * np.sqrt(252)
            sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0

            cummax = equity_curve[col].cummax()
            drawdown = (equity_curve[col] - cummax) / cummax
            max_dd = drawdown.min()

            downside_vol = r[r < 0].std() * np.sqrt(252)
            sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

            results[col] = {
                "Total Return": total_return,
                "CAGR": cagr,
                "Annualized Volatility": vol,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Max Drawdown": max_dd,
                "Calmar Ratio": calmar,
            }

        return pd.DataFrame(results)
