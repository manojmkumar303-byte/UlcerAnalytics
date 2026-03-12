import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Union

@dataclass
class RiskReport:
    """
    Assumptions:
    - Returns are independent (No autocorrelation modeled)
    - Liquidity penalty is a heuristic based on log-bankroll
    - Risk of Ruin is a simplified constant-stake model
    - Transaction costs/Slippage are NOT modeled in the metrics
    """
    sortino: float
    ulcer_index: float
    max_dd_duration: int
    recovery_factor: float
    prob_of_ruin: float
    log_convexity: float
    liq_tax: float

class RiskEngine:
    def __init__(self, periods_per_year: int = 252, risk_free_rate: float = 0.045):
        self.ppy = periods_per_year
        self.rfr = risk_free_rate

    def _sanitize(self, returns: list, bankroll: list):
        ret = np.asarray(returns, dtype=float)
        bh = np.asarray(bankroll, dtype=float)
        
        if np.any(np.isnan(ret)) or np.any(np.isnan(bh)):
            raise ValueError("Data Corruption: NaN detected in feeds.")
        if len(ret) != (len(bh) - 1):
            raise ValueError(f"Length mismatch: {len(ret)} returns vs {len(bh)} bankroll points.")
        if bh[0] <= 0:
            raise ValueError("Fatal Error: Starting bankroll must be positive.")
        return ret, bh

    def compute_metrics(self, returns: list, bankroll: list) -> RiskReport:
        ret, bh = self._sanitize(returns, bankroll)
        N = len(ret)
        
        # 1. Scaled Target Return
        target_ret = self.rfr / self.ppy
        
        # 2. Vectorized Drawdown & Duration (Starts-based check)
        peak = np.maximum.accumulate(bh)
        dd = (bh - peak) / peak
        underwater = dd < 0
        changes = np.diff(np.concatenate(([0], underwater.astype(int), [0])))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        
        # Robust duration check: ensures starts are tracked even if ends haven't occurred
        max_duration = int((ends - starts).max()) if len(starts) > 0 and len(ends) > 0 else 0

        # 3. Ulcer Index (Consistent Decimal Scale)
        ulcer_index = np.sqrt(np.mean(dd**2))

        # 4. Log-Equity Convexity
        log_bh = np.log(bh)
        log_convexity = np.mean(np.diff(log_bh, n=2)) if N > 1 else 0.0

        # 5. Risk of Ruin (Expectancy/Payoff Model)
        win_mask = ret > 0
        loss_mask = ret < 0
        win_rate = np.mean(win_mask)
        
        avg_win = np.mean(ret[win_mask]) if np.any(win_mask) else 0.0
        avg_loss = np.abs(np.mean(ret[loss_mask])) if np.any(loss_mask) else 0.0
        
        # Expectancy Edge calculation
        if avg_loss == 0:
            edge = 1.0 if avg_win > 0 else 0.0
        else:
            edge = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        
        # Stability Clamp & Ruin Formula
        edge_clamped = np.clip(edge, -0.99, 0.99)
        units = 10 # 10-unit risk buffer
        prob_ruin = ((1 - edge_clamped) / (1 + edge_clamped))**units if edge_clamped > 0 else 1.0

        # 6. Sortino with Log-Vol Liquidity Tax
        # Log-volatility is more stable for high-variance series
        log_vol = np.std(np.log1p(ret))
        liq_tax = 0.001 * np.log1p(np.mean(bh)/10000) * (1 + log_vol)
        
        excess = ret - target_ret
        downside_std = np.sqrt(np.mean(np.minimum(excess, 0)**2))
        
        if downside_std == 0:
            sortino = np.inf if np.mean(excess) > 0 else 0.0
        else:
            sortino = (np.mean(excess) - liq_tax) / downside_std

        # 7. Recovery Factor
        max_dd = np.abs(np.min(dd))
        total_ret = (bh[-1] - bh[0]) / bh[0] if bh[0] > 0 else 0.0
        recovery_factor = total_ret / max_dd if max_dd > 0 else np.inf

        return RiskReport(
            sortino=round(float(sortino), 4),
            ulcer_index=round(float(ulcer_index), 4),
            max_dd_duration=max_duration,
            recovery_factor=round(float(recovery_factor), 2),
            prob_of_ruin=round(float(prob_ruin), 4),
            log_convexity=round(float(log_convexity), 6),
            liq_tax=round(float(liq_tax), 6)
        )

# --- THE AUDITED PRODUCTION RUN ---
engine = RiskEngine()
# 5 returns, 6 bankroll points (t0-t5)
returns_feed = [0.02, -0.01, 0.03, -0.04, 0.05]
equity_curve = [1000, 1020, 1010, 1040, 1000, 1050]

try:
    report = engine.compute_metrics(returns_feed, equity_curve)
    print("--- FINAL INSTITUTIONAL RISK REPORT v20.0 ---")
    for key, value in asdict(report).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
except ValueError as e:
    print(f"CRITICAL ERROR: {e}")
