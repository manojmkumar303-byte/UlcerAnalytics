# UlcerAnalytics

Quantitative risk analytics engine for evaluating trading strategy stability.

## Features

- Sortino Ratio
- Ulcer Index
- Risk of Ruin
- Drawdown Duration
- Recovery Factor
- Equity Convexity
- Liquidity Modeling

## Example Usage

```python
from risk_engine import RiskEngine

engine = RiskEngine()

returns = [0.02,-0.01,0.03,-0.04,0.05]
equity = [1000,1020,1010,1040,1000,1050]

report = engine.compute_metrics(returns,equity)

print(report)
