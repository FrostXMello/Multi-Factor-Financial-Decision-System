"""Advanced strategy research and portfolio analytics (M7)."""

from strategy_research.portfolio import portfolio_analytics
from strategy_research.regimes import classify_regimes, evaluate_returns_by_regime
from strategy_research.report import build_research_analytics_report
from strategy_research.robustness import walk_forward_robustness
from strategy_research.strategy_compare import compare_strategies

__all__ = [
    "classify_regimes",
    "evaluate_returns_by_regime",
    "portfolio_analytics",
    "compare_strategies",
    "walk_forward_robustness",
    "build_research_analytics_report",
]
