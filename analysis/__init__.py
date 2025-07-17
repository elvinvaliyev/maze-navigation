"""
Analysis package for maze navigation experiments.

This package contains modules for:
- Statistical analysis of agent performance
- Performance metrics calculation
- Comparative analysis between agents
- Visualization generation
"""

__version__ = "1.0.0"
__author__ = "Maze Navigation Team"

# Import main analysis classes
from .statistical_analysis import StatisticalAnalyzer
from .performance_metrics import PerformanceAnalyzer
from .comparative_analysis import ComparativeAnalyzer
from .visualization_engine import VisualizationEngine

__all__ = [
    'StatisticalAnalyzer',
    'PerformanceAnalyzer', 
    'ComparativeAnalyzer',
    'VisualizationEngine'
] 