"""
Micrograd: A minimal neural network library with automatic differentiation.

This package provides:
- Value: A scalar-valued autograd engine
- Neural network layers (Neuron, Layer, MLP)
- Utility functions for visualization and debugging
"""

from .engine import Value
from .nn import Neuron, Layer, MLP

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = ["Value", "Neuron", "Layer", "MLP"]
