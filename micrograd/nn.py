"""
Neural network components for micrograd.

This module implements basic neural network layers using the Value class
from the engine module.
"""

import random
from typing import List, Union
from .engine import Value


class Neuron:
    """
    A single neuron with weights, bias, and activation function.
    """
    
    def __init__(self, nin: int, activation: str = 'tanh'):
        """
        Initialize a neuron.
        
        Args:
            nin: Number of input connections
            activation: Activation function ('tanh', 'relu', 'sigmoid', or 'linear')
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
    
    def __call__(self, x: List[Union[Value, float]]):
        """
        Forward pass through the neuron.
        
        Args:
            x: Input values
            
        Returns:
            Output value after applying weights, bias, and activation
        """
        # Ensure inputs are Value objects
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        if self.activation == 'tanh':
            out = act.tanh()
        elif self.activation == 'relu':
            out = act.relu()
        elif self.activation == 'sigmoid':
            out = act.sigmoid()
        elif self.activation == 'linear':
            out = act
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
        return out
    
    def parameters(self):
        """Return all parameters (weights and bias) of this neuron."""
        return self.w + [self.b]


class Layer:
    """
    A layer of neurons.
    """
    
    def __init__(self, nin: int, nout: int, activation: str = 'tanh'):
        """
        Initialize a layer.
        
        Args:
            nin: Number of input connections per neuron
            nout: Number of neurons in this layer
            activation: Activation function for all neurons in this layer
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x: List[Union[Value, float]]):
        """
        Forward pass through the layer.
        
        Args:
            x: Input values
            
        Returns:
            List of output values from all neurons, or single value if only one neuron
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        """Return all parameters of all neurons in this layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron (feedforward neural network).
    """
    
    def __init__(self, nin: int, nouts: List[int], activations: Union[str, List[str]] = 'tanh'):
        """
        Initialize an MLP.
        
        Args:
            nin: Number of input features
            nouts: List of layer sizes (number of neurons in each layer)
            activations: Activation function(s). Can be a single string for all layers
                        or a list of strings for each layer
        """
        sz = [nin] + nouts
        
        # Handle activations
        if isinstance(activations, str):
            activations = [activations] * len(nouts)
        elif len(activations) != len(nouts):
            raise ValueError("Number of activations must match number of layers")
        
        self.layers = [Layer(sz[i], sz[i+1], activations[i]) for i in range(len(nouts))]
    
    def __call__(self, x: List[Union[Value, float]]):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input values
            
        Returns:
            Output values from the final layer
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """Return all parameters of the MLP."""
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            p.grad = 0.0
    
    def get_loss(self, predictions: List[Value], targets: List[float], loss_type: str = 'mse'):
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Target values
            loss_type: Type of loss ('mse' for mean squared error)
            
        Returns:
            Loss value
        """
        if loss_type == 'mse':
            losses = [(pred - target)**2 for pred, target in zip(predictions, targets)]
            return sum(losses) / len(losses)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def update_parameters(self, learning_rate: float = 0.01):
        """
        Update parameters using gradient descent.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        for p in self.parameters():
            p.data -= learning_rate * p.grad
