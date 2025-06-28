"""
Core autograd engine for micrograd.

This module implements the Value class which represents a scalar value
and its gradient in a computational graph.
"""

import math

class Value:
    """
    Stores a single scalar value and its gradient.
    
    This class supports basic mathematical operations and builds a computational
    graph for automatic differentiation (backpropagation).
    """
    
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = ''):
        """
        Initialize a Value object.
        
        Args:
            data: The scalar value
            _children: Child nodes in the computational graph (internal use)
            _op: Operation that created this value (internal use)  
            label: Optional label for visualization
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        """Addition operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        """Reverse add (for other + self)."""
        return self + other
    
    def __sub__(self, other):
        """Subtraction operation."""
        return self + (-other)
    
    def __rsub__(self, other):
        """Reverse subtract (for other - self)."""
        return other + (-self)
    
    def __mul__(self, other):
        """Multiplication operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        """Reverse multiply (for other * self)."""
        return self * other
    
    def __truediv__(self, other):
        """Division operation."""
        return self * other**-1
    
    def __rtruediv__(self, other):
        """Reverse divide (for other / self)."""
        return other * self**-1
    
    def __pow__(self, other):
        """Power operation."""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        """Negation operation."""
        return self * -1
    
    def exp(self):
        """Exponential function."""
        out = Value(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        """Natural logarithm."""
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        """Hyperbolic tangent activation function."""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """ReLU activation function."""
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        """Sigmoid activation function."""
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')
        
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        """
        Compute gradients for all values in the computational graph.
        
        This method performs backpropagation starting from this Value.
        """
        # Build topological order of all children in the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# Convenience functions
def exp(x):
    """Exponential function for Value objects."""
    return x.exp()

def log(x):
    """Natural logarithm for Value objects."""
    return x.log()

def tanh(x):
    """Hyperbolic tangent for Value objects."""
    return x.tanh()

def relu(x):
    """ReLU activation for Value objects."""
    return x.relu()

def sigmoid(x):
    """Sigmoid activation for Value objects."""
    return x.sigmoid()
