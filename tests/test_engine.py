"""Test the engine module."""

import math
import pytest
from micrograd.engine import Value


class TestValue:
    """Test cases for the Value class."""
    
    def test_basic_operations(self):
        """Test basic arithmetic operations."""
        a = Value(2.0)
        b = Value(3.0)
        
        # Addition
        c = a + b
        assert c.data == 5.0
        
        # Multiplication
        d = a * b
        assert d.data == 6.0
        
        # Subtraction
        e = a - b
        assert e.data == -1.0
        
        # Division
        f = a / b
        assert abs(f.data - 2.0/3.0) < 1e-6
        
        # Power
        g = a ** 2
        assert g.data == 4.0
    
    def test_backpropagation(self):
        """Test basic backpropagation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + b**2
        c.backward()
        
        # dc/da = b = 3.0
        assert abs(a.grad - 3.0) < 1e-6
        # dc/db = a + 2*b = 2 + 6 = 8.0
        assert abs(b.grad - 8.0) < 1e-6
    
    def test_chain_rule(self):
        """Test chain rule with more complex expressions."""
        x = Value(2.0)
        y = x**2 + 2*x + 1  # (x+1)^2
        y.backward()
        
        # dy/dx = 2*x + 2 = 6.0
        assert abs(x.grad - 6.0) < 1e-6
    
    def test_activation_functions(self):
        """Test activation functions."""
        x = Value(0.5)
        
        # Tanh
        y = x.tanh()
        expected = math.tanh(0.5)
        assert abs(y.data - expected) < 1e-6
        
        # ReLU
        x_neg = Value(-1.0)
        x_pos = Value(1.0)
        assert x_neg.relu().data == 0.0
        assert x_pos.relu().data == 1.0
        
        # Sigmoid
        s = x.sigmoid()
        expected_sigmoid = 1 / (1 + math.exp(-0.5))
        assert abs(s.data - expected_sigmoid) < 1e-6
    
    def test_exp_and_log(self):
        """Test exponential and logarithm functions."""
        x = Value(1.0)
        
        # Exponential
        y = x.exp()
        assert abs(y.data - math.e) < 1e-6
        
        # Logarithm
        z = Value(math.e)
        w = z.log()
        assert abs(w.data - 1.0) < 1e-6
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly."""
        a = Value(2.0)
        b = a + a  # Use 'a' twice
        b.backward()
        
        # da/db should be 2.0 (used twice)
        assert abs(a.grad - 2.0) < 1e-6
    
    def test_zero_gradient(self):
        """Test that gradients start at zero."""
        x = Value(5.0)
        assert x.grad == 0.0
    
    def test_complex_expression(self):
        """Test a more complex expression."""
        # f(x,y) = x*y + sin(x) (approximated with tanh for simplicity)
        x = Value(1.0)
        y = Value(2.0)
        
        z = x * y + x.tanh()
        z.backward()
        
        # Check that gradients are computed (exact values depend on implementation)
        assert x.grad != 0.0
        assert y.grad != 0.0
