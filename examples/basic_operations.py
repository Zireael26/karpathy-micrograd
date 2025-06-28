"""
Basic operations example for micrograd.

This example demonstrates the core functionality of the Value class
and automatic differentiation.
"""

from micrograd.engine import Value
import math


def basic_arithmetic():
    """Demonstrate basic arithmetic operations."""
    print("=== Basic Arithmetic Operations ===")
    
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    
    # Addition
    c = a + b
    c.label = 'c=a+b'
    print(f"a + b = {c.data}")
    
    # Multiplication
    d = a * b
    d.label = 'd=a*b'
    print(f"a * b = {d.data}")
    
    # More complex expression
    e = a * b + b**2
    e.label = 'e=a*b+b^2'
    print(f"a * b + b^2 = {e.data}")
    
    # Compute gradients
    e.backward()
    print(f"de/da = {a.grad}")
    print(f"de/db = {b.grad}")
    print()


def activation_functions():
    """Demonstrate activation functions."""
    print("=== Activation Functions ===")
    
    x = Value(0.5, label='x')
    
    # Tanh
    y_tanh = x.tanh()
    y_tanh.label = 'tanh(x)'
    print(f"tanh({x.data}) = {y_tanh.data:.4f}")
    
    # ReLU
    x_neg = Value(-1.0, label='x_neg')
    x_pos = Value(1.0, label='x_pos')
    
    relu_neg = x_neg.relu()
    relu_pos = x_pos.relu()
    print(f"relu({x_neg.data}) = {relu_neg.data}")
    print(f"relu({x_pos.data}) = {relu_pos.data}")
    
    # Sigmoid
    sigmoid_out = x.sigmoid()
    sigmoid_out.label = 'sigmoid(x)'
    print(f"sigmoid({x.data}) = {sigmoid_out.data:.4f}")
    
    # Compute gradients for sigmoid
    sigmoid_out.backward()
    print(f"d(sigmoid)/dx = {x.grad:.4f}")
    print()


def chain_rule_example():
    """Demonstrate chain rule with a complex expression."""
    print("=== Chain Rule Example ===")
    
    # f(x) = sin(x^2) approximated as tanh(x^2)
    x = Value(1.5, label='x')
    
    x_squared = x**2
    x_squared.label = 'x^2'
    
    f = x_squared.tanh()
    f.label = 'tanh(x^2)'
    
    print(f"x = {x.data}")
    print(f"x^2 = {x_squared.data}")
    print(f"tanh(x^2) = {f.data:.4f}")
    
    # Compute gradient
    f.backward()
    print(f"df/dx = {x.grad:.4f}")
    
    # Compare with analytical solution
    # d/dx[tanh(x^2)] = sech^2(x^2) * 2x = (1 - tanh^2(x^2)) * 2x
    analytical = (1 - f.data**2) * 2 * x.data
    print(f"Analytical df/dx = {analytical:.4f}")
    print(f"Difference = {abs(x.grad - analytical):.6f}")
    print()


def gradient_accumulation():
    """Demonstrate gradient accumulation."""
    print("=== Gradient Accumulation ===")
    
    # Use the same variable multiple times
    a = Value(2.0, label='a')
    
    # Expression where 'a' appears multiple times
    b = a + a  # 2*a
    c = a * a  # a^2
    d = b + c  # 2*a + a^2
    d.label = '2*a + a^2'
    
    print(f"a = {a.data}")
    print(f"2*a + a^2 = {d.data}")
    
    # Compute gradient
    d.backward()
    print(f"d(2*a + a^2)/da = {a.grad}")
    
    # Analytical: d/da[2*a + a^2] = 2 + 2*a = 2 + 2*2 = 6
    analytical = 2 + 2 * a.data
    print(f"Analytical gradient = {analytical}")
    print(f"Match: {abs(a.grad - analytical) < 1e-6}")
    print()


def exponential_and_log():
    """Demonstrate exponential and logarithm functions."""
    print("=== Exponential and Logarithm ===")
    
    x = Value(1.0, label='x')
    
    # Exponential
    exp_x = x.exp()
    exp_x.label = 'exp(x)'
    print(f"exp({x.data}) = {exp_x.data:.4f}")
    print(f"Expected: {math.e:.4f}")
    
    # Logarithm
    y = Value(math.e, label='y')
    log_y = y.log()
    log_y.label = 'log(y)'
    print(f"log({y.data:.4f}) = {log_y.data:.4f}")
    print(f"Expected: 1.0")
    
    # Gradient of exp
    exp_x.backward()
    print(f"d(exp(x))/dx = {x.grad:.4f}")
    print(f"Expected (exp(x)): {math.e:.4f}")
    print()


if __name__ == "__main__":
    basic_arithmetic()
    activation_functions()
    chain_rule_example()
    gradient_accumulation()
    exponential_and_log()
    
    print("=== Summary ===")
    print("This example demonstrated:")
    print("1. Basic arithmetic operations (+, -, *, /, **)")
    print("2. Activation functions (tanh, relu, sigmoid)")
    print("3. Chain rule application")
    print("4. Gradient accumulation when variables are reused")
    print("5. Exponential and logarithm functions")
    print("6. Automatic differentiation (backpropagation)")
