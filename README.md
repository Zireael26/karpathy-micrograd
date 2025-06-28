# Micrograd - Neural Network from Scratch

A minimal implementation of a neural network library inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). This project implements automatic differentiation (backpropagation) and neural networks from scratch in Python.

## 🎯 Project Overview

Micrograd is a tiny Autograd engine that implements backpropagation over a dynamically built DAG (Directed Acyclic Graph). It's designed to be educational and help understand the internals of neural networks.

### Key Features
- ✅ Scalar-valued autograd engine
- ✅ Support for basic mathematical operations (+, -, *, /, **, exp, log, tanh, relu)
- ✅ Backpropagation through computational graphs
- ✅ Neural network layers (MLP - Multi-Layer Perceptron)
- ✅ Visualization of computational graphs
- ✅ Gradient checking utilities

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd micrograd
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Example

```python
from micrograd.engine import Value

# Create scalar values
a = Value(2.0)
b = Value(3.0)

# Perform operations
c = a * b + b**2
c.backward()  # Compute gradients

print(f"c = {c.data}")
print(f"dc/da = {a.grad}")
print(f"dc/db = {b.grad}")
```

#### Neural Network Example

```python
from micrograd.nn import MLP

# Create a 3-input, 2-hidden layer (4,4), 1-output MLP
model = MLP(3, [4, 4, 1])

# Sample input
x = [2.0, 3.0, -1.0]
y = model(x)

print(f"Output: {y}")
```

## 📁 Project Structure

```
micrograd/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py          # Core autograd engine (Value class)
│   ├── nn.py             # Neural network layers
│   └── utils.py          # Utility functions
├── tests/
│   ├── test_engine.py
│   ├── test_nn.py
│   └── test_utils.py
├── examples/
│   ├── basic_operations.py
│   ├── neural_network.py
│   └── visualization.py
├── notebooks/
│   └── micrograd.ipynb   # Main development notebook
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_engine.py

# Run with verbose output
pytest -v
```

## 📊 Examples

Check out the `examples/` directory for:
- Basic mathematical operations
- Neural network training examples
- Computational graph visualization

## 🎓 Learning Resources

This implementation follows along with:
- [Andrej Karpathy's micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original micrograd implementation
- The deep learning community for making education accessible

---

**Note**: This is an educational project designed to understand the fundamentals of neural networks and automatic differentiation. For production use, consider established libraries like PyTorch or TensorFlow.
