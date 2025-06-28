"""Test the neural network module."""

import pytest
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


class TestNeuron:
    """Test cases for the Neuron class."""
    
    def test_neuron_creation(self):
        """Test neuron creation with different parameters."""
        n = Neuron(3)  # 3 inputs
        assert len(n.w) == 3
        assert n.b is not None
        assert n.activation == 'tanh'
        
        # Test with different activation
        n_relu = Neuron(2, activation='relu')
        assert n_relu.activation == 'relu'
    
    def test_neuron_forward_pass(self):
        """Test forward pass through a neuron."""
        n = Neuron(2, activation='linear')
        
        # Set known weights and bias
        n.w[0].data = 1.0
        n.w[1].data = 2.0
        n.b.data = 0.5
        
        # Test forward pass
        x = [1.0, 2.0]
        out = n(x)
        
        # Expected: 1*1 + 2*2 + 0.5 = 5.5
        assert abs(out.data - 5.5) < 1e-6
    
    def test_neuron_parameters(self):
        """Test parameter retrieval."""
        n = Neuron(3)
        params = n.parameters()
        assert len(params) == 4  # 3 weights + 1 bias


class TestLayer:
    """Test cases for the Layer class."""
    
    def test_layer_creation(self):
        """Test layer creation."""
        layer = Layer(3, 2)  # 3 inputs, 2 neurons
        assert len(layer.neurons) == 2
        assert all(len(n.w) == 3 for n in layer.neurons)
    
    def test_layer_forward_pass(self):
        """Test forward pass through a layer."""
        layer = Layer(2, 3, activation='linear')
        
        # Set known weights and biases
        for i, neuron in enumerate(layer.neurons):
            neuron.w[0].data = 1.0
            neuron.w[1].data = 1.0
            neuron.b.data = i  # Different bias for each neuron
        
        x = [1.0, 1.0]
        out = layer(x)
        
        assert len(out) == 3
        # Expected outputs: [2.0, 3.0, 4.0] (since bias is 0, 1, 2)
        for i, o in enumerate(out):
            assert abs(o.data - (2.0 + i)) < 1e-6
    
    def test_single_neuron_layer(self):
        """Test layer with single neuron returns scalar."""
        layer = Layer(2, 1)
        x = [1.0, 2.0]
        out = layer(x)
        
        # Should return a single Value, not a list
        assert isinstance(out, Value)
    
    def test_layer_parameters(self):
        """Test parameter retrieval from layer."""
        layer = Layer(2, 3)
        params = layer.parameters()
        assert len(params) == 9  # 3 neurons * (2 weights + 1 bias)


class TestMLP:
    """Test cases for the MLP class."""
    
    def test_mlp_creation(self):
        """Test MLP creation."""
        mlp = MLP(3, [4, 2, 1])  # 3 inputs, hidden layers of 4,2, output of 1
        assert len(mlp.layers) == 3
        
        # Check layer sizes
        assert len(mlp.layers[0].neurons) == 4
        assert len(mlp.layers[1].neurons) == 2
        assert len(mlp.layers[2].neurons) == 1
    
    def test_mlp_forward_pass(self):
        """Test forward pass through MLP."""
        mlp = MLP(2, [2, 1], activations='linear')
        
        x = [1.0, 2.0]
        out = mlp(x)
        
        # Should return a single Value (since output layer has 1 neuron)
        assert isinstance(out, Value)
    
    def test_mlp_parameters(self):
        """Test parameter retrieval from MLP."""
        mlp = MLP(3, [4, 2, 1])
        params = mlp.parameters()
        
        # Layer 1: 4 neurons * (3 weights + 1 bias) = 16
        # Layer 2: 2 neurons * (4 weights + 1 bias) = 10  
        # Layer 3: 1 neuron * (2 weights + 1 bias) = 3
        # Total: 29 parameters
        assert len(params) == 29
    
    def test_mlp_zero_grad(self):
        """Test gradient zeroing."""
        mlp = MLP(2, [2, 1])
        
        # Set some gradients
        for p in mlp.parameters():
            p.grad = 1.0
        
        # Zero gradients
        mlp.zero_grad()
        
        # Check all gradients are zero
        for p in mlp.parameters():
            assert p.grad == 0.0
    
    def test_mlp_training_step(self):
        """Test a basic training step."""
        mlp = MLP(2, [2, 1])
        
        # Sample training data
        x = [1.0, 2.0]
        target = 3.0
        
        # Forward pass
        pred = mlp(x)
        
        # Compute loss (simple squared error)
        loss = (pred - target) ** 2
        
        # Backward pass
        mlp.zero_grad()
        loss.backward()
        
        # Check that gradients are computed
        for p in mlp.parameters():
            # Gradients should be non-zero after backprop
            assert p.grad != 0.0 or p.data == 0.0  # Handle case where parameter is exactly 0
    
    def test_mlp_different_activations(self):
        """Test MLP with different activation functions."""
        # Single activation for all layers
        mlp1 = MLP(2, [2, 1], activations='relu')
        
        # Different activations for each layer
        mlp2 = MLP(2, [2, 1], activations=['tanh', 'sigmoid'])
        
        x = [1.0, 2.0]
        
        # Both should work without errors
        out1 = mlp1(x)
        out2 = mlp2(x)
        
        assert isinstance(out1, Value)
        assert isinstance(out2, Value)
    
    def test_mlp_parameter_update(self):
        """Test parameter update mechanism."""
        mlp = MLP(2, [1], activations='linear')
        
        # Get initial parameter values
        initial_params = [p.data for p in mlp.parameters()]
        
        # Set some gradients
        for p in mlp.parameters():
            p.grad = 1.0
        
        # Update parameters
        learning_rate = 0.01
        mlp.update_parameters(learning_rate)
        
        # Check that parameters have been updated
        updated_params = [p.data for p in mlp.parameters()]
        
        for initial, updated in zip(initial_params, updated_params):
            expected = initial - learning_rate * 1.0  # grad was 1.0
            assert abs(updated - expected) < 1e-6
