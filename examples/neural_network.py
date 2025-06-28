"""
Neural network training example for micrograd.

This example demonstrates training a simple neural network on a toy dataset.
"""

import random
import matplotlib.pyplot as plt
from micrograd.engine import Value
from micrograd.nn import MLP


def generate_toy_data(n_samples=100):
    """Generate a simple toy dataset for binary classification."""
    data = []
    
    for _ in range(n_samples):
        # Generate random 2D points
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        
        # Simple decision boundary: x1 + x2 > 0
        # Add some noise to make it more interesting
        decision = x1 + x2 + 0.1 * random.gauss(0, 1)
        label = 1.0 if decision > 0 else -1.0
        
        data.append(([x1, x2], label))
    
    return data


def train_neural_network():
    """Train a neural network on the toy dataset."""
    print("=== Neural Network Training Example ===")
    
    # Generate dataset
    train_data = generate_toy_data(100)
    test_data = generate_toy_data(20)
    
    # Create model: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
    model = MLP(2, [16, 16, 1])
    print(f"Model created with {len(model.parameters())} parameters")
    
    # Training parameters
    learning_rate = 0.01
    epochs = 100
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        # Forward pass on all training data
        total_loss = Value(0.0)
        correct = 0
        
        for inputs, target in train_data:
            # Forward pass
            pred = model(inputs)
            
            # Compute loss (mean squared error)
            loss = (pred - target) ** 2
            total_loss = total_loss + loss
            
            # Check accuracy (for binary classification)
            predicted_class = 1.0 if pred.data > 0 else -1.0
            if predicted_class == target:
                correct += 1
        
        # Average loss
        avg_loss = total_loss * (1.0 / len(train_data))
        losses.append(avg_loss.data)
        
        # Backward pass
        model.zero_grad()
        avg_loss.backward()
        
        # Update parameters
        model.update_parameters(learning_rate)
        
        # Print progress
        if epoch % 10 == 0:
            accuracy = correct / len(train_data) * 100
            print(f"Epoch {epoch:3d}: Loss = {avg_loss.data:.4f}, Accuracy = {accuracy:.1f}%")
    
    # Final training accuracy
    final_accuracy = correct / len(train_data) * 100
    print(f"Final training accuracy: {final_accuracy:.1f}%")
    
    # Test accuracy
    correct_test = 0
    for inputs, target in test_data:
        pred = model(inputs)
        predicted_class = 1.0 if pred.data > 0 else -1.0
        if predicted_class == target:
            correct_test += 1
    
    test_accuracy = correct_test / len(test_data) * 100
    print(f"Test accuracy: {test_accuracy:.1f}%")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model, train_data, test_data


def visualize_decision_boundary(model, data):
    """Visualize the learned decision boundary."""
    print("Visualizing decision boundary...")
    
    # Create a grid of points
    import numpy as np
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    resolution = 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Make predictions on the grid
    Z = []
    for i in range(xx.shape[0]):
        row = []
        for j in range(xx.shape[1]):
            pred = model([xx[i, j], yy[i, j]])
            row.append(pred.data)
        Z.append(row)
    
    Z = np.array(Z)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Model Output')
    
    # Plot data points
    for inputs, target in data:
        color = 'red' if target == 1.0 else 'blue'
        marker = 'o' if target == 1.0 else 's'
        plt.scatter(inputs[0], inputs[1], c=color, marker=marker, s=50, alpha=0.7)
    
    plt.title('Decision Boundary Visualization')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class +1'),
                       Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Class -1')]
    plt.legend(handles=legend_elements)
    
    plt.show()


def regression_example():
    """Example of using the neural network for regression."""
    print("\n=== Regression Example ===")
    
    # Generate regression data: y = x^2 + noise
    train_data = []
    for _ in range(50):
        x = random.uniform(-2, 2)
        y = x**2 + 0.1 * random.gauss(0, 1)  # Add noise
        train_data.append(([x], y))
    
    # Create model: 1 input -> 10 hidden -> 10 hidden -> 1 output
    model = MLP(1, [10, 10, 1])
    
    # Training parameters
    learning_rate = 0.01
    epochs = 200
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = Value(0.0)
        
        for inputs, target in train_data:
            pred = model(inputs)
            loss = (pred - target) ** 2
            total_loss = total_loss + loss
        
        avg_loss = total_loss * (1.0 / len(train_data))
        losses.append(avg_loss.data)
        
        # Backward pass
        model.zero_grad()
        avg_loss.backward()
        model.update_parameters(learning_rate)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss.data:.4f}")
    
    # Visualize results
    import numpy as np
    x_test = np.linspace(-2, 2, 100)
    y_pred = []
    
    for x in x_test:
        pred = model([x])
        y_pred.append(pred.data)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    x_train = [d[0][0] for d in train_data]
    y_train = [d[1] for d in train_data]
    plt.scatter(x_train, y_train, alpha=0.6, label='Training Data')
    
    # Plot predictions
    plt.plot(x_test, y_pred, 'r-', linewidth=2, label='Neural Network')
    
    # Plot true function
    y_true = x_test**2
    plt.plot(x_test, y_true, 'g--', linewidth=2, label='True Function (x²)')
    
    plt.title('Neural Network Regression: Learning x²')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Train classification model
    model, train_data, test_data = train_neural_network()
    
    # Visualize decision boundary
    visualize_decision_boundary(model, train_data)
    
    # Regression example
    regression_example()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Creating and training a multi-layer perceptron")
    print("2. Binary classification on a toy dataset")
    print("3. Gradient descent optimization")
    print("4. Decision boundary visualization")
    print("5. Regression task (learning x²)")
    print("6. Loss curve monitoring")
