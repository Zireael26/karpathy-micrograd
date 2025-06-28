"""
Utility functions for micrograd.

This module provides visualization tools, gradient checking, and other
helpful utilities for working with the micrograd library.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Set
from .engine import Value


def visualize_graph(root: Value, format: str = 'png', filename: str = None):
    """
    Visualize the computational graph using matplotlib.
    
    Args:
        root: The root Value node to visualize from
        format: Output format ('png', 'svg', etc.)
        filename: Optional filename to save the graph
    """
    try:
        import graphviz
        
        dot = graphviz.Digraph(format=format, graph_attr={'rankdir': 'LR'})
        
        # Build the graph
        nodes, edges = trace(root)
        
        for n in nodes:
            uid = str(id(n))
            # Create node label
            label = f"{{{n.label} | data {n.data:.4f} | grad {n.grad:.4f}}}"
            dot.node(name=uid, label=label, shape='record')
            
            if n._op:
                # Add operation node
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        if filename:
            dot.render(filename, cleanup=True)
        else:
            return dot
            
    except ImportError:
        print("Graphviz not available. Using simple matplotlib visualization.")
        _simple_graph_viz(root)


def _simple_graph_viz(root: Value):
    """Simple visualization using matplotlib when graphviz is not available."""
    nodes, edges = trace(root)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Simple layout - arrange nodes in a grid
    positions = {}
    levels = {}
    
    # Assign levels (depth from root)
    def assign_level(node, level=0):
        if node not in levels or levels[node] > level:
            levels[node] = level
            for child in node._prev:
                assign_level(child, level + 1)
    
    assign_level(root)
    
    # Group nodes by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)
    
    # Assign positions
    for level, nodes_at_level in level_groups.items():
        for i, node in enumerate(nodes_at_level):
            x = level * 3
            y = i * 2 - len(nodes_at_level)
            positions[node] = (x, y)
    
    # Draw nodes
    for node in nodes:
        x, y = positions[node]
        rect = patches.Rectangle((x-0.5, y-0.3), 1, 0.6, 
                               linewidth=1, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(x, y, f'{node.label}\n{node.data:.3f}\n{node.grad:.3f}', 
                ha='center', va='center', fontsize=8)
    
    # Draw edges
    for n1, n2 in edges:
        x1, y1 = positions[n1]
        x2, y2 = positions[n2]
        ax.arrow(x1+0.5, y1, x2-x1-1, y2-y1, head_width=0.1, head_length=0.1, 
                fc='black', ec='black')
    
    ax.set_xlim(-1, max(pos[0] for pos in positions.values()) + 1)
    ax.set_ylim(min(pos[1] for pos in positions.values()) - 1, 
                max(pos[1] for pos in positions.values()) + 1)
    ax.set_aspect('equal')
    ax.set_title('Computational Graph')
    plt.show()


def trace(root: Value):
    """
    Build a set of all nodes and edges in the computational graph.
    
    Args:
        root: The root Value node
        
    Returns:
        Tuple of (nodes, edges) sets
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def gradient_check(f, inputs: List[Value], epsilon: float = 1e-7):
    """
    Perform gradient checking using finite differences.
    
    Args:
        f: Function to check gradients for
        inputs: List of input Value objects
        epsilon: Small perturbation for finite difference
        
    Returns:
        Dictionary mapping input to (analytical_grad, numerical_grad, difference)
    """
    results = {}
    
    for i, inp in enumerate(inputs):
        # Store original value and gradient
        original_data = inp.data
        
        # Compute function value
        out = f()
        out.backward()
        analytical_grad = inp.grad
        
        # Reset gradients
        for param in inputs:
            param.grad = 0.0
        
        # Compute numerical gradient using finite differences
        inp.data = original_data + epsilon
        out_plus = f()
        
        inp.data = original_data - epsilon
        out_minus = f()
        
        numerical_grad = (out_plus.data - out_minus.data) / (2 * epsilon)
        
        # Restore original value
        inp.data = original_data
        
        # Compute difference
        diff = abs(analytical_grad - numerical_grad)
        
        results[f'input_{i}'] = {
            'analytical': analytical_grad,
            'numerical': numerical_grad,
            'difference': diff,
            'relative_error': diff / (abs(analytical_grad) + abs(numerical_grad) + 1e-8)
        }
    
    return results


def print_gradient_check(results: dict, tolerance: float = 1e-5):
    """
    Pretty print gradient checking results.
    
    Args:
        results: Results from gradient_check function
        tolerance: Tolerance for considering gradients as matching
    """
    print("Gradient Check Results:")
    print("-" * 60)
    print(f"{'Input':<10} {'Analytical':<12} {'Numerical':<12} {'Difference':<12} {'Status':<8}")
    print("-" * 60)
    
    for inp_name, result in results.items():
        status = "✓ PASS" if result['difference'] < tolerance else "✗ FAIL"
        print(f"{inp_name:<10} {result['analytical']:<12.6f} {result['numerical']:<12.6f} "
              f"{result['difference']:<12.2e} {status:<8}")
    
    print("-" * 60)


def plot_loss_curve(losses: List[float], title: str = "Training Loss"):
    """
    Plot the loss curve during training.
    
    Args:
        losses: List of loss values
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()


def summary_stats(values: List[Value]):
    """
    Compute summary statistics for a list of Value objects.
    
    Args:
        values: List of Value objects
        
    Returns:
        Dictionary with statistics
    """
    data_values = [v.data for v in values]
    grad_values = [v.grad for v in values]
    
    return {
        'data': {
            'mean': sum(data_values) / len(data_values),
            'min': min(data_values),
            'max': max(data_values),
            'std': (sum((x - sum(data_values)/len(data_values))**2 for x in data_values) / len(data_values))**0.5
        },
        'gradients': {
            'mean': sum(grad_values) / len(grad_values),
            'min': min(grad_values),
            'max': max(grad_values),
            'std': (sum((x - sum(grad_values)/len(grad_values))**2 for x in grad_values) / len(grad_values))**0.5
        }
    }
