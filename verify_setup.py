#!/usr/bin/env python3
"""
Verification script to check that the micrograd environment is set up correctly.
"""

import sys
import platform

def main():
    print("=" * 50)
    print("MICROGRAD ENVIRONMENT VERIFICATION")
    print("=" * 50)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print()
    
    # Check required packages
    packages_to_check = [
        'numpy', 'matplotlib', 'jupyter', 'notebook', 
        'pytest', 'graphviz', 'micrograd'
    ]
    
    print("Package Verification:")
    print("-" * 20)
    
    for package in packages_to_check:
        try:
            if package == 'micrograd':
                import micrograd
                from micrograd.engine import Value
                from micrograd.nn import MLP
                version = "0.1.0"
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
            
            print(f"✓ {package}: {version}")
        except ImportError as e:
            print(f"✗ {package}: Import failed - {e}")
    
    print()
    
    # Test basic micrograd functionality
    print("Micrograd Functionality Test:")
    print("-" * 30)
    
    try:
        from micrograd.engine import Value
        
        # Test basic operations
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + Value(1.0)
        c.backward()
        
        print(f"✓ Basic operations: {a.data} * {b.data} + 1.0 = {c.data}")
        print(f"✓ Gradients computed: da/dc = {a.grad}, db/dc = {b.grad}")
        
        # Test neural network
        from micrograd.nn import MLP
        model = MLP(3, [4, 4, 1])
        x = [Value(1.0), Value(2.0), Value(3.0)]
        output = model(x)
        
        print(f"✓ Neural network: MLP(3, [4, 4, 1]) works")
        print(f"  Input: [1.0, 2.0, 3.0]")
        if isinstance(output, list) and len(output) > 0:
            out_val = output[0]
            if isinstance(out_val, Value):
                print(f"  Output: {out_val.data:.4f}")
            else:
                print(f"  Output: {out_val:.4f}")
        else:
            print(f"  Output: {output}")
        
    except Exception as e:
        print(f"✗ Micrograd test failed: {e}")
    
    print()
    print("=" * 50)
    print("SETUP VERIFICATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
