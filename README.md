# Advanced ODE Solver Library

A comprehensive C++ library for solving Ordinary Differential Equations (ODEs) using both symbolic and numerical methods.

## Features

### Root Finding Methods
- Bisection Method
- Newton-Raphson Method
- Chebyshev's Method
- Muller's Method
- Multipoint Iteration Method
- Secant Method
- Regula-Falsi Method
- Brent's Method
- Steffensen's Method
- Ridder's Method
- Laguerre's Method (complex roots)
- Fixed Point Iteration

### ODE Solvers
- Runge-Kutta methods
- Adams-Bashforth method
- Bulirsch-Stoer algorithm
- Symbolic integrator for simple cases
- Adaptive step-size control
- Boundary value problem solver

### Symbolic Capabilities
- Basic symbolic manipulation
- ODE parsing (e.g., y′′+ sin(y)=0)
- Algebraic simplification
- Expression evaluation

### Root and Zero-Crossing Detection
- Hybrid bracketing methods
- Solution tracking
- Zero-crossing detection
- Interval validation

### Visualization
- CSV/Text output format
- Command-line plotting utilities
- Solution curve visualization

## Project Structure

```
ODE_solver/
├── src/
│   ├── core/           # Core solving algorithms
│   ├── symbolic/       # Symbolic manipulation
│   ├── numeric/        # Numerical methods
│   ├── visualization/  # Plotting utilities
│   └── utils/          # Helper functions
├── include/            # Header files
├── tests/              # Unit tests
├── examples/           # Usage examples
└── docs/              # Documentation
```

## Requirements

- C++17 or higher
- CMake 3.15+
- Boost Library (for symbolic computation)


## Building

```bash
mkdir build
cd build
cmake ..
make
```