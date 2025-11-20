/**
 * @file Brent.cpp
 * @brief Implementation of Brent's method for root finding
 * @author Your Name
 * @version 1.0.0
 * 
 * Brent's method combines the reliability of bisection with the speed
 * of inverse quadratic interpolation and secant method. It is considered
 * the best general-purpose root-finding algorithm.
 * 
 * Algorithm:
 * - Uses inverse quadratic interpolation when possible
 * - Falls back to secant method if interpolation is unsuitable
 * - Falls back to bisection if secant is making slow progress
 * - Maintains bracket guarantee for robustness
 * 
 * Convergence: Superlinear (~1.6 order)
 * Advantages: Fast + reliable, best of both worlds
 * Disadvantages: More complex than simple methods
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ode_solver {

/**
 * @brief Brent's method - hybrid root finding
 * 
 * Combines inverse quadratic interpolation, secant method, and bisection
 * for optimal performance. Considered the gold standard for bracketing methods.
 */
class BrentMethod : public BracketingMethod {
public:
    /**
     * @brief Constructor
     * @param f Function to find root of
     * @param a Left bracket endpoint
     * @param b Right bracket endpoint
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit BrentMethod(Function f,
                        double a,
                        double b,
                        double tolerance = 1e-6,
                        int max_iter = 100)
        : BracketingMethod(std::move(f),
                          BracketConfig{a, b, true},
                          tolerance,
                          max_iter) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Brent's Method");
            printVerbose("Bracket: [" + std::to_string(a_) + ", " + 
                        std::to_string(b_) + "]");
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Brent's method iteration");
        }
        
        // Initialize
        double a = a_;
        double b = b_;
        double c = b;  // Third point for interpolation
        double d = 0.0; // Previous iteration's step
        
        double f_a = f_a_;
        double f_b = f_b_;
        double f_c = f_b;
        
        // Ensure |f(b)| <= |f(a)| (b is best guess so far)
        if (std::abs(f_a) < std::abs(f_b)) {
            std::swap(a, b);
            std::swap(f_a, f_b);
        }
        
        bool used_bisection = true; // Flag for previous step type
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Check convergence
            double tolerance_current = 2.0 * std::numeric_limits<double>::epsilon() * 
                                      std::abs(b) + 0.5 * tolerance_;
            double midpoint = 0.5 * (a - b);
            
            if (std::abs(midpoint) <= tolerance_current || std::abs(f_b) < tolerance_) {
                last_root_ = b;
                last_error_ = std::abs(f_b);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(b));
                    printVerbose("f(root) = " + std::to_string(f_b));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                }
                
                recordError(last_error_);
                return b;
            }
            
            double s; // New point to try
            
            // Try inverse quadratic interpolation
            if (f_a != f_c && f_b != f_c) {
                // Three distinct function values, use IQI
                s = a * f_b * f_c / ((f_a - f_b) * (f_a - f_c)) +
                    b * f_a * f_c / ((f_b - f_a) * (f_b - f_c)) +
                    c * f_a * f_b / ((f_c - f_a) * (f_c - f_b));
                
                if (verbose_ && current_iterations_ % 5 == 0) {
                    printVerbose("Iteration " + std::to_string(current_iterations_) + 
                               ": Using inverse quadratic interpolation");
                }
            } else {
                // Use secant method
                s = b - f_b * (b - a) / (f_b - f_a);
                
                if (verbose_ && current_iterations_ % 5 == 0) {
                    printVerbose("Iteration " + std::to_string(current_iterations_) + 
                               ": Using secant method");
                }
            }
            
            // Decide whether to accept interpolation or use bisection
            bool use_bisection = false;
            
            // Condition 1: s must be between (3a+b)/4 and b
            double bound1 = (3.0 * a + b) / 4.0;
            double bound2 = b;
            if (bound1 > bound2) std::swap(bound1, bound2);
            
            if (s < bound1 || s > bound2) {
                use_bisection = true;
            }
            
            // Condition 2: Step size conditions
            if (!use_bisection) {
                if (used_bisection) {
                    // Previous step was bisection
                    if (std::abs(s - b) >= 0.5 * std::abs(b - c)) {
                        use_bisection = true;
                    }
                } else {
                    // Previous step was interpolation
                    if (std::abs(s - b) >= 0.5 * std::abs(c - d)) {
                        use_bisection = true;
                    }
                }
            }
            
            // Condition 3: Step too small for tolerance
            if (!use_bisection) {
                if (used_bisection) {
                    if (std::abs(b - c) < tolerance_current) {
                        use_bisection = true;
                    }
                } else {
                    if (std::abs(c - d) < tolerance_current) {
                        use_bisection = true;
                    }
                }
            }
            
            // Use bisection if conditions not met
            if (use_bisection) {
                s = 0.5 * (a + b);
                used_bisection = true;
                
                if (verbose_ && current_iterations_ % 5 == 0) {
                    printVerbose("  Falling back to bisection");
                }
            } else {
                used_bisection = false;
            }
            
            // Evaluate at new point
            double f_s = evaluateFunction(s);
            
            // Update points
            d = c;
            c = b;
            f_c = f_b;
            
            // Update bracket
            if (f_a * f_s < 0.0) {
                b = s;
                f_b = f_s;
            } else {
                a = s;
                f_a = f_s;
            }
            
            // Ensure |f(b)| <= |f(a)|
            if (std::abs(f_a) < std::abs(f_b)) {
                std::swap(a, b);
                std::swap(f_a, f_b);
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("  b = " + std::to_string(b) + 
                           ", f(b) = " + std::to_string(f_b));
            }
            
            recordError(std::abs(f_b));
        }
        
        // Maximum iterations reached
        last_root_ = b;
        last_error_ = std::abs(f_b);
        
        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: " + std::to_string(b));
            printVerbose("f(b) = " + std::to_string(f_b));
        }
        
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        
        throw ConvergenceException("Brent: Maximum iterations reached", info);
    }
    
    double iterate() override {
        // Single iteration not well-defined for Brent's method
        // since it maintains complex state
        throw std::logic_error(
            "Brent's method: Use solve() instead of iterate()");
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = (current_iterations_ < max_iterations_) ?
                      SolverStatus::SUCCESS : SolverStatus::MAX_ITERATIONS;
        info.message = "Brent's method";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Brent's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::SUPERLINEAR; // ~1.6 order
    }
    
    bool requiresDerivative() const override {
        return false;
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }

private:
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

} // namespace ode_solver