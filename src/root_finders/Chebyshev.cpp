/**
 * @file Chebyshev.cpp
 * @brief Implementation of Chebyshev's method for root finding
 * @author Your Name
 * @version 1.0.0
 * 
 * Chebyshev's method is a third-order root-finding algorithm that uses
 * both first and second derivatives. It converges faster than Newton-Raphson
 * but requires computation of f''(x).
 * 
 * Algorithm:
 *   x_{n+1} = x_n - (1 + L_n/2) * f(x_n)/f'(x_n)
 *   where L_n = f(x_n) * f''(x_n) / [f'(x_n)]²
 * 
 * Convergence: Cubic (order 3)
 * Advantages: Very fast convergence near root
 * Disadvantages: Requires two derivatives, may diverge if initial guess poor
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ode_solver {

/**
 * @brief Chebyshev's method implementation
 * 
 * Third-order method using first and second derivatives.
 * Converges faster than Newton-Raphson when close to root.
 */
class ChebyshevMethod : public SecondDerivativeMethod {
public:
    /**
     * @brief Constructor
     * @param f Function f(x)
     * @param df First derivative f'(x)
     * @param ddf Second derivative f''(x)
     * @param x0 Initial guess
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit ChebyshevMethod(Function f,
                            DerivativeFunction df,
                            SecondDerivativeFunction ddf,
                            double x0,
                            double tolerance = 1e-6,
                            int max_iter = 100)
        : SecondDerivativeMethod(std::move(f), std::move(df), std::move(ddf),
                                x0, tolerance, max_iter) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Chebyshev's Method");
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
            
            double f0 = evaluateFunction(x0_);
            printVerbose("f(x0) = " + std::to_string(f0));
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        resetDerivativeEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Chebyshev iteration");
            printVerbose("Tolerance: " + std::to_string(tolerance_));
        }
        
        double x = x0_;
        double x_prev = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Evaluate function and derivatives at current point
            double f = evaluateFunction(x);
            double df = evaluateDerivative(x);
            double ddf = evaluateSecondDerivative(x);
            
            // Check for convergence
            if (std::abs(f) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(f);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(x));
                    printVerbose("f(root) = " + std::to_string(f));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                    printVerbose("Function evaluations: " + 
                               std::to_string(function_evaluations_));
                    printVerbose("Derivative evaluations: " + 
                               std::to_string(derivative_evaluations_));
                }
                
                recordError(last_error_);
                return x;
            }
            
            // Check for zero derivative (division by zero)
            if (isTooSmall(df)) {
                throw NumericalInstabilityException(
                    "Chebyshev: Derivative too small at x = " + std::to_string(x));
            }
            
            // Compute Chebyshev correction
            // L = f * f'' / (f')²
            double df_squared = df * df;
            double L = (f * ddf) / df_squared;
            
            // Chebyshev iteration: x_{n+1} = x_n - (1 + L/2) * f/f'
            double correction = f / df;
            double x_new = x - (1.0 + 0.5 * L) * correction;
            
            // Check for numerical issues
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Chebyshev: Non-finite value encountered at iteration " + 
                    std::to_string(current_iterations_));
            }
            
            // Check for oscillation or divergence
            if (current_iterations_ > 0) {
                double change = std::abs(x_new - x);
                double prev_change = std::abs(x - x_prev);
                
                if (change > 100.0 * prev_change && current_iterations_ > 3) {
                    if (verbose_) {
                        printVerbose("WARNING: Possible divergence detected");
                    }
                    
                    throw ConvergenceException(
                        "Chebyshev: Solution appears to be diverging");
                }
                
                // Check for stalling
                if (change < 1e-15 && std::abs(f) > tolerance_) {
                    if (verbose_) {
                        printVerbose("WARNING: Progress stalled");
                    }
                    
                    throw ConvergenceException(
                        "Chebyshev: Iteration stalled without convergence");
                }
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new) +
                           ", f(x) = " + std::to_string(f) +
                           ", |correction| = " + std::to_string(std::abs(correction)));
            }
            
            // Update for next iteration
            x_prev = x;
            x = x_new;
            current_x_ = x_new;
            
            recordError(std::abs(f));
        }
        
        // Maximum iterations reached
        last_root_ = x;
        last_error_ = std::abs(evaluateFunction(x));
        
        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: " + std::to_string(x));
            printVerbose("f(x) = " + std::to_string(last_error_));
        }
        
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        
        throw ConvergenceException("Chebyshev: Maximum iterations reached", info);
    }
    
    double iterate() override {
        double x = current_x_;
        
        // Evaluate function and derivatives
        double f = evaluateFunction(x);
        double df = evaluateDerivative(x);
        double ddf = evaluateSecondDerivative(x);
        
        if (isTooSmall(df)) {
            throw NumericalInstabilityException(
                "Chebyshev: Derivative too small");
        }
        
        // Compute L = f * f'' / (f')²
        double L = (f * ddf) / (df * df);
        
        // Chebyshev update
        double x_new = x - (1.0 + 0.5 * L) * (f / df);
        
        current_x_ = x_new;
        updateRoot(x_new, std::abs(f));
        
        return x_new;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = (current_iterations_ < max_iterations_) ?
                      SolverStatus::SUCCESS : SolverStatus::MAX_ITERATIONS;
        info.message = "Chebyshev's method (order 3)";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Chebyshev's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::CUBIC; // Order 3
    }
    
    bool requiresDerivative() const override {
        return true; // Requires both f' and f''
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }
    
    /**
     * @brief Compare with Newton-Raphson for same problem
     * @return String with comparison
     */
    std::string compareWithNewton() const {
        std::ostringstream oss;
        oss << "Chebyshev vs Newton-Raphson:\n";
        oss << "  Chebyshev: Order 3 (cubic), needs f' and f''\n";
        oss << "  Newton-Raphson: Order 2 (quadratic), needs only f'\n";
        oss << "  Chebyshev converges faster near root\n";
        oss << "  But Newton may be more efficient overall due to fewer derivative evaluations";
        return oss.str();
    }

private:
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

/**
 * @brief Modified Chebyshev method (Halley's method variant)
 * 
 * Alternative formulation that may have better convergence properties.
 */
class HalleyMethod : public SecondDerivativeMethod {
public:
    explicit HalleyMethod(Function f,
                         DerivativeFunction df,
                         SecondDerivativeFunction ddf,
                         double x0,
                         double tolerance = 1e-6,
                         int max_iter = 100)
        : SecondDerivativeMethod(std::move(f), std::move(df), std::move(ddf),
                                x0, tolerance, max_iter) {}
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        
        double x = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            double f = evaluateFunction(x);
            double df = evaluateDerivative(x);
            double ddf = evaluateSecondDerivative(x);
            
            if (std::abs(f) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(f);
                recordError(last_error_);
                return x;
            }
            
            if (isTooSmall(df)) {
                throw NumericalInstabilityException("Halley: Derivative too small");
            }
            
            // Halley's method: x_{n+1} = x_n - 2*f*f' / (2*(f')² - f*f'')
            double numerator = 2.0 * f * df;
            double denominator = 2.0 * df * df - f * ddf;
            
            if (isTooSmall(denominator)) {
                throw NumericalInstabilityException("Halley: Denominator too small");
            }
            
            double x_new = x - numerator / denominator;
            
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException("Halley: Non-finite value");
            }
            
            x = x_new;
            recordError(std::abs(f));
        }
        
        throw ConvergenceException("Halley: Maximum iterations reached");
    }
    
    double iterate() override {
        double x = current_x_;
        double f = evaluateFunction(x);
        double df = evaluateDerivative(x);
        double ddf = evaluateSecondDerivative(x);
        
        double numerator = 2.0 * f * df;
        double denominator = 2.0 * df * df - f * ddf;
        
        double x_new = x - numerator / denominator;
        current_x_ = x_new;
        
        return x_new;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Halley's method";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Halley's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::CUBIC;
    }
    
    bool requiresDerivative() const override {
        return true;
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }
};

} // namespace ode_solver