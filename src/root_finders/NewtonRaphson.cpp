/**
 * @file NewtonRaphson.cpp
 * @brief Implementation of Newton-Raphson method
 * @author Your Name
 * @version 1.0.0
 * 
 * Newton-Raphson is the most famous root-finding algorithm.
 * It uses the function and its derivative to achieve quadratic convergence.
 * 
 * Algorithm:
 *   x_{n+1} = x_n - f(x_n) / f'(x_n)
 * 
 * Geometric interpretation: Find where tangent line crosses x-axis
 * 
 * Convergence: Quadratic (order 2)
 * Advantages: Very fast convergence near root
 * Disadvantages: Requires derivative, may diverge if poor initial guess
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

namespace ode_solver {

/**
 * @brief Newton-Raphson method implementation
 * 
 * The classic root-finding algorithm. Uses tangent line approximation
 * to find where function crosses zero.
 */
class NewtonRaphsonMethod : public DerivativeMethod {
public:
    /**
     * @brief Constructor
     * @param f Function f(x)
     * @param df Derivative f'(x)
     * @param x0 Initial guess
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit NewtonRaphsonMethod(Function f,
                                DerivativeFunction df,
                                double x0,
                                double tolerance = 1e-6,
                                int max_iter = 100)
        : DerivativeMethod(std::move(f), std::move(df), x0, tolerance, max_iter)
        , min_derivative_(1e-12)
        , max_step_(1e10)
        , use_damping_(false)
        , damping_factor_(1.0) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Newton-Raphson Method");
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
            
            double f0 = evaluateFunction(x0_);
            double df0 = evaluateDerivative(x0_);
            printVerbose("f(x0) = " + std::to_string(f0));
            printVerbose("f'(x0) = " + std::to_string(df0));
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        resetDerivativeEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Newton-Raphson iteration");
            printVerbose("Tolerance: " + std::to_string(tolerance_));
        }
        
        double x = x0_;
        double x_prev = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Evaluate function and derivative
            double f = evaluateFunction(x);
            double df = evaluateDerivative(x);
            
            // Check convergence
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
            
            // Check for near-zero derivative (horizontal tangent)
            if (std::abs(df) < min_derivative_) {
                if (verbose_) {
                    printVerbose("WARNING: Derivative too small at x = " + 
                               std::to_string(x));
                }
                
                throw NumericalInstabilityException(
                    "Newton-Raphson: Derivative too small (|f'| = " + 
                    std::to_string(std::abs(df)) + "). " +
                    "Tangent is nearly horizontal.");
            }
            
            // Newton-Raphson step
            double step = f / df;
            double x_new = x - damping_factor_ * step;
            
            // Check for numerical issues
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Newton-Raphson: Non-finite value at iteration " + 
                    std::to_string(current_iterations_));
            }
            
            // Check for divergence (step too large)
            if (std::abs(step) > max_step_) {
                if (verbose_) {
                    printVerbose("WARNING: Large step detected, |step| = " + 
                               std::to_string(std::abs(step)));
                }
                
                // Enable damping if not already on
                if (!use_damping_) {
                    use_damping_ = true;
                    damping_factor_ = 0.5;
                    if (verbose_) {
                        printVerbose("Enabling damping with factor 0.5");
                    }
                    x_new = x - damping_factor_ * step;
                }
            }
            
            // Check for oscillation
            if (current_iterations_ > 1) {
                double change = std::abs(x_new - x);
                double prev_change = std::abs(x - x_prev);
                
                if (change > 2.0 * prev_change && change > tolerance_ * 100) {
                    if (verbose_) {
                        printVerbose("WARNING: Possible oscillation or divergence");
                    }
                    
                    // Try damping
                    if (!use_damping_) {
                        use_damping_ = true;
                        damping_factor_ = 0.5;
                        if (verbose_) {
                            printVerbose("Applying damping to stabilize");
                        }
                    }
                }
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new) +
                           ", f(x) = " + std::to_string(f) +
                           ", f'(x) = " + std::to_string(df) +
                           ", |step| = " + std::to_string(std::abs(step)));
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
        
        throw ConvergenceException("Newton-Raphson: Maximum iterations reached", info);
    }
    
    double iterate() override {
        double x = current_x_;
        
        double f = evaluateFunction(x);
        double df = evaluateDerivative(x);
        
        if (isTooSmall(df, min_derivative_)) {
            throw NumericalInstabilityException(
                "Newton-Raphson: Derivative too small");
        }
        
        double x_new = x - f / df;
        
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
        info.message = "Newton-Raphson method";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Newton-Raphson Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::QUADRATIC;
    }
    
    bool requiresDerivative() const override {
        return true;
    }
    
    bool supportsComplexRoots() const override {
        return false; // Real version only
    }
    
    /**
     * @brief Set minimum acceptable derivative magnitude
     * @param min_deriv Minimum |f'(x)| allowed
     */
    void setMinimumDerivative(double min_deriv) {
        if (min_deriv <= 0) {
            throw InvalidParameterException("Minimum derivative must be positive");
        }
        min_derivative_ = min_deriv;
    }
    
    /**
     * @brief Enable damped Newton method
     * @param factor Damping factor λ ∈ (0,1]: x_{n+1} = x_n - λ·f/f'
     * 
     * Damping can help convergence from poor initial guesses
     * by taking smaller steps.
     */
    void setDampingFactor(double factor) {
        if (factor <= 0.0 || factor > 1.0) {
            throw InvalidParameterException("Damping factor must be in (0,1]");
        }
        damping_factor_ = factor;
        use_damping_ = (factor < 1.0);
        
        if (verbose_) {
            printVerbose("Damping factor set to " + std::to_string(factor));
        }
    }
    
    /**
     * @brief Set maximum allowed step size
     * @param max_step Maximum |x_{n+1} - x_n|
     */
    void setMaximumStep(double max_step) {
        if (max_step <= 0) {
            throw InvalidParameterException("Maximum step must be positive");
        }
        max_step_ = max_step;
    }
    
    /**
     * @brief Estimate basin of attraction size
     * @return Rough estimate of how close x0 needs to be
     * 
     * For Newton to converge, generally need |x0 - x*| < 1/|f''(x*)|
     */
    double estimateBasinSize() const {
        // This is a very rough heuristic
        double x = last_root_;
        if (std::isnan(x)) x = x0_;
        
        double df = evaluateDerivative(x);
        return 1.0 / std::max(std::abs(df), 1.0);
    }

private:
    double min_derivative_;   ///< Minimum acceptable |f'(x)|
    double max_step_;         ///< Maximum step size
    bool use_damping_;        ///< Whether damping is enabled
    double damping_factor_;   ///< Damping factor λ
};

/**
 * @brief Modified Newton method with multiplicity
 * 
 * For roots with multiplicity m > 1, standard Newton converges linearly.
 * Modified Newton restores quadratic convergence:
 *   x_{n+1} = x_n - m·f(x_n)/f'(x_n)
 */
class ModifiedNewtonMethod : public DerivativeMethod {
public:
    explicit ModifiedNewtonMethod(Function f,
                                 DerivativeFunction df,
                                 double x0,
                                 int multiplicity,
                                 double tolerance = 1e-6,
                                 int max_iter = 100)
        : DerivativeMethod(std::move(f), std::move(df), x0, tolerance, max_iter)
        , multiplicity_(multiplicity) {
        
        if (multiplicity < 1) {
            throw InvalidParameterException("Multiplicity must be >= 1");
        }
        
        if (verbose_) {
            printVerbose("Modified Newton for multiplicity " + 
                        std::to_string(multiplicity));
        }
    }
    
    double solve() override {
        double x = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            double f = evaluateFunction(x);
            double df = evaluateDerivative(x);
            
            if (std::abs(f) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(f);
                return x;
            }
            
            if (isTooSmall(df)) {
                throw NumericalInstabilityException("Modified Newton: Derivative too small");
            }
            
            // Modified Newton with multiplicity
            double x_new = x - multiplicity_ * f / df;
            
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException("Modified Newton: Non-finite value");
            }
            
            x = x_new;
        }
        
        throw ConvergenceException("Modified Newton: Maximum iterations reached");
    }
    
    double iterate() override {
        double x = current_x_;
        double f = evaluateFunction(x);
        double df = evaluateDerivative(x);
        
        current_x_ = x - multiplicity_ * f / df;
        return current_x_;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        return info;
    }
    
    std::string getMethodName() const override {
        return "Modified Newton (m=" + std::to_string(multiplicity_) + ")";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::QUADRATIC;
    }
    
    bool requiresDerivative() const override { return true; }
    bool supportsComplexRoots() const override { return false; }

private:
    int multiplicity_;
};

} // namespace ode_solver