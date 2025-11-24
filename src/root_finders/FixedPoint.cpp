/**
 * @file FixedPoint.cpp
 * @brief Implementation of Fixed-Point Iteration method
 * @author Your Name
 * @version 1.0.0
 * 
 * Fixed-point iteration solves equations of the form x = g(x) by
 * repeatedly applying the function: x_{n+1} = g(x_n)
 * 
 * To solve f(x) = 0, rearrange to x = g(x). For example:
 *   x² - 2 = 0  →  x = 2/x  →  g(x) = 2/x
 * 
 * Convergence condition: |g'(x*)| < 1 at fixed point x*
 * 
 * Algorithm:
 *   x_{n+1} = g(x_n)
 *   Repeat until |x_{n+1} - x_n| < tolerance
 * 
 * Convergence: Linear (if |g'(x*)| < 1)
 * Advantages: Simple, no derivatives of original function needed
 * Disadvantages: May diverge, sensitive to g(x) formulation
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

namespace ode_solver {

/**
 * @brief Fixed-Point Iteration method
 * 
 * Solves x = g(x) by iteration: x_{n+1} = g(x_n)
 * Convergence requires |g'(x*)| < 1 at the fixed point.
 */
class FixedPointIteration : public OpenMethod {
public:
    /**
     * @brief Constructor for fixed-point iteration
     * @param g Iteration function g(x) where solution satisfies x = g(x)
     * @param x0 Initial guess
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit FixedPointIteration(Function g,
                                 double x0,
                                 double tolerance = 1e-6,
                                 int max_iter = 1000)
        : OpenMethod([g](double x) { return x - g(x); }, x0, tolerance, max_iter)
        , g_(std::move(g))
        , relaxation_factor_(1.0)
        , use_aitken_(false)
        , stagnation_count_(0) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Fixed-Point Iteration");
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
            printVerbose("g(x0) = " + std::to_string(g_(x0_)));
        }
    }
    
    /**
     * @brief Constructor from original function f(x) = 0
     * @param f Original function f(x) = 0
     * @param g Iteration function g(x)
     * @param x0 Initial guess
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit FixedPointIteration(Function f,
                                 Function g,
                                 double x0,
                                 double tolerance = 1e-6,
                                 int max_iter = 1000)
        : OpenMethod(std::move(f), x0, tolerance, max_iter)
        , g_(std::move(g))
        , relaxation_factor_(1.0)
        , use_aitken_(false)
        , stagnation_count_(0) {
        
        if (verbose_) {
            printVerbose("Initializing Fixed-Point Iteration");
            printVerbose("Solving f(x) = 0 using x = g(x)");
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Fixed-Point iteration");
            printVerbose("Tolerance: " + std::to_string(tolerance_));
            if (relaxation_factor_ != 1.0) {
                printVerbose("Using relaxation factor: " + 
                           std::to_string(relaxation_factor_));
            }
            if (use_aitken_) {
                printVerbose("Aitken acceleration enabled");
            }
        }
        
        double x = x0_;
        double x_prev = x0_;
        double x_prev2 = x0_;
        stagnation_count_ = 0;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Compute next iteration
            double g_x = g_(x);
            ++function_evaluations_;
            
            // Check for numerical issues
            if (!isFinite(g_x)) {
                throw NumericalInstabilityException(
                    "Fixed-Point: Non-finite value from g(x) at x = " + 
                    std::to_string(x));
            }
            
            double x_new;
            
            if (use_aitken_ && current_iterations_ >= 2) {
                // Aitken's Δ² acceleration
                x_new = applyAitkenAcceleration(x_prev2, x_prev, x);
                
                if (verbose_ && current_iterations_ % 10 == 0) {
                    printVerbose("  Using Aitken acceleration");
                }
            } else {
                // Standard fixed-point iteration with optional relaxation
                // x_{n+1} = x_n + ω(g(x_n) - x_n)
                x_new = x + relaxation_factor_ * (g_x - x);
            }
            
            // Check for convergence
            double change = std::abs(x_new - x);
            double relative_change = (std::abs(x) > 1e-10) ? 
                                    change / std::abs(x) : change;
            
            if (verbose_ && current_iterations_ % 10 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new) +
                           ", |change| = " + std::to_string(change));
            }
            
            // Check for convergence
            if (change < tolerance_ || relative_change < tolerance_) {
                last_root_ = x_new;
                last_error_ = change;
                
                // Verify it's actually a fixed point
                double residual = std::abs(x_new - g_(x_new));
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Fixed point: x = " + std::to_string(x_new));
                    printVerbose("g(x) = " + std::to_string(g_(x_new)));
                    printVerbose("Residual |x - g(x)| = " + std::to_string(residual));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                    printVerbose("Function evaluations: " + 
                               std::to_string(function_evaluations_));
                }
                
                recordError(last_error_);
                return x_new;
            }
            
            // Check for stagnation (very slow progress)
            if (change < 1e-12 && change > tolerance_) {
                ++stagnation_count_;
                if (stagnation_count_ > 10) {
                    if (verbose_) {
                        printVerbose("WARNING: Iteration appears stagnant");
                    }
                    
                    // Try enabling Aitken acceleration if not already on
                    if (!use_aitken_ && current_iterations_ >= 2) {
                        use_aitken_ = true;
                        stagnation_count_ = 0;
                        if (verbose_) {
                            printVerbose("Enabling Aitken acceleration");
                        }
                    }
                }
            } else {
                stagnation_count_ = 0;
            }
            
            // Check for divergence
            if (std::abs(x_new) > 1e10) {
                throw ConvergenceException(
                    "Fixed-Point: Iteration appears to be diverging. "
                    "Check that |g'(x*)| < 1 at the fixed point.");
            }
            
            // Detect oscillation
            if (current_iterations_ > 5) {
                if (std::abs(x_new - x_prev) < 1e-10 && 
                    std::abs(x_new - x) > tolerance_ * 10) {
                    if (verbose_) {
                        printVerbose("WARNING: Oscillation detected");
                    }
                    
                    // Try relaxation
                    if (relaxation_factor_ == 1.0) {
                        relaxation_factor_ = 0.5;
                        if (verbose_) {
                            printVerbose("Applying relaxation factor 0.5");
                        }
                    }
                }
            }
            
            // Update for next iteration
            x_prev2 = x_prev;
            x_prev = x;
            x = x_new;
            current_x_ = x_new;
            
            recordError(change);
        }
        
        // Maximum iterations reached
        last_root_ = x;
        last_error_ = std::abs(x - g_(x));
        
        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: x = " + std::to_string(x));
            printVerbose("g(x) = " + std::to_string(g_(x)));
            printVerbose("Residual: " + std::to_string(last_error_));
        }
        
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached. May not have converged.";
        info.execution_time_ms = getElapsedTime(timer_start);
        
        throw ConvergenceException(
            "Fixed-Point: Maximum iterations reached. "
            "Verify that |g'(x*)| < 1 for convergence.", info);
    }
    
    double iterate() override {
        double x = current_x_;
        double g_x = g_(x);
        
        double x_new = x + relaxation_factor_ * (g_x - x);
        
        current_x_ = x_new;
        updateRoot(x_new, std::abs(x_new - x));
        
        return x_new;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = (current_iterations_ < max_iterations_) ?
                      SolverStatus::SUCCESS : SolverStatus::MAX_ITERATIONS;
        info.message = "Fixed-Point Iteration";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Fixed-Point Iteration";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::LINEAR;
    }
    
    bool requiresDerivative() const override {
        return false;
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }
    
    /**
     * @brief Set relaxation factor for damped iteration
     * @param omega Relaxation factor (0 < ω ≤ 1)
     * 
     * Damped iteration: x_{n+1} = x_n + ω(g(x_n) - x_n)
     * Use ω < 1 to slow down convergence and improve stability
     */
    void setRelaxationFactor(double omega) {
        if (omega <= 0.0 || omega > 1.0) {
            throw InvalidParameterException(
                "Relaxation factor must be in (0, 1]");
        }
        relaxation_factor_ = omega;
        
        if (verbose_) {
            printVerbose("Relaxation factor set to " + std::to_string(omega));
        }
    }
    
    /**
     * @brief Enable Aitken's Δ² acceleration
     * @param enable True to enable acceleration
     * 
     * Aitken acceleration can significantly speed up convergence
     * for slowly converging sequences.
     */
    void enableAitkenAcceleration(bool enable = true) {
        use_aitken_ = enable;
        
        if (verbose_) {
            printVerbose(enable ? "Aitken acceleration enabled" : 
                                 "Aitken acceleration disabled");
        }
    }
    
    /**
     * @brief Get the iteration function
     * @return Reference to g(x)
     */
    const Function& getIterationFunction() const {
        return g_;
    }
    
    /**
     * @brief Estimate convergence rate from last iterations
     * @return Estimated |g'(x*)|
     */
    double estimateConvergenceRate() const {
        if (error_history_.size() < 3) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        
        size_t n = error_history_.size();
        // Estimate rate: e_{n+1} / e_n ≈ |g'(x*)|
        return error_history_[n-1] / error_history_[n-2];
    }

private:
    Function g_;                  ///< Iteration function g(x)
    double relaxation_factor_;    ///< Damping factor ω
    bool use_aitken_;            ///< Use Aitken acceleration
    int stagnation_count_;       ///< Count of stagnant iterations
    
    /**
     * @brief Apply Aitken's Δ² acceleration
     * @param x0 Value at iteration n-2
     * @param x1 Value at iteration n-1
     * @param x2 Value at iteration n
     * @return Accelerated value
     */
    double applyAitkenAcceleration(double x0, double x1, double x2) const {
        // Aitken's Δ² formula:
        // x̂ = x2 - (x2 - x1)² / (x2 - 2*x1 + x0)
        
        double delta1 = x1 - x0;
        double delta2 = x2 - x1;
        double denominator = delta2 - delta1;
        
        if (std::abs(denominator) < 1e-15) {
            // Denomininator too small, return standard iteration
            return x2;
        }
        
        double x_accelerated = x0 - (delta1 * delta1) / denominator;
        
        // Check if acceleration is reasonable
        if (!isFinite(x_accelerated)) {
            return x2;
        }
        
        return x_accelerated;
    }
    
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

/**
 * @brief Steffensen's method - acceleration of fixed-point iteration
 * 
 * Achieves quadratic convergence without derivatives by using
 * Aitken acceleration at every step.
 */
class SteffensenMethod : public OpenMethod {
public:
    explicit SteffensenMethod(Function g,
                             double x0,
                             double tolerance = 1e-6,
                             int max_iter = 100)
        : OpenMethod([g](double x) { return x - g(x); }, x0, tolerance, max_iter)
        , g_(std::move(g)) {
        
        if (verbose_) {
            printVerbose("Initializing Steffensen's Method");
            printVerbose("Quadratic convergence without derivatives");
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        
        if (verbose_) {
            printVerbose("Starting Steffensen iteration");
        }
        
        double x = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Compute two fixed-point iterations
            double g_x = g_(x);
            double g_g_x = g_(g_x);
            
            // Check for convergence before acceleration
            if (std::abs(x - g_x) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(x - g_x);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(x));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                }
                
                return x;
            }
            
            // Aitken acceleration
            double denominator = g_g_x - 2.0 * g_x + x;
            
            if (std::abs(denominator) < 1e-15) {
                // Denominator too small, use standard iteration
                x = g_x;
                continue;
            }
            
            double x_new = x - (g_x - x) * (g_x - x) / denominator;
            
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Steffensen: Non-finite value");
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new));
            }
            
            // Check for divergence
            if (std::abs(x_new) > 1e10) {
                throw ConvergenceException("Steffensen: Diverging");
            }
            
            x = x_new;
            recordError(std::abs(denominator));
        }
        
        throw ConvergenceException("Steffensen: Maximum iterations reached");
    }
    
    double iterate() override {
        double x = current_x_;
        double g_x = g_(x);
        double g_g_x = g_(g_x);
        
        double denominator = g_g_x - 2.0 * g_x + x;
        double x_new = x - (g_x - x) * (g_x - x) / denominator;
        
        current_x_ = x_new;
        return x_new;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        return info;
    }
    
    std::string getMethodName() const override {
        return "Steffensen's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::QUADRATIC; // Order 2 without derivatives!
    }
    
    bool requiresDerivative() const override {
        return false; // Achieves quadratic without derivatives
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }

private:
    Function g_;
};

} // namespace ode_solver