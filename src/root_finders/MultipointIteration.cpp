/**
 * @file MultipointIteration.cpp
 * @brief Implementation of multipoint iteration methods
 * @author Your Name
 * @version 1.0.0
 * 
 * Multipoint iteration methods improve upon standard fixed-point iteration
 * by using multiple function evaluations per iteration to achieve higher
 * order convergence without derivatives.
 * 
 * Variants:
 * - Two-point method (order ~1.4)
 * - Three-point method (order ~1.8)
 * - Traub's method (order 2)
 * 
 * Convergence: Superlinear to quadratic
 * Advantages: Higher order without derivatives
 * Disadvantages: Multiple evaluations per iteration
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ode_solver {

/**
 * @brief Multipoint iteration method
 * 
 * Uses multiple function evaluations per iteration to achieve
 * superlinear convergence without computing derivatives.
 */
class MultipointIterationMethod : public OpenMethod {
public:
    enum class Variant {
        TWO_POINT,      ///< Two evaluations per iteration, order ~1.4
        THREE_POINT,    ///< Three evaluations per iteration, order ~1.8
        TRAUB           ///< Traub's method, order 2
    };
    
    /**
     * @brief Constructor
     * @param f Function to find root of
     * @param x0 Initial guess
     * @param variant Which multipoint variant to use
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit MultipointIterationMethod(Function f,
                                      double x0,
                                      Variant variant = Variant::TRAUB,
                                      double tolerance = 1e-6,
                                      int max_iter = 100)
        : OpenMethod(std::move(f), x0, tolerance, max_iter)
        , variant_(variant)
        , h_(0.01) {  // Step size for finite differences
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Multipoint Iteration Method");
            printVerbose("Variant: " + getVariantName());
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        if (verbose_) {
            printVerbose("Starting multipoint iteration");
        }
        
        double x = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            double f_x = evaluateFunction(x);
            
            // Check convergence
            if (std::abs(f_x) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(f_x);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(x));
                    printVerbose("f(root) = " + std::to_string(f_x));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                    printVerbose("Function evaluations: " + 
                               std::to_string(function_evaluations_));
                }
                
                recordError(last_error_);
                return x;
            }
            
            double x_new;
            
            switch (variant_) {
                case Variant::TWO_POINT:
                    x_new = twoPointIteration(x, f_x);
                    break;
                    
                case Variant::THREE_POINT:
                    x_new = threePointIteration(x, f_x);
                    break;
                    
                case Variant::TRAUB:
                    x_new = traubIteration(x, f_x);
                    break;
                    
                default:
                    throw std::logic_error("Invalid variant");
            }
            
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Multipoint: Non-finite value");
            }
            
            // Check for divergence
            if (std::abs(x_new) > 1e10) {
                throw ConvergenceException("Multipoint: Diverging");
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new) +
                           ", f(x) = " + std::to_string(f_x) +
                           ", |change| = " + std::to_string(std::abs(x_new - x)));
            }
            
            x = x_new;
            current_x_ = x_new;
            recordError(std::abs(f_x));
        }
        
        // Maximum iterations reached
        last_root_ = x;
        last_error_ = std::abs(evaluateFunction(x));
        
        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
        }
        
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        
        throw ConvergenceException("Multipoint: Maximum iterations reached", info);
    }
    
    double iterate() override {
        double x = current_x_;
        double f_x = evaluateFunction(x);
        
        double x_new;
        switch (variant_) {
            case Variant::TWO_POINT:
                x_new = twoPointIteration(x, f_x);
                break;
            case Variant::THREE_POINT:
                x_new = threePointIteration(x, f_x);
                break;
            case Variant::TRAUB:
                x_new = traubIteration(x, f_x);
                break;
            default:
                throw std::logic_error("Invalid variant");
        }
        
        current_x_ = x_new;
        return x_new;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Multipoint iteration (" + getVariantName() + ")";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Multipoint Iteration (" + getVariantName() + ")";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        switch (variant_) {
            case Variant::TWO_POINT:
                return ConvergenceRate::SUPERLINEAR;
            case Variant::THREE_POINT:
                return ConvergenceRate::SUPERLINEAR;
            case Variant::TRAUB:
                return ConvergenceRate::QUADRATIC;
            default:
                return ConvergenceRate::LINEAR;
        }
    }
    
    bool requiresDerivative() const override {
        return false;
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }
    
    /**
     * @brief Set step size for finite difference approximations
     * @param h Step size
     */
    void setStepSize(double h) {
        if (h <= 0) {
            throw InvalidParameterException("Step size must be positive");
        }
        h_ = h;
    }

private:
    Variant variant_;
    double h_;  // Step size for finite differences
    
    /**
     * @brief Two-point iteration
     * 
     * Uses two function evaluations:
     * y = x - f(x)/f'(x) where f' is approximated
     * x_new = y - f(y)/f'(x)
     */
    double twoPointIteration(double x, double f_x) {
        // Approximate derivative using forward difference
        double f_x_plus_h = evaluateFunction(x + h_);
        double df_approx = (f_x_plus_h - f_x) / h_;
        
        if (std::abs(df_approx) < 1e-15) {
            throw NumericalInstabilityException(
                "Two-point: Derivative approximation too small");
        }
        
        // Newton-like step
        double y = x - f_x / df_approx;
        double f_y = evaluateFunction(y);
        
        // Second step using same derivative approximation
        double x_new = y - f_y / df_approx;
        
        return x_new;
    }
    
    /**
     * @brief Three-point iteration
     * 
     * Uses three function evaluations for higher order.
     */
    double threePointIteration(double x, double f_x) {
        // First approximation
        double f_x_plus_h = evaluateFunction(x + h_);
        double df_approx = (f_x_plus_h - f_x) / h_;
        
        if (std::abs(df_approx) < 1e-15) {
            throw NumericalInstabilityException("Three-point: Derivative too small");
        }
        
        double y = x - f_x / df_approx;
        double f_y = evaluateFunction(y);
        
        // Improved derivative approximation using both points
        double df_improved = (f_y - f_x) / (y - x);
        
        if (std::abs(df_improved) < 1e-15) {
            df_improved = df_approx;  // Fall back
        }
        
        // Final step
        double z = y - f_y / df_improved;
        double f_z = evaluateFunction(z);
        
        // One more correction
        double df_final = (f_z - f_y) / (z - y);
        if (std::abs(df_final) < 1e-15) {
            df_final = df_improved;
        }
        
        double x_new = z - f_z / df_final;
        
        return x_new;
    }
    
    /**
     * @brief Traub's iteration (order 2 without derivatives)
     * 
     * Traub's method achieves quadratic convergence without derivatives:
     * y = x - f(x)/f'_approx(x)
     * x_new = y - f(y) * (x - y) / (f(x) - f(y))
     */
    double traubIteration(double x, double f_x) {
        // Approximate derivative
        double f_x_plus_h = evaluateFunction(x + h_);
        double df_approx = (f_x_plus_h - f_x) / h_;
        
        if (std::abs(df_approx) < 1e-15) {
            throw NumericalInstabilityException("Traub: Derivative too small");
        }
        
        // First Newton-like step
        double y = x - f_x / df_approx;
        double f_y = evaluateFunction(y);
        
        // Traub's correction using secant-like formula
        double denominator = f_x - f_y;
        if (std::abs(denominator) < 1e-15) {
            // Fall back to simple iteration
            return y;
        }
        
        double correction = f_y * (x - y) / denominator;
        double x_new = y - correction;
        
        return x_new;
    }
    
    std::string getVariantName() const {
        switch (variant_) {
            case Variant::TWO_POINT:   return "Two-Point";
            case Variant::THREE_POINT: return "Three-Point";
            case Variant::TRAUB:       return "Traub";
            default:                   return "Unknown";
        }
    }
    
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

/**
 * @brief Wegstein's method - a variant of multipoint iteration
 * 
 * Popular in chemical engineering for solving coupled nonlinear equations.
 */
class WegsteinMethod : public OpenMethod {
public:
    explicit WegsteinMethod(Function g,
                           double x0,
                           double tolerance = 1e-6,
                           int max_iter = 100)
        : OpenMethod([g](double x) { return x - g(x); }, x0, tolerance, max_iter)
        , g_(std::move(g))
        , q_(-1.0) {  // Acceleration parameter
        
        if (verbose_) {
            printVerbose("Initializing Wegstein's Method");
        }
    }
    
    double solve() override {
        double x = x0_;
        double x_prev = x0_;
        double g_x_prev = g_(x0_);
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            double g_x = g_(x);
            
            if (std::abs(x - g_x) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(x - g_x);
                return x;
            }
            
            // Wegstein acceleration
            if (current_iterations_ > 0) {
                double numerator = g_x - g_x_prev;
                double denominator = (g_x - x) - (g_x_prev - x_prev);
                
                if (std::abs(denominator) > 1e-15) {
                    q_ = numerator / denominator;
                    
                    // Limit q for stability
                    q_ = std::max(-5.0, std::min(5.0, q_));
                }
            }
            
            // Wegstein formula: x_new = (1-q)*g(x) + q*x
            double x_new = (1.0 - q_) * g_x + q_ * x;
            
            x_prev = x;
            g_x_prev = g_x;
            x = x_new;
        }
        
        throw ConvergenceException("Wegstein: Maximum iterations reached");
    }
    
    double iterate() override {
        double x = current_x_;
        double g_x = g_(x);
        current_x_ = (1.0 - q_) * g_x + q_ * x;
        return current_x_;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        return info;
    }
    
    std::string getMethodName() const override {
        return "Wegstein's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::SUPERLINEAR;
    }
    
    bool requiresDerivative() const override {
        return false;
    }
    
    bool supportsComplexRoots() const override {
        return false;
    }

private:
    Function g_;
    double q_;  // Acceleration parameter
};

} // namespace ode_solver