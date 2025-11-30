/**
 * @file Muller.cpp
 * @brief Implementation of Muller's method for root finding
 * @author Your Name
 * @version 1.0.0
 * 
 * Muller's method uses parabolic interpolation through three points
 * to approximate the function and find roots. It can find complex roots
 * even when starting from real initial guesses.
 * 
 * Algorithm:
 *   Given three points (x₀, f₀), (x₁, f₁), (x₂, f₂)
 *   Fit parabola through them: P(x) = a(x-x₂)² + b(x-x₂) + c
 *   Find root of parabola as next approximation
 *   Replace oldest point with new point
 * 
 * Convergence: Superlinear (~1.84 order)
 * Advantages: Can find complex roots, doesn't need derivatives
 * Disadvantages: More complex than secant, needs three initial points
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <complex>

namespace ode_solver {

/**
 * @brief Muller's method for root finding
 * 
 * Uses parabolic interpolation through three points.
 * Can find complex roots even from real starting points.
 */
class MullerMethod : public OpenMethod {
public:
    /**
     * @brief Constructor with three initial points
     * @param f Function to find root of
     * @param x0 First initial point
     * @param x1 Second initial point (default: x0 + 0.5)
     * @param x2 Third initial point (default: x0 - 0.5)
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit MullerMethod(Function f,
                         double x0,
                         double x1 = std::numeric_limits<double>::quiet_NaN(),
                         double x2 = std::numeric_limits<double>::quiet_NaN(),
                         double tolerance = 1e-6,
                         int max_iter = 100)
        : OpenMethod(std::move(f), x0, tolerance, max_iter)
        , x1_(std::isnan(x1) ? x0 + 0.5 : x1)
        , x2_(std::isnan(x2) ? x0 - 0.5 : x2) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Muller's Method");
            printVerbose("Initial points: x0=" + std::to_string(x0_) +
                        ", x1=" + std::to_string(x1_) +
                        ", x2=" + std::to_string(x2_));
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Muller iteration");
        }
        
        // Initialize three points
        double x0 = x0_;
        double x1 = x1_;
        double x2 = x2_;
        
        double f0 = evaluateFunction(x0);
        double f1 = evaluateFunction(x1);
        double f2 = evaluateFunction(x2);
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Check convergence at most recent point
            if (std::abs(f2) < tolerance_) {
                last_root_ = x2;
                last_error_ = std::abs(f2);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(x2));
                    printVerbose("f(root) = " + std::to_string(f2));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                }
                
                recordError(last_error_);
                return x2;
            }
            
            // Compute divided differences
            double h0 = x1 - x0;
            double h1 = x2 - x1;
            double delta0 = (f1 - f0) / h0;
            double delta1 = (f2 - f1) / h1;
            
            // Coefficients of parabola P(x) = a(x-x2)² + b(x-x2) + c
            double a = (delta1 - delta0) / (h1 + h0);
            double b = a * h1 + delta1;
            double c = f2;
            
            // Discriminant for quadratic formula
            double discriminant = b * b - 4.0 * a * c;
            
            // Handle complex roots (discriminant < 0)
            if (discriminant < 0) {
                if (verbose_) {
                    printVerbose("Complex root detected at iteration " + 
                               std::to_string(current_iterations_));
                    printVerbose("Discriminant: " + std::to_string(discriminant));
                }
                
                // For real root finding, we can't continue
                // In practice, might want to switch to complex arithmetic
                throw NumericalInstabilityException(
                    "Muller: Encountered complex root. " 
                    "Use complex version for complex roots.");
            }
            
            double sqrt_disc = std::sqrt(discriminant);
            
            // Choose sign to maximize denominator (avoid cancellation)
            double denom1 = b + sqrt_disc;
            double denom2 = b - sqrt_disc;
            double denominator = (std::abs(denom1) > std::abs(denom2)) ?
                                denom1 : denom2;
            
            if (std::abs(denominator) < 1e-15) {
                throw NumericalInstabilityException(
                    "Muller: Denominator too small");
            }
            
            // Next approximation using quadratic formula
            // x_new = x2 - 2c / (b ± √(b² - 4ac))
            double dx = -2.0 * c / denominator;
            double x_new = x2 + dx;
            
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Muller: Non-finite value");
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new) +
                           ", f(x) = " + std::to_string(f2) +
                           ", |step| = " + std::to_string(std::abs(dx)));
            }
            
            // Update points: drop oldest, add new
            double f_new = evaluateFunction(x_new);
            
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f2;
            x2 = x_new;
            f2 = f_new;
            
            recordError(std::abs(f2));
        }
        
        // Maximum iterations reached
        last_root_ = x2;
        last_error_ = std::abs(f2);
        
        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: " + std::to_string(x2));
        }
        
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        
        throw ConvergenceException("Muller: Maximum iterations reached", info);
    }
    
    double iterate() override {
        // Muller needs three points, so single iteration not well-defined
        throw std::logic_error("Muller: Use solve() instead of iterate()");
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Muller's method";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Muller's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::SUPERLINEAR; // ~1.84 order
    }
    
    bool requiresDerivative() const override {
        return false;
    }
    
    bool supportsComplexRoots() const override {
        return true; // Can find complex roots
    }

private:
    double x1_;
    double x2_;
    
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

/**
 * @brief Complex version of Muller's method
 * 
 * Explicitly handles complex arithmetic for finding complex roots.
 */
class ComplexMullerMethod {
public:
    using Complex = std::complex<double>;
    
    explicit ComplexMullerMethod(std::function<Complex(Complex)> f,
                                Complex x0,
                                Complex x1,
                                Complex x2,
                                double tolerance = 1e-6,
                                int max_iter = 100)
        : f_(std::move(f))
        , x0_(x0)
        , x1_(x1)
        , x2_(x2)
        , tolerance_(tolerance)
        , max_iterations_(max_iter) {}
    
    Complex solve() {
        Complex x0 = x0_;
        Complex x1 = x1_;
        Complex x2 = x2_;
        
        Complex f0 = f_(x0);
        Complex f1 = f_(x1);
        Complex f2 = f_(x2);
        
        for (int iter = 0; iter < max_iterations_; ++iter) {
            if (std::abs(f2) < tolerance_) {
                return x2;
            }
            
            Complex h0 = x1 - x0;
            Complex h1 = x2 - x1;
            Complex delta0 = (f1 - f0) / h0;
            Complex delta1 = (f2 - f1) / h1;
            
            Complex a = (delta1 - delta0) / (h1 + h0);
            Complex b = a * h1 + delta1;
            Complex c = f2;
            
            Complex discriminant = b * b - 4.0 * a * c;
            Complex sqrt_disc = std::sqrt(discriminant);
            
            Complex denom1 = b + sqrt_disc;
            Complex denom2 = b - sqrt_disc;
            Complex denominator = (std::abs(denom1) > std::abs(denom2)) ?
                                 denom1 : denom2;
            
            Complex dx = -2.0 * c / denominator;
            Complex x_new = x2 + dx;
            Complex f_new = f_(x_new);
            
            // Update
            x0 = x1; f0 = f1;
            x1 = x2; f1 = f2;
            x2 = x_new; f2 = f_new;
        }
        
        throw std::runtime_error("Complex Muller: Maximum iterations reached");
    }

private:
    std::function<Complex(Complex)> f_;
    Complex x0_, x1_, x2_;
    double tolerance_;
    int max_iterations_;
};

} // namespace ode_solver