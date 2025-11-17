/**
 * @file Bisection.cpp
 * @brief Implementation of the Bisection method for root finding
 * @author Your Name
 * @version 1.0.0
 * 
 * The Bisection method is the most reliable root-finding method.
 * It is guaranteed to converge if the function is continuous and
 * changes sign over the interval [a, b].
 * 
 * Algorithm:
 * - Start with interval [a, b] where f(a)·f(b) < 0
 * - Compute midpoint c = (a + b)/2
 * - Replace either a or b with c based on sign of f(c)
 * - Repeat until |b - a| < tolerance
 * 
 * Convergence: Linear (halves interval each iteration)
 * Advantages: Always converges, very robust
 * Disadvantages: Slow convergence, requires bracket
 */

#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ode_solver {

/**
 * @brief Bisection method implementation
 * 
 * Most reliable bracketing method. Guaranteed to converge for continuous
 * functions that change sign over [a, b].
 */
class BisectionMethod : public BracketingMethod {
public:
    /**
     * @brief Constructor
     * @param f Function to find root of
     * @param a Left bracket endpoint
     * @param b Right bracket endpoint
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit BisectionMethod(Function f,
                            double a,
                            double b,
                            double tolerance = 1e-6,
                            int max_iter = 1000)
        : BracketingMethod(std::move(f), 
                          BracketConfig{a, b, true}, 
                          tolerance, 
                          max_iter) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Bisection Method");
            printVerbose("Bracket: [" + std::to_string(a_) + ", " + 
                        std::to_string(b_) + "]");
            printVerbose("f(a) = " + std::to_string(f_a_) + 
                        ", f(b) = " + std::to_string(f_b_));
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Bisection iteration");
            printVerbose("Tolerance: " + std::to_string(tolerance_));
        }
        
        // Check if endpoints are roots
        if (std::abs(f_a_) < tolerance_) {
            last_root_ = a_;
            last_error_ = std::abs(f_a_);
            return a_;
        }
        if (std::abs(f_b_) < tolerance_) {
            last_root_ = b_;
            last_error_ = std::abs(f_b_);
            return b_;
        }
        
        double a = a_;
        double b = b_;
        double f_a = f_a_;
        double f_b = f_b_;
        double c = 0.0;
        double f_c = 0.0;
        
        // Main bisection loop
        for (current_iterations_ = 0; 
             current_iterations_ < max_iterations_; 
             ++current_iterations_) {
            
            // Compute midpoint
            c = 0.5 * (a + b);
            f_c = evaluateFunction(c);
            
            // Current error estimate (bracket width)
            double bracket_width = std::abs(b - a);
            double function_value = std::abs(f_c);
            
            if (verbose_ && current_iterations_ % 10 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) + 
                           ": c = " + std::to_string(c) +
                           ", f(c) = " + std::to_string(f_c) +
                           ", bracket width = " + std::to_string(bracket_width));
            }
            
            // Check convergence
            if (function_value < tolerance_ || bracket_width < tolerance_) {
                last_root_ = c;
                last_error_ = std::min(function_value, bracket_width);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(c));
                    printVerbose("f(root) = " + std::to_string(f_c));
                    printVerbose("Final bracket width: " + std::to_string(bracket_width));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                    printVerbose("Function evaluations: " + 
                               std::to_string(function_evaluations_));
                }
                
                recordError(last_error_);
                return c;
            }
            
            // Update bracket based on sign of f(c)
            if (f_a * f_c < 0.0) {
                // Root is in [a, c]
                b = c;
                f_b = f_c;
            } else {
                // Root is in [c, b]
                a = c;
                f_a = f_c;
            }
            
            recordError(bracket_width);
        }
        
        // Maximum iterations reached
        last_root_ = c;
        last_error_ = std::abs(b - a);
        
        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best root estimate: " + std::to_string(c));
            printVerbose("f(c) = " + std::to_string(f_c));
            printVerbose("Final bracket width: " + std::to_string(std::abs(b - a)));
        }
        
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        
        throw ConvergenceException("Bisection: Maximum iterations reached", info);
    }
    
    double iterate() override {
        // Perform one bisection iteration
        double c = 0.5 * (a_ + b_);
        updateBracket(c);
        return c;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = (current_iterations_ < max_iterations_) ? 
                      SolverStatus::SUCCESS : SolverStatus::MAX_ITERATIONS;
        info.message = "Bisection method";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Bisection Method";
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
     * @brief Get number of iterations required for desired accuracy
     * @param a Left endpoint
     * @param b Right endpoint
     * @param tolerance Desired tolerance
     * @return Number of iterations needed
     */
    static int estimateIterations(double a, double b, double tolerance) {
        double bracket_width = std::abs(b - a);
        return static_cast<int>(std::ceil(std::log2(bracket_width / tolerance)));
    }

private:
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

} // namespace ode_solver