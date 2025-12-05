/**
 * @file Ridder.cpp
 * @brief Implementation of Ridder's method for root finding
 * @author Your Name
 * @version 1.0.0
 *
 * Ridder's method is an improvement over Regula Falsi that achieves exponential
 * convergence while maintaining the guaranteed convergence property of bracketing
 * methods. It combines bisection with false position using an exponential factor.
 *
 * Algorithm:
 * Given interval [a,b] where f(a)·f(b) < 0
 * 1. Compute midpoint m = (a + b)/2
 * 2. Fit exponential function through three points using false position on modified function
 * 3. Compute new approximation c from exponential fit
 * 4. Replace appropriate bracket
 *
 * Convergence: Exponential (~1.93 order for most functions)
 * Advantages: Fast convergence, guaranteed bracket, no derivatives needed
 * Disadvantages: More complex than bisection, requires 3 function evaluations per iteration
 *
 * The method works by fitting: f(x) ≈ A·e^(Bx) + C
 * And solving for root using false position on the exponential surface
 */

#include "ode_solver/RootFinders.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace ode_solver {

/**
 * @brief Ridder's method for root finding
 *
 * Combines the reliability of bracketing methods with exponential convergence
 * using an exponential fitting approach.
 */
class RidderMethod : public BracketingMethod {
public:
    /**
     * @brief Constructor
     * @param f Function to find root of
     * @param config Bracket configuration [a,b]
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit RidderMethod(Function f,
                          const BracketConfig& config,
                          double tolerance = 1e-6,
                          int max_iter = 1000)
        : BracketingMethod(std::move(f), config, tolerance, max_iter) {
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Ridder's Method");
            printVerbose("Bracket: [" + std::to_string(a_) + ", " +
                        std::to_string(b_) + "]");
            printVerbose("Initial f(a) = " + std::to_string(f_a_) +
                        ", f(b) = " + std::to_string(f_b_));
        }
    }

    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();

        if (verbose_) {
            printVerbose("Starting Ridder iteration");
        }

        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {

            // Check convergence on bracket width
            double bracket_width = std::abs(b_ - a_);
            if (bracket_width < tolerance_) {
                double c = 0.5 * (a_ + b_);
                last_root_ = c;
                last_error_ = bracket_width;

                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(c));
                    printVerbose("Bracket width: " + std::to_string(bracket_width));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                }

                recordError(last_error_);
                return c;
            }

            // Compute midpoint
            double m = 0.5 * (a_ + b_);
            double f_m = evaluateFunction(m);

            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                            ": bracket = [" + std::to_string(a_) + ", " +
                            std::to_string(b_) + "]");
            }

            // Compute Ridder's formula for new point
            // Uses exponential fitting through three points (a,f_a), (m,f_m), (b,f_b)
            double c = computeRidderPoint(a_, b_, m, f_a_, f_b_, f_m);

            if (!isFinite(c)) {
                throw NumericalInstabilityException(
                    "Ridder: Non-finite approximation");
            }

            // Ensure c is strictly between a and b
            if (c <= a_ || c >= b_) {
                // Fall back to bisection
                c = m;
                if (verbose_) {
                    printVerbose("WARNING: Ridder point outside bracket, using bisection point");
                }
            }

            double f_c = evaluateFunction(c);

            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Ridder point: c = " + std::to_string(c) +
                            ", f(c) = " + std::to_string(f_c));
            }

            // Check convergence at new point
            if (std::abs(f_c) < tolerance_) {
                last_root_ = c;
                last_error_ = std::abs(f_c);

                if (verbose_) {
                    printVerbose("Converged at computed point!");
                    printVerbose("Root: " + std::to_string(c));
                    printVerbose("f(root) = " + std::to_string(f_c));
                }

                recordError(last_error_);
                return c;
            }

            // Update bracket maintaining the property that f(a)·f(b) < 0
            if (f_a_ * f_c < 0) {
                // Root is in [a, c]
                b_ = c;
                f_b_ = f_c;
            } else if (f_m * f_c < 0) {
                // Root is in [m, c] or [c, m]
                if (c < m) {
                    b_ = m;
                    f_b_ = f_m;
                } else {
                    a_ = m;
                    f_a_ = f_m;
                }
            } else {
                // Root is in [c, b]
                a_ = c;
                f_a_ = f_c;
            }

            recordError(bracket_width);
        }

        // Maximum iterations reached
        double c = 0.5 * (a_ + b_);
        last_root_ = c;
        last_error_ = std::abs(b_ - a_);

        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
        }

        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        throw ConvergenceException("Ridder: Maximum iterations reached", info);
    }

    double iterate() override {
        double m = 0.5 * (a_ + b_);
        double f_m = evaluateFunction(m);

        double c = computeRidderPoint(a_, b_, m, f_a_, f_b_, f_m);

        if (!isFinite(c)) {
            throw NumericalInstabilityException(
                "Ridder: Non-finite value in iterate");
        }

        double f_c = evaluateFunction(c);
        updateBracket(c);

        current_x_ = c;
        return c;
    }

    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Ridder's Method";
        return info;
    }

    std::string getMethodName() const override {
        return "Ridder's Method";
    }

    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::SUPERLINEAR;  // ~1.93 order
    }

    bool requiresDerivative() const override {
        return false;
    }

    bool supportsComplexRoots() const override {
        return false;
    }

private:
    /**
     * @brief Compute Ridder's approximation using exponential fitting
     *
     * Ridder's formula:
     * Uses the three points (a,f_a), (m,f_m), (b,f_b) to fit an exponential
     * Q(x) = A·exp(B·x) and finds where it crosses zero.
     *
     * The formula derived is:
     * c = m + (m - a)·sign(f_a - f_b)·f_m / √(f_m² - f_a·f_b)
     *
     * This can be rewritten as:
     * c = m - f_m·(m - a) / (sign(f_a - f_b)·√(f_m² - f_a·f_b))
     */
    double computeRidderPoint(double a, double b, double m,
                             double f_a, double f_b, double f_m) {
        // Compute discriminant
        double discriminant = f_m * f_m - f_a * f_b;

        // Check for valid discriminant
        if (discriminant < 0) {
            // This shouldn't happen if bracket is valid, but handle gracefully
            if (verbose_) {
                printVerbose("WARNING: Negative discriminant in Ridder formula");
            }
            discriminant = 0;
        }

        double sqrt_disc = std::sqrt(discriminant);

        if (sqrt_disc < 1e-15) {
            // Fall back to bisection
            return m;
        }

        // Sign factor
        double sign_factor = (f_a - f_b < 0) ? -1.0 : 1.0;

        // Ridder's formula
        // c = m + (m - a)·sign·f_m / √(f_m² - f_a·f_b)
        double numerator = (m - a) * sign_factor * f_m;
        double c = m + numerator / sqrt_disc;

        return c;
    }

    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

} // namespace ode_solver