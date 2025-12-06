/**
 * @file Steffenson.cpp
 * @brief Implementation of Steffenson's method for root finding
 * @author Your Name
 * @version 1.0.0
 *
 * Steffenson's method (or Steffensen's method) uses fixed-point iteration combined
 * with Aitken's Δ² acceleration to achieve quadratic convergence without computing
 * derivatives. It's also known as the bracket-free method.
 *
 * Algorithm:
 * Given fixed-point problem x = g(x) where f(x) = x - g(x):
 * 1. Start with initial guess x₀
 * 2. Compute y = g(x_n)
 * 3. Compute z = g(y)
 * 4. Apply Aitken acceleration:
 *    x_{n+1} = x_n - (y - x_n)² / (z - 2y + x_n)
 *
 * Alternatively, for root finding directly:
 * 1. Start with x₀
 * 2. Compute y = f(x_n) + x_n
 * 3. Compute z = f(y) + y
 * 4. Use secant-like formula with Aitken acceleration
 *
 * Convergence: Quadratic (order 2) without computing derivatives
 * Advantages: Quadratic convergence without derivatives, 3 function evaluations per iteration
 * Disadvantages: Requires 3 function evaluations, may diverge on poor initial guesses
 *
 * The method can be viewed as Newton's method where the derivative is approximated
 * using the Aitken Δ² formula.
 */

#include "ode_solver/RootFinders.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace ode_solver {

/**
 * @brief Steffenson's method for root finding
 *
 * Combines fixed-point iteration with Aitken acceleration to achieve
 * quadratic convergence without derivatives.
 */
class SteffensonMethod : public OpenMethod {
public:
    enum class Mode {
        ROOT_FINDING,      ///< Direct root finding (solve f(x) = 0)
        FIXED_POINT        ///< Fixed-point iteration (solve x = g(x))
    };

    /**
     * @brief Constructor for root finding mode
     * @param f Function to find root of (for ROOT_FINDING mode)
     * @param x0 Initial guess
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit SteffensonMethod(Function f,
                             double x0,
                             double tolerance = 1e-6,
                             int max_iter = 1000)
        : OpenMethod(std::move(f), x0, tolerance, max_iter)
        , mode_(Mode::ROOT_FINDING)
        , fixed_point_func_(nullptr) {
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Steffenson's Method (Root Finding Mode)");
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
        }
    }

    /**
     * @brief Constructor for fixed-point iteration mode
     * @param g Fixed-point function (x = g(x))
     * @param x0 Initial guess
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit SteffensonMethod(Function g,
                             double x0,
                             const std::string& mode_label,  // Pass "fixed_point"
                             double tolerance = 1e-6,
                             int max_iter = 1000)
        : OpenMethod([g](double x) { return x - g(x); }, x0, tolerance, max_iter)
        , mode_(Mode::FIXED_POINT)
        , fixed_point_func_(std::move(g)) {
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Steffenson's Method (Fixed-Point Mode)");
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
        }
    }

    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();

        if (verbose_) {
            printVerbose("Starting Steffenson iteration");
        }

        double x = x0_;

        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {

            // Evaluate f at current point
            double f_x = evaluateFunction(x);

            // Check convergence on function value
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

            // Compute intermediate point y
            // In root finding mode: y = f(x) + x
            // In fixed-point mode: y = g(x)
            double y;
            if (mode_ == Mode::ROOT_FINDING) {
                y = x + f_x;
            } else {
                y = fixed_point_func_(x);
            }

            // Compute f at y
            double f_y = evaluateFunction(y);

            // Compute second intermediate point z
            double z;
            if (mode_ == Mode::ROOT_FINDING) {
                z = y + f_y;
            } else {
                z = fixed_point_func_(y);
            }

            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                            ": x = " + std::to_string(x) +
                            ", f(x) = " + std::to_string(f_x) +
                            ", y = " + std::to_string(y) +
                            ", f(y) = " + std::to_string(f_y));
            }

            // Compute Aitken Δ² acceleration
            double x_new = computeAitkenAcceleration(x, y, z, f_x, f_y);

            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Steffenson: Non-finite estimate");
            }

            // Check for divergence
            if (std::abs(x_new) > 1e10) {
                throw ConvergenceException("Steffenson: Diverging");
            }

            double step = std::abs(x_new - x);

            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Aitken step: x_new = " + std::to_string(x_new) +
                            ", |step| = " + std::to_string(step));
            }

            x = x_new;
            current_x_ = x;
            recordError(std::abs(f_x));
        }

        // Maximum iterations reached
        last_root_ = x;
        last_error_ = std::abs(evaluateFunction(x));

        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: " + std::to_string(x));
        }

        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        throw ConvergenceException("Steffenson: Maximum iterations reached", info);
    }

    double iterate() override {
        double x = current_x_;
        double f_x = evaluateFunction(x);

        double y;
        if (mode_ == Mode::ROOT_FINDING) {
            y = x + f_x;
        } else {
            y = fixed_point_func_(x);
        }

        double f_y = evaluateFunction(y);

        double z;
        if (mode_ == Mode::ROOT_FINDING) {
            z = y + f_y;
        } else {
            z = fixed_point_func_(y);
        }

        double x_new = computeAitkenAcceleration(x, y, z, f_x, f_y);

        if (!isFinite(x_new)) {
            throw NumericalInstabilityException(
                "Steffenson: Non-finite value in iterate");
        }

        current_x_ = x_new;
        return x_new;
    }

    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = (mode_ == Mode::ROOT_FINDING) ?
                       "Steffenson's Method (Root Finding)" :
                       "Steffenson's Method (Fixed-Point)";
        return info;
    }

    std::string getMethodName() const override {
        return (mode_ == Mode::ROOT_FINDING) ?
               "Steffenson's Method (Root Finding)" :
               "Steffenson's Method (Fixed-Point)";
    }

    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::QUADRATIC;  // Order 2 convergence
    }

    bool requiresDerivative() const override {
        return false;
    }

    bool supportsComplexRoots() const override {
        return false;
    }

    Mode getMode() const {
        return mode_;
    }

private:
    Mode mode_;
    Function fixed_point_func_;  ///< g(x) for fixed-point mode

    /**
     * @brief Compute Aitken Δ² acceleration
     *
     * Aitken's formula for accelerating convergence:
     * x_new = x - (y - x)² / (z - 2y + x)
     *       = (x·z - y²) / (z - 2y + x)
     *
     * This is equivalent to applying the Δ² operator:
     * Δy_n = y_{n+1} - y_n
     * Δ²y_n = Δy_{n+1} - Δy_n
     * x_new = y_n - (Δy_n)² / Δ²y_n
     */
    double computeAitkenAcceleration(double x, double y, double z,
                                    double f_x, double f_y) {
        // Denominator of Aitken formula: z - 2y + x
        double denominator = z - 2.0 * y + x;

        if (std::abs(denominator) < 1e-15) {
            if (verbose_) {
                printVerbose("WARNING: Aitken denominator too small, using simple step");
            }
            // Fall back to simpler formula
            double slope = (f_y - f_x) / (y - x + 1e-15);
            return y - f_y / (slope + 1e-15);
        }

        // Aitken formula: x_new = x - (y - x)² / (z - 2y + x)
        double numerator = (y - x) * (y - x);
        double x_new = x - numerator / denominator;

        return x_new;
    }

    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

} // namespace ode_solver