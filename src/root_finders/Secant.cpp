/**
 * @file Secant.cpp
 * @brief Implementation of the Secant method for root finding
 * @author Your Name
 * @version 1.0.0
 *
 * The Secant method is an open method that approximates the derivative using
 * finite differences, eliminating the need for explicit derivative computation.
 * It requires two initial guesses and uses a secant line approximation.
 *
 * Algorithm:
 * Given initial guesses x₀ and x₁:
 * x_{n+1} = x_n - f(x_n)·(x_n - x_{n-1})/(f(x_n) - f(x_{n-1}))
 *
 * The method can be viewed as approximating f'(x_n) with the slope of the
 * secant line through (x_{n-1}, f(x_{n-1})) and (x_n, f(x_n)).
 *
 * Convergence: Superlinear (~1.618 order - golden ratio)
 * Advantages: No derivative needed, fast convergence, only 1 new evaluation per iteration
 * Disadvantages: Not guaranteed to converge, can diverge from poor starting points
 *
 * Unlike Newton-Raphson which requires derivative, Secant approximates it,
 * making it practical for functions where derivatives are expensive or unavailable.
 */

#include "ode_solver/RootFinders.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace ode_solver {

/**
 * @brief Secant method for root finding
 *
 * Approximates Newton's method using secant line slope instead of derivative.
 * Requires two initial points.
 */
class SecantMethod : public OpenMethod {
public:
    /**
     * @brief Constructor
     * @param f Function to find root of
     * @param x0 First initial guess
     * @param x1 Second initial guess (if NaN, uses x0 + small step)
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit SecantMethod(Function f,
                         double x0,
                         double x1 = std::numeric_limits<double>::quiet_NaN(),
                         double tolerance = 1e-6,
                         int max_iter = 1000)
        : OpenMethod(std::move(f), x0, tolerance, max_iter)
        , x_prev_(std::isnan(x1) ? x0 + 0.001 * (std::abs(x0) + 1.0) : x1)
        , step_size_(0.001) {
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Secant Method");
            printVerbose("Initial guesses: x0 = " + std::to_string(x0_) +
                        ", x1 = " + std::to_string(x_prev_));
        }
    }

    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();

        if (verbose_) {
            printVerbose("Starting Secant iteration");
        }

        // Evaluate at both initial points
        double x_prev = x_prev_;
        double x_curr = x0_;
        double f_prev = evaluateFunction(x_prev);
        double f_curr = evaluateFunction(x_curr);

        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {

            // Check convergence on function value
            if (std::abs(f_curr) < tolerance_) {
                last_root_ = x_curr;
                last_error_ = std::abs(f_curr);

                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(x_curr));
                    printVerbose("f(root) = " + std::to_string(f_curr));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                    printVerbose("Function evaluations: " +
                                std::to_string(function_evaluations_));
                }

                recordError(last_error_);
                return x_curr;
            }

            // Compute secant slope: (f_curr - f_prev) / (x_curr - x_prev)
            double denominator = f_curr - f_prev;

            if (std::abs(denominator) < 1e-15) {
                throw NumericalInstabilityException(
                    "Secant: Function values too close (flat slope)");
            }

            if (std::abs(x_curr - x_prev) < 1e-15) {
                throw NumericalInstabilityException(
                    "Secant: X values too close (degenerate step)");
            }

            double slope = denominator / (x_curr - x_prev);

            // Newton-like step: x_new = x_curr - f_curr / slope
            double x_new = x_curr - f_curr / slope;

            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Secant: Non-finite next estimate");
            }

            // Check for divergence
            if (std::abs(x_new) > 1e10) {
                throw ConvergenceException("Secant: Diverging (estimate too large)");
            }

            double f_new = evaluateFunction(x_new);

            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                            ": x = " + std::to_string(x_new) +
                            ", f(x) = " + std::to_string(f_new) +
                            ", |step| = " + std::to_string(std::abs(x_new - x_curr)) +
                            ", slope = " + std::to_string(slope));
            }

            // Update for next iteration
            x_prev = x_curr;
            f_prev = f_curr;
            x_curr = x_new;
            f_curr = f_new;

            current_x_ = x_curr;
            recordError(std::abs(f_curr));
        }

        // Maximum iterations reached
        last_root_ = x_curr;
        last_error_ = std::abs(f_curr);

        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: " + std::to_string(x_curr));
        }

        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        throw ConvergenceException("Secant: Maximum iterations reached", info);
    }

    double iterate() override {
        double f_prev = evaluateFunction(x_prev_);
        double f_curr = evaluateFunction(current_x_);

        double denominator = f_curr - f_prev;
        if (std::abs(denominator) < 1e-15) {
            throw NumericalInstabilityException(
                "Secant: Denominator too small in iterate");
        }

        double slope = denominator / (current_x_ - x_prev_);

        double x_new = current_x_ - f_curr / slope;

        if (!isFinite(x_new)) {
            throw NumericalInstabilityException(
                "Secant: Non-finite value in iterate");
        }

        x_prev_ = current_x_;
        current_x_ = x_new;

        return x_new;
    }

    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Secant Method";
        return info;
    }

    std::string getMethodName() const override {
        return "Secant Method";
    }

    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::SUPERLINEAR;  // ~1.618 order (golden ratio)
    }

    bool requiresDerivative() const override {
        return false;
    }

    bool supportsComplexRoots() const override {
        return false;
    }

    /**
     * @brief Get the previous x value
     */
    double getPreviousX() const {
        return x_prev_;
    }

    /**
     * @brief Set step size for initial second point generation
     */
    void setStepSize(double h) {
        if (h <= 0) {
            throw InvalidParameterException("Step size must be positive");
        }
        step_size_ = h;
    }

private:
    double x_prev_;    ///< Previous x value (x_{n-1})
    double step_size_; ///< Step size for initial point generation

    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

} // namespace ode_solver
