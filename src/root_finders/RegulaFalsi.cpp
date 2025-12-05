/**
 * @file RegulaFalsi.cpp
 * @brief Implementation of Regula Falsi (False Position) method for root finding
 * @author Your Name
 * @version 1.0.0
 *
 * Regula Falsi is a bracketing method that uses linear interpolation between
 * function values at the bracket endpoints. It's guaranteed to converge but
 * can exhibit slow convergence near roots (linear convergence).
 *
 * Algorithm:
 * Given interval [a,b] where f(a)·f(b) < 0
 * Compute intersection of secant line through (a,f(a)) and (b,f(b)) with x-axis:
 * c = a - f(a)·(b - a)/(f(b) - f(a))
 * Replace bracket: [a,c] or [c,b] based on sign of f(c)
 *
 * Convergence: Linear (~1.0 order) but often faster in practice than bisection
 * Advantages: Guaranteed convergence, often faster than bisection
 * Disadvantages: Can get stuck if one endpoint has very small function value
 *
 * Variants implemented:
 * - Standard: Both endpoints updated each iteration
 * - Illinois: Halve function value at sticky endpoint
 * - Pegasus: Use weighted function value for sticky endpoint
 */

#include "ode_solver/RootFinders.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace ode_solver {

/**
 * @brief Regula Falsi (False Position) method
 *
 * A bracketing method using linear interpolation with various variants
 * to prevent endpoint stalling.
 */
class RegulaFalsiMethod : public BracketingMethod {
public:
    enum class Variant {
        STANDARD,  ///< Standard false position
        ILLINOIS,  ///< Illinois variant - halve stuck endpoint
        PEGASUS    ///< Pegasus variant - use weighted function value
    };

    /**
     * @brief Constructor
     * @param f Function to find root of
     * @param config Bracket configuration [a,b]
     * @param variant Which variant to use
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit RegulaFalsiMethod(Function f,
                               const BracketConfig& config,
                               Variant variant = Variant::ILLINOIS,
                               double tolerance = 1e-6,
                               int max_iter = 1000)
        : BracketingMethod(std::move(f), config, tolerance, max_iter)
        , variant_(variant)
        , iteration_count_(0) {
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Regula Falsi Method");
            printVerbose("Variant: " + getVariantName());
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
            printVerbose("Starting Regula Falsi iteration");
        }

        // Store original function values for Pegasus variant
        double f_a_orig = f_a_;
        double f_b_orig = f_b_;
        int stuck_side = 0;  // 0: none, -1: left, 1: right

        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {

            // Check for convergence
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

            // Compute false position (secant intersection with x-axis)
            double c = computeFalsePosition(a_, b_, f_a_, f_b_);

            if (!isFinite(c)) {
                throw NumericalInstabilityException(
                    "RegulaFalsi: Non-finite false position");
            }

            // Ensure c is strictly between a and b
            if (c <= a_ || c >= b_) {
                // Fall back to bisection if something goes wrong
                c = 0.5 * (a_ + b_);
                if (verbose_) {
                    printVerbose("WARNING: False position outside bracket, using bisection");
                }
            }

            double f_c = evaluateFunction(c);

            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                            ": c = " + std::to_string(c) +
                            ", f(c) = " + std::to_string(f_c) +
                            ", width = " + std::to_string(std::abs(b_ - a_)));
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

            // Update bracket and apply variant-specific logic
            if (f_a_ * f_c < 0) {
                // Root is in [a, c]
                b_ = c;
                f_b_ = f_c;

                // Handle sticky left endpoint
                if (stuck_side == -1) {
                    applyVariantCorrection(true);  // Left side is sticky
                }
                stuck_side = 0;

            } else {
                // Root is in [c, b]
                a_ = c;
                f_a_ = f_c;

                // Handle sticky right endpoint
                if (stuck_side == 1) {
                    applyVariantCorrection(false);  // Right side is sticky
                }
                stuck_side = 0;
            }

            recordError(bracket_width);
        }

        // Maximum iterations reached
        double c = 0.5 * (a_ + b_);
        last_root_ = c;
        last_error_ = std::abs(b_ - a_);

        if (verbose_) {
            printVerbose("WARNING: Maximum iterations reached");
            printVerbose("Best estimate: " + std::to_string(c));
        }

        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::MAX_ITERATIONS;
        info.message = "Maximum iterations reached";
        info.execution_time_ms = getElapsedTime(timer_start);
        throw ConvergenceException("RegulaFalsi: Maximum iterations reached", info);
    }

    double iterate() override {
        // Single iteration
        double c = computeFalsePosition(a_, b_, f_a_, f_b_);

        if (!isFinite(c)) {
            throw NumericalInstabilityException(
                "RegulaFalsi: Non-finite value in iterate");
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
        info.message = "Regula Falsi (" + getVariantName() + ")";
        return info;
    }

    std::string getMethodName() const override {
        return "Regula Falsi (" + getVariantName() + ")";
    }

    ConvergenceRate getConvergenceRate() const override {
        // Standard variant is linear, others are superlinear in practice
        return ConvergenceRate::LINEAR;
    }

    bool requiresDerivative() const override {
        return false;
    }

    bool supportsComplexRoots() const override {
        return false;
    }

    /**
     * @brief Set variant to use
     */
    void setVariant(Variant variant) {
        variant_ = variant;
    }

    Variant getVariant() const {
        return variant_;
    }

private:
    Variant variant_;
    int iteration_count_;

    /**
     * @brief Compute false position (secant intersection with x-axis)
     *
     * Formula: c = a - f(a)·(b - a) / (f(b) - f(a))
     *        = (a·f(b) - b·f(a)) / (f(b) - f(a))
     */
    double computeFalsePosition(double a, double b, double f_a, double f_b) {
        double denominator = f_b - f_a;

        if (std::abs(denominator) < 1e-15) {
            throw NumericalInstabilityException(
                "RegulaFalsi: Function values too close");
        }

        // Use second form to avoid cancellation
        double c = (a * f_b - b * f_a) / denominator;

        return c;
    }

    /**
     * @brief Apply variant-specific correction to prevent endpoint stalling
     * @param left_sticky If true, left endpoint is sticky; else right
     */
    void applyVariantCorrection(bool left_sticky) {
        switch (variant_) {
            case Variant::STANDARD:
                // No correction
                break;

            case Variant::ILLINOIS:
                // Halve the function value at sticky endpoint
                if (left_sticky) {
                    f_a_ *= 0.5;
                } else {
                    f_b_ *= 0.5;
                }
                if (verbose_) {
                    printVerbose("Applied Illinois correction");
                }
                break;

            case Variant::PEGASUS:
                // Use weighted average for sticky endpoint
                if (left_sticky) {
                    double f_a_new = evaluateFunction(a_);
                    if (f_a_new != f_a_) {
                        // Weight: ratio of new to old function value
                        f_a_ = f_a_ * f_a_new / (f_a_ + f_a_new);
                    }
                } else {
                    double f_b_new = evaluateFunction(b_);
                    if (f_b_new != f_b_) {
                        f_b_ = f_b_ * f_b_new / (f_b_ + f_b_new);
                    }
                }
                if (verbose_) {
                    printVerbose("Applied Pegasus correction");
                }
                break;
        }
    }

    std::string getVariantName() const {
        switch (variant_) {
            case Variant::STANDARD: return "Standard";
            case Variant::ILLINOIS: return "Illinois";
            case Variant::PEGASUS: return "Pegasus";
            default: return "Unknown";
        }
    }

    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

} // namespace ode_solver