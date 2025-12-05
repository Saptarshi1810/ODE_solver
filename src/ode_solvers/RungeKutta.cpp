/**
 * @file RungeKutta.cpp
 * @brief Implementation of Runge-Kutta ODE solvers
 * @author Your Name
 * @version 1.0.0
 */

#include "ode_solver/ODESolvers.hpp"
#include "ode_solver/EquationSolver.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ode_solver {

// ============================================================================
// RungeKutta4 Implementation
// ============================================================================

/**
 * @brief Classical 4th-order Runge-Kutta method
 * 
 * The workhorse ODE solver with excellent balance of accuracy and efficiency.
 */
class RungeKutta4 : public ExplicitODESolver {
public:
    explicit RungeKutta4(ODESystem system, double h = 0.01)
        : ExplicitODESolver(std::move(system), h) {}
    
    std::vector<double> step(double t, const std::vector<double>& y) override {
        const size_t n = y.size();
        std::vector<double> y_next(n);
        std::vector<double> y_temp(n);
        
        // k1 = f(t, y)
        auto k1 = system_(t, y);
        
        // k2 = f(t + h/2, y + h*k1/2)
        for (size_t i = 0; i < n; ++i) {
            y_temp[i] = y[i] + 0.5 * step_size_ * k1[i];
        }
        auto k2 = system_(t + 0.5 * step_size_, y_temp);
        
        // k3 = f(t + h/2, y + h*k2/2)
        for (size_t i = 0; i < n; ++i) {
            y_temp[i] = y[i] + 0.5 * step_size_ * k2[i];
        }
        auto k3 = system_(t + 0.5 * step_size_, y_temp);
        
        // k4 = f(t + h, y + h*k3)
        for (size_t i = 0; i < n; ++i) {
            y_temp[i] = y[i] + step_size_ * k3[i];
        }
        auto k4 = system_(t + step_size_, y_temp);
        
        // Combine: y_next = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        for (size_t i = 0; i < n; ++i) {
            y_next[i] = y[i] + (step_size_ / 6.0) * 
                       (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        }
        
        // Check for numerical issues
        if (!isFinite(y_next)) {
            throw NumericalInstabilityException(
                "RK4: Non-finite values detected in solution");
        }
        
        return y_next;
    }
    
    ODESolution solve(double t_start, double t_end, 
                     const std::vector<double>& y0) override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        ODESolution solution;
        solution.dimension = static_cast<int>(y0.size());
        
        // Validate inputs
        validateSolveInputs(t_start, t_end, y0);
        
        // Determine direction and adjust step size
        double direction = (t_end > t_start) ? 1.0 : -1.0;
        double h = direction * std::abs(step_size_);
        
        // Initialize solution
        double t = t_start;
        std::vector<double> y = y0;
        
        solution.t_values.push_back(t);
        solution.y_values.push_back(y);
        
        if (verbose_) {
            printVerbose("Starting RK4 integration from t=" + 
                        std::to_string(t_start) + " to t=" + std::to_string(t_end));
            printVerbose("Step size h=" + std::to_string(h) + 
                        ", System dimension=" + std::to_string(y0.size()));
        }
        
        // Integration loop
        while ((direction > 0 && t < t_end) || (direction < 0 && t > t_end)) {
            // Adjust last step to hit t_end exactly
            if ((direction > 0 && t + h > t_end) || 
                (direction < 0 && t + h < t_end)) {
                h = t_end - t;
            }
            
            // Store old step size and restore after
            double old_h = step_size_;
            step_size_ = h;
            
            // Take a step
            y = step(t, y);
            t += h;
            
            step_size_ = old_h;
            
            // Store solution
            solution.t_values.push_back(t);
            solution.y_values.push_back(y);
            
            // Check for events
            if (event_config_ && solution.t_values.size() >= 2) {
                auto event_time = detectEvent(
                    solution.t_values[solution.t_values.size() - 2],
                    solution.y_values[solution.y_values.size() - 2],
                    t, y
                );
                
                if (event_time) {
                    solution.event_times.push_back(*event_time);
                    if (verbose_) {
                        printVerbose("Event detected at t=" + std::to_string(*event_time));
                    }
                    if (event_config_->stop_at_event) {
                        if (verbose_) {
                            printVerbose("Stopping integration at event");
                        }
                        break;
                    }
                }
            }
            
            incrementIterations();
            
            if (verbose_ && current_iterations_ % 100 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) + 
                           ", t=" + std::to_string(t));
            }
            
            if (maxIterationsExceeded()) {
                solution.convergence_info.status = SolverStatus::MAX_ITERATIONS;
                solution.convergence_info.message = "Maximum iterations reached";
                if (verbose_) {
                    printVerbose("WARNING: Maximum iterations exceeded");
                }
                break;
            }
        }
        
        // Fill convergence info
        solution.convergence_info.iterations = current_iterations_;
        solution.convergence_info.execution_time_ms = getElapsedTime(timer_start);
        if (solution.convergence_info.status != SolverStatus::MAX_ITERATIONS) {
            solution.convergence_info.status = SolverStatus::SUCCESS;
            solution.convergence_info.message = "Integration completed successfully";
        }
        solution.total_steps = current_iterations_;
        solution.avg_step_size = std::abs(t_end - t_start) / current_iterations_;
        
        if (verbose_) {
            printVerbose("Integration complete: " + 
                        std::to_string(current_iterations_) + " steps in " +
                        std::to_string(solution.convergence_info.execution_time_ms) + " ms");
        }
        
        return solution;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        info.message = "RK4 integration";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Runge-Kutta 4";
    }
    
    int getMethodOrder() const override {
        return 4;
    }
    
    int getFunctionEvaluationsPerStep() const override {
        return 4; // k1, k2, k3, k4
    }

private:
    int function_evaluations_ = 0;
    
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

// ============================================================================
// AdaptiveRK45 Implementation (Dormand-Prince)
// ============================================================================

/**
 * @brief Adaptive Runge-Kutta 4-5 method with Dormand-Prince coefficients
 * 
 * Embedded method with automatic step-size control.
 */
class AdaptiveRK45 : public ExplicitODESolver {
private:
    // Dormand-Prince coefficients
    static constexpr double a2 = 1.0/5.0;
    static constexpr double a3 = 3.0/10.0;
    static constexpr double a4 = 4.0/5.0;
    static constexpr double a5 = 8.0/9.0;
    static constexpr double a6 = 1.0;
    static constexpr double a7 = 1.0;
    
    static constexpr double b21 = 1.0/5.0;
    static constexpr double b31 = 3.0/40.0, b32 = 9.0/40.0;
    static constexpr double b41 = 44.0/45.0, b42 = -56.0/15.0, b43 = 32.0/9.0;
    static constexpr double b51 = 19372.0/6561.0, b52 = -25360.0/2187.0;
    static constexpr double b53 = 64448.0/6561.0, b54 = -212.0/729.0;
    static constexpr double b61 = 9017.0/3168.0, b62 = -355.0/33.0;
    static constexpr double b63 = 46732.0/5247.0, b64 = 49.0/176.0, b65 = -5103.0/18656.0;
    static constexpr double b71 = 35.0/384.0, b73 = 500.0/1113.0;
    static constexpr double b74 = 125.0/192.0, b75 = -2187.0/6784.0, b76 = 11.0/84.0;
    
    // 5th order solution coefficients
    static constexpr double c1 = 35.0/384.0, c3 = 500.0/1113.0;
    static constexpr double c4 = 125.0/192.0, c5 = -2187.0/6784.0, c6 = 11.0/84.0;
    
    // 4th order solution coefficients (for error estimate)
    static constexpr double d1 = 5179.0/57600.0, d3 = 7571.0/16695.0;
    static constexpr double d4 = 393.0/640.0, d5 = -92097.0/339200.0;
    static constexpr double d6 = 187.0/2100.0, d7 = 1.0/40.0;
    
public:
    explicit AdaptiveRK45(ODESystem system, double h = 0.01, double tolerance = 1e-6)
        : ExplicitODESolver(std::move(system), h, tolerance)
        , step_accepts_(0)
        , step_rejects_(0) {
        adaptive_step_enabled_ = true;
        adaptive_config_.error_tolerance = tolerance;
    }
    
    std::vector<double> step(double t, const std::vector<double>& y) override {
        // For non-adaptive use, just return one step with current h
        double h = step_size_;
        return adaptiveStep(t, y, h).first;
    }
    
    /**
     * @brief Take one adaptive step
     * @param t Current time
     * @param y Current state
     * @param h Current step size (will be modified)
     * @return Pair of (new state, accepted step size)
     */
    std::pair<std::vector<double>, double> adaptiveStep(
        double t, const std::vector<double>& y, double& h) {
        
        const size_t n = y.size();
        std::vector<double> y_temp(n);
        
        int reject_count = 0;
        
        while (reject_count < adaptive_config_.max_step_rejects) {
            // Compute the 7 k stages
            auto k1 = system_(t, y);
            
            // k2
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + h * b21 * k1[i];
            auto k2 = system_(t + a2*h, y_temp);
            
            // k3
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + h * (b31*k1[i] + b32*k2[i]);
            auto k3 = system_(t + a3*h, y_temp);
            
            // k4
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + h * (b41*k1[i] + b42*k2[i] + b43*k3[i]);
            auto k4 = system_(t + a4*h, y_temp);
            
            // k5
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + h * (b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]);
            auto k5 = system_(t + a5*h, y_temp);
            
            // k6
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + h * (b61*k1[i] + b62*k2[i] + b63*k3[i] + 
                                        b64*k4[i] + b65*k5[i]);
            auto k6 = system_(t + a6*h, y_temp);
            
            // k7
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + h * (b71*k1[i] + b73*k3[i] + b74*k4[i] + 
                                        b75*k5[i] + b76*k6[i]);
            auto k7 = system_(t + a7*h, y_temp);
            
            // Compute 5th order solution
            std::vector<double> y5(n);
            for (size_t i = 0; i < n; ++i) {
                y5[i] = y[i] + h * (c1*k1[i] + c3*k3[i] + c4*k4[i] + 
                                    c5*k5[i] + c6*k6[i]);
            }
            
            // Compute 4th order solution for error estimate
            std::vector<double> y4(n);
            for (size_t i = 0; i < n; ++i) {
                y4[i] = y[i] + h * (d1*k1[i] + d3*k3[i] + d4*k4[i] + 
                                    d5*k5[i] + d6*k6[i] + d7*k7[i]);
            }
            
            // Estimate error
            double error = estimateError(y4, y5);
            
            // Check for numerical issues
            if (!isFinite(y5)) {
                throw NumericalInstabilityException(
                    "RK45: Non-finite values detected");
            }
            
            // Adjust step size
            if (error < adaptive_config_.error_tolerance || error < 1e-15) {
                // Accept step
                double h_new = computeOptimalStepSize(error, h, 4);
                
                // Don't increase too aggressively
                h_new = std::min(h_new, 5.0 * h);
                
                // Clamp to bounds
                h = std::max(adaptive_config_.min_step, 
                            std::min(adaptive_config_.max_step, h_new));
                
                ++step_accepts_;
                
                if (verbose_ && step_accepts_ % 50 == 0) {
                    printVerbose("RK45: " + std::to_string(step_accepts_) + 
                               " accepts, " + std::to_string(step_rejects_) + 
                               " rejects, h=" + std::to_string(h));
                }
                
                return {y5, h};
            } else {
                // Reject step, reduce h
                ++step_rejects_;
                ++reject_count;
                
                double h_new = computeOptimalStepSize(error, h, 4);
                h = std::max(h_new, adaptive_config_.min_step);
                
                if (verbose_) {
                    printVerbose("RK45: Step rejected, error=" + std::to_string(error) + 
                               ", new h=" + std::to_string(h));
                }
            }
        }
        
        throw ConvergenceException(
            "RK45: Too many consecutive step rejections (" + 
            std::to_string(reject_count) + ")");
    }
    
    ODESolution solve(double t_start, double t_end, 
                     const std::vector<double>& y0) override {
        auto timer_start = startTimer();
        resetIterationCount();
        step_accepts_ = 0;
        step_rejects_ = 0;
        
        ODESolution solution;
        solution.dimension = static_cast<int>(y0.size());
        
        // Validate inputs
        validateSolveInputs(t_start, t_end, y0);
        
        // Initialize
        double t = t_start;
        std::vector<double> y = y0;
        double h = step_size_;
        
        solution.t_values.push_back(t);
        solution.y_values.push_back(y);
        
        if (verbose_) {
            printVerbose("Starting RK45 adaptive integration");
            printVerbose("Initial h=" + std::to_string(h) + 
                        ", tolerance=" + std::to_string(adaptive_config_.error_tolerance));
        }
        
        // Integration loop
        while (t < t_end) {
            // Don't step past t_end
            if (t + h > t_end) {
                h = t_end - t;
            }
            
            // Take adaptive step
            auto [y_new, h_new] = adaptiveStep(t, y, h);
            
            t += h;
            y = y_new;
            h = h_new;
            
            // Store solution
            solution.t_values.push_back(t);
            solution.y_values.push_back(y);
            
            // Check for events
            if (event_config_ && solution.t_values.size() >= 2) {
                auto event_time = detectEvent(
                    solution.t_values[solution.t_values.size() - 2],
                    solution.y_values[solution.y_values.size() - 2],
                    t, y
                );
                
                if (event_time) {
                    solution.event_times.push_back(*event_time);
                    if (event_config_->stop_at_event) {
                        break;
                    }
                }
            }
            
            incrementIterations();
            
            if (maxIterationsExceeded()) {
                solution.convergence_info.status = SolverStatus::MAX_ITERATIONS;
                solution.convergence_info.message = "Maximum iterations reached";
                break;
            }
        }
        
        // Fill convergence info
        solution.convergence_info.iterations = current_iterations_;
        solution.convergence_info.execution_time_ms = getElapsedTime(timer_start);
        if (solution.convergence_info.status != SolverStatus::MAX_ITERATIONS) {
            solution.convergence_info.status = SolverStatus::SUCCESS;
            solution.convergence_info.message = "Adaptive integration completed";
        }
        solution.total_steps = step_accepts_;
        solution.avg_step_size = (t_end - t_start) / step_accepts_;
        
        if (verbose_) {
            printVerbose("RK45 Complete: " + std::to_string(step_accepts_) + 
                        " accepts, " + std::to_string(step_rejects_) + " rejects");
            printVerbose("Average step size: " + std::to_string(solution.avg_step_size));
        }
        
        return solution;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        info.message = "RK45 adaptive: " + std::to_string(step_accepts_) + 
                      " accepts, " + std::to_string(step_rejects_) + " rejects";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Adaptive RK45 (Dormand-Prince)";
    }
    
    int getMethodOrder() const override {
        return 4; // 4th order accurate
    }
    
    int getFunctionEvaluationsPerStep() const override {
        return 6; // FSAL property: 6 evaluations per step
    }
    
    /**
     * @brief Get step acceptance statistics
     */
    std::pair<int, int> getStepStatistics() const {
        return {step_accepts_, step_rejects_};
    }

private:
    int step_accepts_;
    int step_rejects_;
};

} // namespace ode_solver   