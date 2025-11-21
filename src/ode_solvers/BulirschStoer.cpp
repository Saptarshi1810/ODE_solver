/**
 * @file BulirschStoer.cpp
 * @brief Implementation of Bulirsch-Stoer extrapolation method
 * @author Your Name
 * @version 1.0.0
 * 
 * Bulirsch-Stoer is a high-accuracy method based on Richardson extrapolation
 * of modified midpoint method results. Excellent for problems requiring
 * very high precision.
 */

#include "ode_solver/ODESolvers.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace ode_solver {

/**
 * @brief Bulirsch-Stoer method with adaptive step control
 * 
 * Uses Richardson extrapolation on modified midpoint method to achieve
 * very high accuracy. Particularly effective for smooth problems.
 */
class BulirschStoer : public ExplicitODESolver {
private:
    // Sequence of substep counts for extrapolation
    // Common choice: 2, 4, 6, 8, 12, 16, 24, 32, ...
    static constexpr int default_sequence[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    static constexpr int sequence_length = 10;
    
public:
    /**
     * @brief Constructor
     * @param system ODE system
     * @param h Initial step size
     * @param tolerance Error tolerance
     * @param max_iter Maximum iterations
     */
    explicit BulirschStoer(ODESystem system,
                          double h = 0.01,
                          double tolerance = 1e-10,
                          int max_iter = 100000)
        : ExplicitODESolver(std::move(system), h, tolerance)
        , max_extrapolations_(8) {
        
        // Enable adaptive stepping by default
        adaptive_step_enabled_ = true;
        adaptive_config_.error_tolerance = tolerance;
        adaptive_config_.min_step = h * 1e-6;
        adaptive_config_.max_step = h * 100;
    }
    
    /**
     * @brief Modified midpoint method - the base integrator
     * @param t0 Initial time
     * @param y0 Initial state
     * @param H Total step size
     * @param n Number of substeps
     * @return Solution at t0 + H
     */
    std::vector<double> modifiedMidpoint(double t0, 
                                        const std::vector<double>& y0,
                                        double H, int n) {
        const size_t dim = y0.size();
        double h = H / n;  // Substep size
        
        // Initial step
        std::vector<double> y_m1 = y0;  // y_{m-1}
        auto f0 = system_(t0, y0);
        
        std::vector<double> y_m(dim);
        for (size_t i = 0; i < dim; ++i) {
            y_m[i] = y0[i] + h * f0[i];  // y_m = y_0 + h*f(t_0, y_0)
        }
        
        // Main loop
        double t = t0 + h;
        for (int m = 1; m < n; ++m) {
            auto f = system_(t, y_m);
            
            std::vector<double> y_new(dim);
            for (size_t i = 0; i < dim; ++i) {
                y_new[i] = y_m1[i] + 2.0 * h * f[i];
            }
            
            y_m1 = y_m;
            y_m = y_new;
            t += h;
        }
        
        // Final smoothing step
        auto f_n = system_(t0 + H, y_m);
        std::vector<double> y_final(dim);
        for (size_t i = 0; i < dim; ++i) {
            y_final[i] = 0.5 * (y_m[i] + y_m1[i] + h * f_n[i]);
        }
        
        return y_final;
    }
    
    /**
     * @brief Polynomial extrapolation to h=0 using Neville's algorithm
     * @param table Extrapolation table (modified in place)
     * @param n_used Number of points used
     * @return Extrapolated value and error estimate
     */
    std::pair<std::vector<double>, double> polynomialExtrapolation(
        std::vector<std::vector<std::vector<double>>>& table,
        int n_used) {
        
        const size_t dim = table[0][0].size();
        
        // Neville's algorithm for polynomial extrapolation
        // T[k][m] = (h[k]^2 * T[k-1][m] - h[k-m]^2 * T[k-1][m-1]) / (h[k]^2 - h[k-m]^2)
        
        for (int m = 1; m < n_used; ++m) {
            for (int k = m; k < n_used; ++k) {
                double h_k = 1.0 / (default_sequence[k] * default_sequence[k]);
                double h_km = 1.0 / (default_sequence[k-m] * default_sequence[k-m]);
                
                for (size_t i = 0; i < dim; ++i) {
                    table[k][m][i] = table[k][m-1][i] + 
                        (table[k][m-1][i] - table[k-1][m-1][i]) / 
                        (h_km / h_k - 1.0);
                }
            }
        }
        
        // Estimate error from difference between last two diagonal elements
        double error = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            error = std::max(error, 
                std::abs(table[n_used-1][n_used-1][i] - 
                        table[n_used-2][n_used-2][i]));
        }
        
        return {table[n_used-1][n_used-1], error};
    }
    
    std::vector<double> step(double t, const std::vector<double>& y) override {
        // For single step, use adaptive step with current h
        double h = step_size_;
        return adaptiveStep(t, y, h).first;
    }
    
    /**
     * @brief Take one adaptive Bulirsch-Stoer step
     * @param t Current time
     * @param y Current state
     * @param h Current step size (modified)
     * @return Pair of (solution, new step size)
     */
    std::pair<std::vector<double>, double> adaptiveStep(
        double t, const std::vector<double>& y, double& h) {
        
        const size_t dim = y.size();
        
        // Extrapolation table: table[k][m] is extrapolation using
        // sequences 0..k with polynomial order m
        std::vector<std::vector<std::vector<double>>> table(
            max_extrapolations_,
            std::vector<std::vector<double>>(
                max_extrapolations_,
                std::vector<double>(dim)
            )
        );
        
        int reject_count = 0;
        
        while (reject_count < adaptive_config_.max_step_rejects) {
            // Try increasing orders of extrapolation
            bool converged = false;
            int n_used = 0;
            
            for (int k = 0; k < max_extrapolations_; ++k) {
                // Compute solution using k-th substep sequence
                int n_steps = default_sequence[k];
                table[k][0] = modifiedMidpoint(t, y, h, n_steps);
                
                if (k == 0) {
                    n_used = 1;
                    continue;
                }
                
                n_used = k + 1;
                
                // Perform extrapolation
                auto [y_extrap, error] = polynomialExtrapolation(table, n_used);
                
                // Check convergence
                if (error < adaptive_config_.error_tolerance) {
                    // Accept step
                    converged = true;
                    
                    // Adjust step size for next step
                    double safety = 0.9;
                    double fac = safety * std::pow(
                        adaptive_config_.error_tolerance / std::max(error, 1e-15),
                        1.0 / (2 * n_used + 1)
                    );
                    
                    // Don't increase too aggressively
                    fac = std::min(fac, 5.0);
                    fac = std::max(fac, 0.1);
                    
                    h = h * fac;
                    h = std::min(h, adaptive_config_.max_step);
                    h = std::max(h, adaptive_config_.min_step);
                    
                    if (verbose_ && reject_count > 0) {
                        printVerbose("BS: Step accepted after " + 
                                   std::to_string(reject_count) + 
                                   " rejections, order=" + std::to_string(n_used));
                    }
                    
                    return {y_extrap, h};
                }
            }
            
            if (!converged) {
                // Reduce step size and try again
                h *= 0.5;
                h = std::max(h, adaptive_config_.min_step);
                ++reject_count;
                
                if (verbose_) {
                    printVerbose("BS: Step rejected, reducing h to " + 
                               std::to_string(h));
                }
            }
        }
        
        throw ConvergenceException(
            "Bulirsch-Stoer: Failed to converge after " + 
            std::to_string(reject_count) + " rejections");
    }
    
    ODESolution solve(double t_start, double t_end,
                     const std::vector<double>& y0) override {
        auto timer_start = startTimer();
        resetIterationCount();
        
        ODESolution solution;
        solution.dimension = static_cast<int>(y0.size());
        
        validateSolveInputs(t_start, t_end, y0);
        
        double t = t_start;
        std::vector<double> y = y0;
        double h = step_size_;
        
        solution.t_values.push_back(t);
        solution.y_values.push_back(y);
        
        if (verbose_) {
            printVerbose("Starting Bulirsch-Stoer integration");
            printVerbose("Initial h=" + std::to_string(h) + 
                        ", tolerance=" + std::to_string(adaptive_config_.error_tolerance));
        }
        
        int step_accepts = 0;
        int step_rejects = 0;
        
        // Integration loop
        while (t < t_end) {
            // Don't overshoot
            if (t + h > t_end) {
                h = t_end - t;
            }
            
            try {
                // Take adaptive step
                auto [y_new, h_new] = adaptiveStep(t, y, h);
                
                t += h;
                y = y_new;
                h = h_new;
                
                ++step_accepts;
                
                // Store solution
                solution.t_values.push_back(t);
                solution.y_values.push_back(y);
                
                // Event detection
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
                
                if (verbose_ && step_accepts % 50 == 0) {
                    printVerbose("BS: " + std::to_string(step_accepts) + 
                               " accepts, h=" + std::to_string(h) + 
                               ", t=" + std::to_string(t));
                }
                
            } catch (const ConvergenceException& e) {
                ++step_rejects;
                if (step_rejects > 100) {
                    throw;
                }
                // Continue with reduced step size
                h *= 0.5;
            }
            
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
            solution.convergence_info.message = 
                "Bulirsch-Stoer: " + std::to_string(step_accepts) + 
                " accepts, " + std::to_string(step_rejects) + " rejects";
        }
        solution.total_steps = step_accepts;
        solution.avg_step_size = (t_end - t_start) / step_accepts;
        
        if (verbose_) {
            printVerbose("Bulirsch-Stoer complete");
            printVerbose("Steps: " + std::to_string(step_accepts) + 
                        " accepts, " + std::to_string(step_rejects) + " rejects");
            printVerbose("Average step size: " + 
                        std::to_string(solution.avg_step_size));
        }
        
        return solution;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Bulirsch-Stoer extrapolation";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Bulirsch-Stoer";
    }
    
    int getMethodOrder() const override {
        // Variable order, typically very high
        return 2 * max_extrapolations_; // Approximate
    }
    
    int getFunctionEvaluationsPerStep() const override {
        // Variable, depends on extrapolation order achieved
        // Rough estimate: sum of first few sequence elements
        return default_sequence[0] + default_sequence[1] + default_sequence[2];
    }
    
    /**
     * @brief Set maximum number of extrapolations to attempt
     * @param max_extrap Maximum order (2-10 recommended)
     */
    void setMaxExtrapolations(int max_extrap) {
        if (max_extrap < 2 || max_extrap > sequence_length) {
            throw InvalidParameterException(
                "Max extrapolations must be between 2 and " + 
                std::to_string(sequence_length));
        }
        max_extrapolations_ = max_extrap;
    }

private:
    int max_extrapolations_;  ///< Maximum extrapolation order to try
};

} // namespace ode_solver