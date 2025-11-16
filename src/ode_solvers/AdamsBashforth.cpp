/**
 * @file AdamsBashforth.cpp
 * @brief Implementation of Adams-Bashforth multistep methods
 * @author Your Name
 * @version 1.0.0
 */

#include "ode_solver/ODESolvers.hpp"
#include <stdexcept>
#include <iostream>

namespace ode_solver {

/**
 * @brief Adams-Bashforth explicit multistep method
 * 
 * Uses information from multiple previous steps for high efficiency.
 * Available in 2nd, 3rd, and 4th order variants.
 */
class AdamsBashforth : public MultistepODESolver {
public:
    /**
     * @brief Constructor
     * @param system ODE system function
     * @param order Order of method (2, 3, or 4)
     * @param h Step size
     * @param tolerance Tolerance
     */
    explicit AdamsBashforth(ODESystem system, 
                           int order = 4,
                           double h = 0.01,
                           double tolerance = 1e-6)
        : MultistepODESolver(std::move(system), h, tolerance)
        , order_(order)
        , bootstrap_solver_(nullptr) {
        
        if (order < 2 || order > 4) {
            throw InvalidParameterException(
                "Adams-Bashforth order must be 2, 3, or 4");
        }
        
        // Create RK4 for bootstrapping
        bootstrap_solver_ = std::make_unique<RungeKutta4>(system_, h);
    }
    
    std::vector<double> step(double t, const std::vector<double>& y) override {
        const size_t n = y.size();
        
        // Check if we have enough history
        if (static_cast<int>(f_history_.size()) < order_) {
            throw std::logic_error(
                "Adams-Bashforth: Insufficient history for stepping. " 
                "Use solve() instead of step() directly.");
        }
        
        std::vector<double> y_next(n);
        
        // Get the most recent derivatives (stored in reverse chronological order)
        // f_history_[0] is most recent, f_history_[order_-1] is oldest
        
        switch (order_) {
            case 2: {
                // AB2: y_{n+1} = y_n + h * (3/2 * f_n - 1/2 * f_{n-1})
                const auto& f_n = f_history_[0];
                const auto& f_n1 = f_history_[1];
                
                for (size_t i = 0; i < n; ++i) {
                    y_next[i] = y[i] + step_size_ * (1.5 * f_n[i] - 0.5 * f_n1[i]);
                }
                break;
            }
            
            case 3: {
                // AB3: y_{n+1} = y_n + h/12 * (23*f_n - 16*f_{n-1} + 5*f_{n-2})
                const auto& f_n = f_history_[0];
                const auto& f_n1 = f_history_[1];
                const auto& f_n2 = f_history_[2];
                
                for (size_t i = 0; i < n; ++i) {
                    y_next[i] = y[i] + (step_size_ / 12.0) * 
                               (23.0*f_n[i] - 16.0*f_n1[i] + 5.0*f_n2[i]);
                }
                break;
            }
            
            case 4: {
                // AB4: y_{n+1} = y_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
                const auto& f_n = f_history_[0];
                const auto& f_n1 = f_history_[1];
                const auto& f_n2 = f_history_[2];
                const auto& f_n3 = f_history_[3];
                
                for (size_t i = 0; i < n; ++i) {
                    y_next[i] = y[i] + (step_size_ / 24.0) * 
                               (55.0*f_n[i] - 59.0*f_n1[i] + 
                                37.0*f_n2[i] - 9.0*f_n3[i]);
                }
                break;
            }
            
            default:
                throw std::logic_error("Invalid Adams-Bashforth order");
        }
        
        // Check for numerical issues
        if (!isFinite(y_next)) {
            throw NumericalInstabilityException(
                "Adams-Bashforth: Non-finite values detected");
        }
        
        return y_next;
    }
    
    ODESolution solve(double t_start, double t_end,
                     const std::vector<double>& y0) override {
        auto timer_start = startTimer();
        resetIterationCount();
        
        ODESolution solution;
        solution.dimension = static_cast<int>(y0.size());
        
        // Validate inputs
        validateSolveInputs(t_start, t_end, y0);
        
        // Clear any existing history
        clearHistory();
        
        if (verbose_) {
            printVerbose("Starting Adams-Bashforth-" + std::to_string(order_) + 
                        " integration");
            printVerbose("Bootstrapping with RK4 for " + std::to_string(order_) + 
                        " steps");
        }
        
        // Bootstrap: use RK4 to generate initial history
        initializeHistory(t_start, y0, step_size_);
        
        // Copy bootstrap solution
        solution.t_values = t_history_;
        solution.y_values = y_history_;
        
        double t = t_history_.back();
        std::vector<double> y = y_history_.back();
        
        if (verbose_) {
            printVerbose("Bootstrap complete, starting Adams-Bashforth steps");
        }
        
        // Main integration loop using Adams-Bashforth
        while (t < t_end) {
            // Adjust last step
            double h = step_size_;
            if (t + h > t_end) {
                h = t_end - t;
            }
            
            // Store old step size
            double old_h = step_size_;
            step_size_ = h;
            
            // Take AB step
            y = step(t, y);
            t += h;
            
            step_size_ = old_h;
            
            // Compute and store new derivative
            auto f_new = system_(t, y);
            
            // Update history
            updateHistory(t, y, f_new);
            
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
            
            if (verbose_ && current_iterations_ % 100 == 0) {
                printVerbose("AB" + std::to_string(order_) + " iteration " + 
                           std::to_string(current_iterations_) + ", t=" + 
                           std::to_string(t));
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
                "Adams-Bashforth-" + std::to_string(order_) + " completed";
        }
        solution.total_steps = current_iterations_;
        solution.avg_step_size = (t_end - t_start) / current_iterations_;
        
        if (verbose_) {
            printVerbose("Adams-Bashforth complete: " + 
                        std::to_string(current_iterations_) + " steps");
        }
        
        return solution;
    }
    
    int getStepsRequired() const override {
        return order_;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Adams-Bashforth-" + std::to_string(order_);
        return info;
    }
    
    std::string getMethodName() const override {
        return "Adams-Bashforth-" + std::to_string(order_);
    }
    
    int getMethodOrder() const override {
        return order_;
    }
    
    int getFunctionEvaluationsPerStep() const override {
        return 1; // Only one new evaluation per step (very efficient!)
    }

protected:
    void initializeHistory(double t_start, 
                          const std::vector<double>& y_start,
                          double h) override {
        if (verbose_) {
            printVerbose("Initializing history using RK4");
        }
        
        // Use RK4 to generate the first 'order_' points
        double t = t_start;
        std::vector<double> y = y_start;
        
        // Store initial point
        auto f = system_(t, y);
        t_history_.push_back(t);
        y_history_.push_back(y);
        f_history_.push_back(f);
        
        // Generate order_-1 more points with RK4
        for (int i = 1; i < order_; ++i) {
            y = bootstrap_solver_->step(t, y);
            t += h;
            f = system_(t, y);
            
            t_history_.push_back(t);
            y_history_.push_back(y);
            f_history_.push_back(f);
        }
        
        // Reverse f_history so that index 0 is most recent
        std::reverse(f_history_.begin(), f_history_.end());
        
        if (verbose_) {
            printVerbose("History initialized with " + 
                        std::to_string(order_) + " points");
        }
    }

private:
    int order_;
    std::unique_ptr<RungeKutta4> bootstrap_solver_;
    
    // Forward declaration of RK4 for bootstrapping
    class RungeKutta4 : public ExplicitODESolver {
    public:
        explicit RungeKutta4(ODESystem system, double h)
            : ExplicitODESolver(std::move(system), h) {}
        
        std::vector<double> step(double t, const std::vector<double>& y) override {
            const size_t n = y.size();
            std::vector<double> y_next(n);
            std::vector<double> y_temp(n);
            
            auto k1 = system_(t, y);
            
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + 0.5 * step_size_ * k1[i];
            auto k2 = system_(t + 0.5 * step_size_, y_temp);
            
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + 0.5 * step_size_ * k2[i];
            auto k3 = system_(t + 0.5 * step_size_, y_temp);
            
            for (size_t i = 0; i < n; ++i)
                y_temp[i] = y[i] + step_size_ * k3[i];
            auto k4 = system_(t + step_size_, y_temp);
            
            for (size_t i = 0; i < n; ++i) {
                y_next[i] = y[i] + (step_size_ / 6.0) * 
                           (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
            }
            
            return y_next;
        }
        
        ODESolution solve(double, double, const std::vector<double>&) override {
            throw std::logic_error("Bootstrap RK4 should not call solve()");
        }
        
        ConvergenceInfo getConvergenceInfo() const override {
            return ConvergenceInfo();
        }
        
        std::string getMethodName() const override { return "RK4 Bootstrap"; }
        int getMethodOrder() const override { return 4; }
        int getFunctionEvaluationsPerStep() const override { return 4; }
    };
};

// ============================================================================
// Adams-Bashforth-Moulton Predictor-Corrector (Optional Enhancement)
// ============================================================================

/**
 * @brief Adams-Bashforth-Moulton predictor-corrector method
 * 
 * Uses Adams-Bashforth as predictor and Adams-Moulton as corrector
 * for improved stability and accuracy.
 */
class AdamsBashforthMoulton : public MultistepODESolver {
public:
    explicit AdamsBashforthMoulton(ODESystem system,
                                  int order = 4,
                                  double h = 0.01,
                                  double tolerance = 1e-6)
        : MultistepODESolver(std::move(system), h, tolerance)
        , order_(order)
        , predictor_(std::make_unique<AdamsBashforth>(system_, order, h, tolerance)) {
        
        if (order < 2 || order > 4) {
            throw InvalidParameterException("ABM order must be 2, 3, or 4");
        }
    }
    
    std::vector<double> step(double t, const std::vector<double>& y) override {
        const size_t n = y.size();
        
        // Predictor: Adams-Bashforth
        auto y_pred = predictor_->step(t, y);
        
        // Evaluate at predicted point
        auto f_pred = system_(t + step_size_, y_pred);
        
        // Corrector: Adams-Moulton
        std::vector<double> y_corr(n);
        
        switch (order_) {
            case 2: {
                // AM2: y_{n+1} = y_n + h/2 * (f_{n+1} + f_n)
                const auto& f_n = f_history_[0];
                for (size_t i = 0; i < n; ++i) {
                    y_corr[i] = y[i] + 0.5 * step_size_ * (f_pred[i] + f_n[i]);
                }
                break;
            }
            
            case 3: {
                // AM3: y_{n+1} = y_n + h/12 * (5*f_{n+1} + 8*f_n - f_{n-1})
                const auto& f_n = f_history_[0];
                const auto& f_n1 = f_history_[1];
                for (size_t i = 0; i < n; ++i) {
                    y_corr[i] = y[i] + (step_size_ / 12.0) * 
                               (5.0*f_pred[i] + 8.0*f_n[i] - f_n1[i]);
                }
                break;
            }
            
            case 4: {
                // AM4: y_{n+1} = y_n + h/24 * (9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
                const auto& f_n = f_history_[0];
                const auto& f_n1 = f_history_[1];
                const auto& f_n2 = f_history_[2];
                for (size_t i = 0; i < n; ++i) {
                    y_corr[i] = y[i] + (step_size_ / 24.0) * 
                               (9.0*f_pred[i] + 19.0*f_n[i] - 
                                5.0*f_n1[i] + f_n2[i]);
                }
                break;
            }
            
            default:
                throw std::logic_error("Invalid ABM order");
        }
        
        if (!isFinite(y_corr)) {
            throw NumericalInstabilityException("ABM: Non-finite values");
        }
        
        return y_corr;
    }
    
    ODESolution solve(double t_start, double t_end,
                     const std::vector<double>& y0) override {
        // Implementation similar to AdamsBashforth but using predictor-corrector
        auto timer_start = startTimer();
        resetIterationCount();
        
        ODESolution solution;
        solution.dimension = static_cast<int>(y0.size());
        
        validateSolveInputs(t_start, t_end, y0);
        clearHistory();
        
        // Bootstrap
        initializeHistory(t_start, y0, step_size_);
        
        solution.t_values = t_history_;
        solution.y_values = y_history_;
        
        double t = t_history_.back();
        std::vector<double> y = y_history_.back();
        
        // Main loop
        while (t < t_end) {
            double h = (t + step_size_ > t_end) ? (t_end - t) : step_size_;
            double old_h = step_size_;
            step_size_ = h;
            
            y = step(t, y);
            t += h;
            
            step_size_ = old_h;
            
            auto f_new = system_(t, y);
            updateHistory(t, y, f_new);
            
            solution.t_values.push_back(t);
            solution.y_values.push_back(y);
            
            incrementIterations();
            
            if (maxIterationsExceeded()) {
                solution.convergence_info.status = SolverStatus::MAX_ITERATIONS;
                break;
            }
        }
        
        solution.convergence_info.iterations = current_iterations_;
        solution.convergence_info.execution_time_ms = getElapsedTime(timer_start);
        solution.convergence_info.status = SolverStatus::SUCCESS;
        solution.total_steps = current_iterations_;
        solution.avg_step_size = (t_end - t_start) / current_iterations_;
        
        return solution;
    }
    
    int getStepsRequired() const override { return order_; }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        info.message = "ABM-" + std::to_string(order_);
        return info;
    }
    
    std::string getMethodName() const override {
        return "Adams-Bashforth-Moulton-" + std::to_string(order_);
    }
    
    int getMethodOrder() const override { return order_; }
    int getFunctionEvaluationsPerStep() const override { return 2; }

protected:
    void initializeHistory(double t_start, const std::vector<double>& y_start,
                          double h) override {
        predictor_->initializeHistory(t_start, y_start, h);
        // Copy history from predictor
        t_history_ = predictor_->t_history_;
        y_history_ = predictor_->y_history_;
        f_history_ = predictor_->f_history_;
    }

private:
    int order_;
    std::unique_ptr<AdamsBashforth> predictor_;
};

} // namespace ode_solver