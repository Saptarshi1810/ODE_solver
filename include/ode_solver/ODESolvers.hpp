#ifndef ODE_SOLVER_ODE_SOLVERS_HPP
#define ODE_SOLVER_ODE_SOLVERS_HPP

#include "EquationSolver.hpp"
#include <algorithm>
#include <deque>
#include <iostream>

namespace ode_solver {


// Configuration Structures

struct AdaptiveStepConfig {
    double min_step = 1e-6;        ///< Minimum allowed step size
    double max_step = 1.0;         ///< Maximum allowed step size
    double safety_factor = 0.9;    ///< Safety factor for step adjustment (0 < safety < 1)
    double error_tolerance = 1e-6; ///< Error tolerance for adaptation
    int max_step_rejects = 10;     ///< Maximum consecutive step rejections before failure
    
    /// Validate configuration parameters
    bool isValid() const {
        return min_step > 0 && max_step > min_step && 
               safety_factor > 0 && safety_factor < 1 &&
               error_tolerance > 0 && max_step_rejects > 0;
    }
};


struct EventConfig {
    EventFunction event_function;     ///< Function g(t,y) whose zeros to detect
    bool stop_at_event = true;        ///< Stop integration when event occurs
    double event_tolerance = 1e-6;    ///< Tolerance for event location
    int max_event_iterations = 20;    ///< Max iterations for bisection refinement
};


struct DenseOutputConfig {
    bool enabled = false;             ///< Enable dense output
    double output_step = 0.1;         ///< Step size for output points
    std::string interpolation_method = "cubic"; ///< Interpolation method
};

// Abstract Base Class: ODESolver


class ODESolver : public EquationSolver {
public:
    explicit ODESolver(ODESystem system,
                      double h = 0.01,
                      double tolerance = 1e-6,
                      int max_iter = 1000000)
        : EquationSolver(tolerance, max_iter)
        , system_(std::move(system))
        , step_size_(h)
        , adaptive_step_enabled_(false)
        , dense_output_enabled_(false)
        , stiffness_detection_enabled_(false) {
        
        if (h <= 0.0) {
            throw InvalidParameterException("Step size must be positive");
        }
    }
    
    virtual ~ODESolver() = default;
    
    // Disable copy, enable move
    ODESolver(const ODESolver&) = delete;
    ODESolver& operator=(const ODESolver&) = delete;
    ODESolver(ODESolver&&) noexcept = default;
    ODESolver& operator=(ODESolver&&) noexcept = default;
    

    //Virtual Methods
 

    virtual ODESolution solve(double t_start, 
                             double t_end,
                             const std::vector<double>& y0) = 0;

    virtual std::vector<double> step(double t, 
                                    const std::vector<double>& y) = 0;
    
 
    virtual int getMethodOrder() const = 0;
    

    virtual bool isStiffnessCapable() const = 0;

    virtual int getFunctionEvaluationsPerStep() const = 0;

    // Configuration Methods
 
    void setStepSize(double h) {
        if (h <= 0.0) {
            throw InvalidParameterException("Step size must be positive");
        }
        step_size_ = h;
    }
    

    void enableAdaptiveStep(bool enable) {
        adaptive_step_enabled_ = enable;
    }

    void setAdaptiveConfig(const AdaptiveStepConfig& config) {
        if (!config.isValid()) {
            throw InvalidParameterException("Invalid adaptive step configuration");
        }
        adaptive_config_ = config;
    }
    
 
    void enableDenseOutput(bool enable) {
        dense_output_enabled_ = enable;
    }

    void setDenseOutputConfig(const DenseOutputConfig& config) {
        dense_output_config_ = config;
        dense_output_enabled_ = config.enabled;
    }
    

    void addEventDetection(const EventConfig& config) {
        event_config_ = config;
    }
    

    void clearEventDetection() {
        event_config_.reset();
    }

    void enableStiffnessDetection(bool enable) {
        stiffness_detection_enabled_ = enable;
    }

    void setJacobian(JacobianFunction jacobian) {
        jacobian_ = std::move(jacobian);
    }

    // Query Methods
 

    double getStepSize() const { return step_size_; }

    bool isAdaptiveStepEnabled() const { return adaptive_step_enabled_; }
    

    bool isDenseOutputEnabled() const { return dense_output_enabled_; }
    

    bool isStiffnessDetectionEnabled() const { return stiffness_detection_enabled_; }

    const AdaptiveStepConfig& getAdaptiveConfig() const { return adaptive_config_; }

    // EquationSolver Interface Implementation

    double getConvergenceOrder() const override {
        return static_cast<double>(getMethodOrder());
    }

protected:

    // Protected Members
  
    ODESystem system_;                           ///< ODE system function f(t,y)
    std::optional<JacobianFunction> jacobian_;   ///< Optional Jacobian ∂f/∂y
    double step_size_;                           ///< Current step size
    bool adaptive_step_enabled_;                 ///< Adaptive stepping flag
    bool dense_output_enabled_;                  ///< Dense output flag
    bool stiffness_detection_enabled_;           ///< Stiffness detection flag
    AdaptiveStepConfig adaptive_config_;         ///< Adaptive step configuration
    DenseOutputConfig dense_output_config_;      ///< Dense output configuration
    std::optional<EventConfig> event_config_;    ///< Event detection configuration
    
    // Protected Helper Methods

    std::optional<double> detectEvent(double t_prev, 
                                     const std::vector<double>& y_prev,
                                     double t_curr,
                                     const std::vector<double>& y_curr) const {
        if (!event_config_) return std::nullopt;
        
        double g_prev = event_config_->event_function(t_prev, y_prev);
        double g_curr = event_config_->event_function(t_curr, y_curr);
        
        // Check for sign change 
        if (g_prev * g_curr < 0) {
            // Bisection to locate event precisely
            double t_low = t_prev, t_high = t_curr;
            std::vector<double> y_low = y_prev, y_high = y_curr;
            
            for (int i = 0; i < event_config_->max_event_iterations; ++i) {
                double t_mid = 0.5 * (t_low + t_high);
                
                // Linear interpolation of state at t_mid
                double alpha = (t_mid - t_low) / (t_high - t_low);
                std::vector<double> y_mid(y_low.size());
                for (size_t j = 0; j < y_low.size(); ++j) {
                    y_mid[j] = y_low[j] + alpha * (y_high[j] - y_low[j]);
                }
                
                double g_mid = event_config_->event_function(t_mid, y_mid);
                
                if (std::abs(g_mid) < event_config_->event_tolerance) {
                    return t_mid;
                }
                
                // Update bracket
                if (g_mid * g_low < 0) {
                    t_high = t_mid;
                    y_high = y_mid;
                } else {
                    t_low = t_mid;
                    y_low = y_mid;
                }
            }
            
            return 0.5 * (t_low + t_high);
        }
        
        return std::nullopt;
    }
    
    double estimateError(const std::vector<double>& y_low,
                        const std::vector<double>& y_high) const {
        double error = 0.0;
        for (size_t i = 0; i < y_low.size(); ++i) {
            error = std::max(error, std::abs(y_high[i] - y_low[i]));
        }
        return error;
    }
    

    double computeOptimalStepSize(double error, double current_h, int order) const {
        if (error < 1e-15) {

            return adaptive_config_.max_step;
        }
        
  
        double exponent = 1.0 / (order + 1.0);
        double ratio = std::pow(adaptive_config_.error_tolerance / error, exponent);
        double h_new = adaptive_config_.safety_factor * current_h * ratio;
        
        // Clamp to min/max bounds
        h_new = std::max(adaptive_config_.min_step, 
                        std::min(adaptive_config_.max_step, h_new));
        
        return h_new;
    }
    

    virtual bool detectStiffness(const ODESolution& solution) const {
    
        return false;
    }
    

    void validateSolveInputs(double t_start, double t_end, 
                           const std::vector<double>& y0) const {
        if (t_end <= t_start) {
            throw InvalidParameterException("t_end must be greater than t_start");
        }
        if (y0.empty()) {
            throw InvalidParameterException("Initial condition cannot be empty");
        }
        if (!isFinite(y0)) {
            throw InvalidParameterException("Initial condition contains non-finite values");
        }
    }
};

// Abstract: Explicit ODE Solver

class ExplicitODESolver : public ODESolver {
public:
    using ODESolver::ODESolver;
    
    bool isStiffnessCapable() const override {
        return false; 
    }
};


// Abstract: Implicit ODE Solver


class ImplicitODESolver : public ODESolver {
public:
    using ODESolver::ODESolver;
    
    bool isStiffnessCapable() const override {
        return true; // Implicit methods handle stiff problems
    }
    
protected:

    virtual std::vector<double> solveImplicitStep(
        double t_n, 
        const std::vector<double>& y_n, 
        double h) = 0;
};


// Abstract: Multistep ODE Solver

class MultistepODESolver : public ODESolver {
public:
    using ODESolver::ODESolver;
    
    virtual int getStepsRequired() const = 0;
    
protected:
    std::deque<double> t_history_;                      ///< Previous time points
    std::deque<std::vector<double>> y_history_;         ///< Previous states
    std::deque<std::vector<double>> f_history_;         ///< Previous derivatives
    
    virtual void initializeHistory(double t_start, 
                                   const std::vector<double>& y_start, 
                                   double h) = 0;
    

    void updateHistory(double t, const std::vector<double>& y, 
                      const std::vector<double>& f) {
        t_history_.push_back(t);
        y_history_.push_back(y);
        f_history_.push_back(f);
        
        int steps_req = getStepsRequired();
        while (static_cast<int>(t_history_.size()) > steps_req) {
            t_history_.pop_front();
            y_history_.pop_front();
            f_history_.pop_front();
        }
    }
    

    void clearHistory() {
        t_history_.clear();
        y_history_.clear();
        f_history_.clear();
    }
};


// Forward Declarations of Concrete Solver Classes



class RungeKutta4;            
class AdaptiveRK45;           
class AdamsBashforth;       

class BulirschStoer;          

class ShootingMethod;        

} // namespace ode_solver

#endif // ODE_SOLVER_ODE_SOLVERS_HPP