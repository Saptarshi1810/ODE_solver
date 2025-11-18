/**
 * @file BoundaryValueSolver.cpp
 * @brief Implementation of shooting method for boundary value problems
 * @author Your Name
 * @version 1.0.0
 * 
 * Solves two-point boundary value problems (BVPs) of the form:
 *   y'' = f(x, y, y'),  with  y(a) = α,  y(b) = β
 * 
 * Uses shooting method: converts BVP to initial value problems.
 */

#include "ode_solver/ODESolvers.hpp"
#include "ode_solver/RootFinders.hpp"
#include <cmath>
#include <functional>

namespace ode_solver {

/**
 * @brief Boundary condition specification
 */
struct BoundaryCondition {
    enum class Type {
        DIRICHLET,   ///< y(x) = value
        NEUMANN,     ///< y'(x) = value
        ROBIN        ///< a*y(x) + b*y'(x) = value
    };
    
    Type type;
    double location;     ///< x-coordinate of boundary
    double value;        ///< Boundary value
    double robin_a = 1.0;  ///< Robin coefficient for y
    double robin_b = 0.0;  ///< Robin coefficient for y'
    
    /// Constructor for Dirichlet BC: y(location) = value
    static BoundaryCondition dirichlet(double loc, double val) {
        return {Type::DIRICHLET, loc, val, 1.0, 0.0};
    }
    
    /// Constructor for Neumann BC: y'(location) = value
    static BoundaryCondition neumann(double loc, double val) {
        return {Type::NEUMANN, loc, val, 0.0, 1.0};
    }
    
    /// Constructor for Robin BC: a*y(location) + b*y'(location) = value
    static BoundaryCondition robin(double loc, double val, double a, double b) {
        return {Type::ROBIN, loc, val, a, b};
    }
};

/**
 * @brief Shooting method for two-point boundary value problems
 * 
 * Converts BVP to IVP by guessing initial conditions and adjusting
 * until boundary conditions are satisfied.
 */
class ShootingMethod : public ODESolver {
public:
    /**
     * @brief Constructor
     * @param system Second-order ODE: y'' = f(x, y, y')
     * @param left_bc Left boundary condition
     * @param right_bc Right boundary condition
     * @param h Step size for IVP solver
     * @param tolerance Tolerance for BC satisfaction
     */
    explicit ShootingMethod(
        std::function<double(double, double, double)> second_order_ode,
        BoundaryCondition left_bc,
        BoundaryCondition right_bc,
        double h = 0.01,
        double tolerance = 1e-6)
        : ODESolver(convertToFirstOrderSystem(second_order_ode), h, tolerance)
        , second_order_ode_(std::move(second_order_ode))
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , ivp_solver_(nullptr)
        , initial_guess_slope_(0.0) {
        
        // Validate boundary conditions
        if (left_bc_.location >= right_bc_.location) {
            throw InvalidParameterException(
                "Left boundary must be < right boundary");
        }
        
        // Create IVP solver (RK4)
        ivp_solver_ = std::make_unique<RungeKutta4>(system_, h);
        ivp_solver_->setVerbose(false);  // Suppress output from IVP solver
    }
    
    /**
     * @brief Solve the boundary value problem
     * @param t_start Left boundary point (ignored, uses left_bc_.location)
     * @param t_end Right boundary point (ignored, uses right_bc_.location)
     * @param y0 Initial guess (ignored, computed from BCs)
     * @return Solution satisfying both boundary conditions
     */
    ODESolution solve(double t_start, double t_end,
                     const std::vector<double>& y0) override {
        (void)t_start;  // Use BC locations instead
        (void)t_end;
        (void)y0;
        
        return solveBVP();
    }
    
    /**
     * @brief Main BVP solving method
     * @return Solution satisfying boundary conditions
     */
    ODESolution solveBVP() {
        auto timer_start = startTimer();
        resetIterationCount();
        
        if (verbose_) {
            printVerbose("Starting Shooting Method for BVP");
            printVerbose("Domain: [" + std::to_string(left_bc_.location) + 
                        ", " + std::to_string(right_bc_.location) + "]");
        }
        
        // For Dirichlet-Dirichlet BVP: y(a) = α, y(b) = β
        if (left_bc_.type == BoundaryCondition::Type::DIRICHLET &&
            right_bc_.type == BoundaryCondition::Type::DIRICHLET) {
            return solveDirichletDirichlet();
        }
        
        // For other BC types, implement similar strategies
        throw std::runtime_error(
            "Shooting method currently only supports Dirichlet-Dirichlet BVPs");
    }
    
    /**
     * @brief Solve Dirichlet-Dirichlet BVP: y(a) = α, y(b) = β
     */
    ODESolution solveDirichletDirichlet() {
        double a = left_bc_.location;
        double b = right_bc_.location;
        double alpha = left_bc_.value;
        double beta = right_bc_.value;
        
        if (verbose_) {
            printVerbose("Solving Dirichlet-Dirichlet BVP:");
            printVerbose("  y(" + std::to_string(a) + ") = " + std::to_string(alpha));
            printVerbose("  y(" + std::to_string(b) + ") = " + std::to_string(beta));
        }
        
        // Define shooting function: F(s) = y(b; s) - β
        // where s is the initial slope y'(a) = s
        auto shooting_function = [this, a, b, alpha, beta](double slope) {
            // Solve IVP with initial conditions y(a) = α, y'(a) = slope
            std::vector<double> y0 = {alpha, slope};
            
            try {
                auto solution = ivp_solver_->solve(a, b, y0);
                
                // Return difference: y(b) - β
                double y_at_b = solution.y_values.back()[0];
                return y_at_b - beta;
                
            } catch (const std::exception& e) {
                // If IVP fails, return large error
                return 1e10;
            }
        };
        
        // Find the correct initial slope using root finding
        // Initial guess: linear interpolation
        double guess_slope = (beta - alpha) / (b - a);
        initial_guess_slope_ = guess_slope;
        
        if (verbose_) {
            printVerbose("Initial slope guess: " + std::to_string(guess_slope));
        }
        
        // Use Secant method to find correct slope
        SecantMethod root_finder(shooting_function, guess_slope, 
                                guess_slope * 1.1, tolerance_);
        root_finder.setVerbose(verbose_);
        root_finder.setMaxIterations(50);
        
        double correct_slope = root_finder.solve();
        
        if (verbose_) {
            printVerbose("Converged to slope: " + std::to_string(correct_slope));
            auto conv_info = root_finder.getConvergenceInfo();
            printVerbose("Root finder iterations: " + 
                        std::to_string(conv_info.iterations));
        }
        
        // Solve IVP one final time with correct initial slope
        std::vector<double> y0 = {alpha, correct_slope};
        auto solution = ivp_solver_->solve(a, b, y0);
        
        // Verify boundary condition at right endpoint
        double y_final = solution.y_values.back()[0];
        double bc_error = std::abs(y_final - beta);
        
        if (verbose_) {
            printVerbose("Final boundary condition error: " + 
                        std::to_string(bc_error));
        }
        
        if (bc_error > tolerance_ * 10) {
            throw ConvergenceException(
                "Shooting method: Boundary condition not satisfied. Error = " + 
                std::to_string(bc_error));
        }
        
        // Fill convergence info
        solution.convergence_info.execution_time_ms = getElapsedTime(timer_start);
        solution.convergence_info.final_error = bc_error;
        solution.convergence_info.status = SolverStatus::SUCCESS;
        solution.convergence_info.message = 
            "Shooting method converged. BC error = " + std::to_string(bc_error);
        
        return solution;
    }
    
    std::vector<double> step(double t, const std::vector<double>& y) override {
        // For BVP, single steps don't make sense
        throw std::logic_error(
            "ShootingMethod: Use solve() instead of step()");
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Shooting method for BVP";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Shooting Method (BVP)";
    }
    
    int getMethodOrder() const override {
        return ivp_solver_->getMethodOrder();
    }
    
    bool isStiffnessCapable() const override {
        return false;
    }
    
    int getFunctionEvaluationsPerStep() const override {
        return ivp_solver_->getFunctionEvaluationsPerStep();
    }
    
    /**
     * @brief Get initial slope used in final solution
     * @return y'(a) value
     */
    double getInitialSlope() const {
        return initial_guess_slope_;
    }

private:
    std::function<double(double, double, double)> second_order_ode_;
    BoundaryCondition left_bc_;
    BoundaryCondition right_bc_;
    std::unique_ptr<RungeKutta4> ivp_solver_;
    double initial_guess_slope_;
    
    /**
     * @brief Convert second-order ODE to first-order system
     * @param f Second-order ODE: y'' = f(x, y, y')
     * @return First-order system: [y', y''] = [v, f(x,y,v)]
     */
    static ODESystem convertToFirstOrderSystem(
        std::function<double(double, double, double)> f) {
        
        return [f](double x, const std::vector<double>& state) {
            // state[0] = y
            // state[1] = y' = v
            double y = state[0];
            double v = state[1];
            
            // Return [y', y''] = [v, f(x,y,v)]
            return std::vector<double>{v, f(x, y, v)};
        };
    }
    
    // Forward declaration of Secant method for root finding
    class SecantMethod {
    public:
        explicit SecantMethod(Function f, double x0, double x1, double tol)
            : f_(std::move(f)), x0_(x0), x1_(x1), tolerance_(tol), 
              max_iter_(100), verbose_(false) {}
        
        void setVerbose(bool v) { verbose_ = v; }
        void setMaxIterations(int m) { max_iter_ = m; }
        
        double solve() {
            double x_prev = x0_;
            double x_curr = x1_;
            double f_prev = f_(x_prev);
            double f_curr = f_(x_curr);
            
            for (int iter = 0; iter < max_iter_; ++iter) {
                if (std::abs(f_curr) < tolerance_) {
                    return x_curr;
                }
                
                // Secant formula
                if (std::abs(f_curr - f_prev) < 1e-15) {
                    throw std::runtime_error("Secant: Division by zero");
                }
                
                double x_next = x_curr - f_curr * (x_curr - x_prev) / 
                                        (f_curr - f_prev);
                
                if (verbose_ && iter % 5 == 0) {
                    std::cout << "  Secant iteration " << iter 
                             << ": x = " << x_next 
                             << ", f(x) = " << f_curr << std::endl;
                }
                
                x_prev = x_curr;
                f_prev = f_curr;
                x_curr = x_next;
                f_curr = f_(x_curr);
            }
            
            throw std::runtime_error(
                "Secant method failed to converge in " + 
                std::to_string(max_iter_) + " iterations");
        }
        
        ConvergenceInfo getConvergenceInfo() const {
            ConvergenceInfo info;
            info.status = SolverStatus::SUCCESS;
            return info;
        }
        
    private:
        Function f_;
        double x0_, x1_;
        double tolerance_;
        int max_iter_;
        bool verbose_;
    };
    
    // Minimal RK4 implementation for IVP solving
    class RungeKutta4 : public ExplicitODESolver {
    public:
        explicit RungeKutta4(ODESystem system, double h)
            : ExplicitODESolver(std::move(system), h) {}
        
        std::vector<double> step(double t, const std::vector<double>& y) override {
            const size_t n = y.size();
            std::vector<double> y_next(n), y_temp(n);
            
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
        
        ODESolution solve(double t_start, double t_end,
                         const std::vector<double>& y0) override {
            ODESolution solution;
            solution.dimension = y0.size();
            
            double t = t_start;
            std::vector<double> y = y0;
            
            solution.t_values.push_back(t);
            solution.y_values.push_back(y);
            
            while (t < t_end) {
                double h = std::min(step_size_, t_end - t);
                double old_h = step_size_;
                step_size_ = h;
                
                y = step(t, y);
                t += h;
                
                step_size_ = old_h;
                
                solution.t_values.push_back(t);
                solution.y_values.push_back(y);
            }
            
            solution.convergence_info.status = SolverStatus::SUCCESS;
            return solution;
        }
        
        ConvergenceInfo getConvergenceInfo() const override {
            return ConvergenceInfo();
        }
        
        std::string getMethodName() const override { return "RK4"; }
        int getMethodOrder() const override { return 4; }
        int getFunctionEvaluationsPerStep() const override { return 4; }
    };
};

} // namespace ode_solver