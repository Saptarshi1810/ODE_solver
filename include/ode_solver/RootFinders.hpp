

#ifndef ODE_SOLVER_ROOT_FINDERS_HPP
#define ODE_SOLVER_ROOT_FINDERS_HPP

#include "EquationSolver.hpp"
#include <complex>
#include <limits>

namespace ode_solver {

// Enumerations

enum class RootFinderCategory {
    BRACKETING,     ///< Requires interval [a,b] where f(a)·f(b) < 0
    OPEN,          ///< Requires initial guess x₀
    POLYNOMIAL,    ///< Specialized for polynomials
    FIXED_POINT    ///< Fixed-point iteration methods
};

enum class ConvergenceRate {
    LINEAR,        ///< O(h) - Bisection, Regula-Falsi
    SUPERLINEAR,   ///< Between linear and quadratic - Secant, Muller
    QUADRATIC,     ///< O(h²) - Newton-Raphson, Steffensen
    CUBIC          ///< O(h³) - Chebyshev, Laguerre
};

// Configuration Structures


struct BracketConfig {
    double a;                    ///< Left bracket endpoint
    double b;                    ///< Right bracket endpoint
    bool verify_bracket = true;  ///< Verify f(a)·f(b) < 0
    double expansion_factor = 2.0; ///< Factor for bracket expansion
    int max_expansions = 10;     ///< Max attempts to find valid bracket
    

    bool isValid(Function f) const {
        if (!verify_bracket) return true;
        return f(a) * f(b) < 0;
    }
};


struct PolynomialConfig {
    std::vector<double> coefficients; ///< Coefficients [a₀, a₁, ..., aₙ]
    bool find_all_roots = true;       ///< Find all roots vs single root
    bool include_complex = true;      ///< Include complex roots
    double deflation_tolerance = 1e-10; ///< Tolerance for deflation
};

// Abstract Base Class: RootFinder

class RootFinder : public EquationSolver {
public:

    explicit RootFinder(Function f,
                       double tolerance = 1e-6,
                       int max_iter = 1000)
        : EquationSolver(tolerance, max_iter)
        , function_(std::move(f))
        , last_root_(std::numeric_limits<double>::quiet_NaN())
        , last_error_(std::numeric_limits<double>::quiet_NaN())
        , function_evaluations_(0) {}
    
    virtual ~RootFinder() = default;
    
    // Disable copy, enable move
    RootFinder(const RootFinder&) = delete;
    RootFinder& operator=(const RootFinder&) = delete;
    RootFinder(RootFinder&&) noexcept = default;
    RootFinder& operator=(RootFinder&&) noexcept = default;
    

    // Pure Virtual Methods

    

    virtual double solve() = 0;
    
    virtual RootFinderCategory getCategory() const = 0;
    

    virtual ConvergenceRate getConvergenceRate() const = 0;
    

    virtual bool requiresDerivative() const = 0;
    

    virtual bool supportsComplexRoots() const = 0;

    virtual double iterate() = 0;
    
    // Query Methods

    double getLastRoot() const { return last_root_; }
    
    double getLastError() const { return last_error_; }

    int getFunctionEvaluations() const { return function_evaluations_; }
    
    const Function& getFunction() const { return function_; }
    
    double getEfficiency() const {
        if (function_evaluations_ == 0) return 0.0;
        auto info = getConvergenceInfo();
        return -std::log10(info.final_error) / function_evaluations_;
    }
    
    // EquationSolver Interface Implementation
    double getConvergenceOrder() const override {
        switch (getConvergenceRate()) {
            case ConvergenceRate::LINEAR:      return 1.0;
            case ConvergenceRate::SUPERLINEAR: return 1.618; // Golden ratio approximation
            case ConvergenceRate::QUADRATIC:   return 2.0;
            case ConvergenceRate::CUBIC:       return 3.0;
            default:                           return 1.0;
        }
    }

protected:
    Function function_;              ///< Function to find root of
    double last_root_;               ///< Last computed root
    double last_error_;              ///< Last error estimate
    mutable int function_evaluations_; ///< Count of function evaluations
    
    double evaluateFunction(double x) const {
        ++function_evaluations_;
        return function_(x);
    }
    

    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
    

    bool hasConverged(double x, double x_prev) const {
        double f_val = std::abs(evaluateFunction(x));
        double change = std::abs(x - x_prev);
        return (f_val < tolerance_) || (change < tolerance_);
    }
    
    bool hasConvergedFunction(double x) const {
        return std::abs(evaluateFunction(x)) < tolerance_;
    }

    void updateRoot(double root, double error) {
        last_root_ = root;
        last_error_ = error;
        recordError(error);
    }
};

// Abstract: Bracketing Methods

class BracketingMethod : public RootFinder {
public:
    explicit BracketingMethod(Function f,
                             const BracketConfig& config,
                             double tolerance = 1e-6,
                             int max_iter = 1000)
        : RootFinder(std::move(f), tolerance, max_iter)
        , a_(config.a)
        , b_(config.b)
        , f_a_(0.0)
        , f_b_(0.0) {
        
        if (a_ >= b_) {
            throw InvalidParameterException("Invalid bracket: a must be < b");
        }
        
        // Evaluate and verify bracket
        f_a_ = evaluateFunction(a_);
        f_b_ = evaluateFunction(b_);
        
        if (config.verify_bracket && f_a_ * f_b_ >= 0) {
            throw InvalidParameterException(
                "Invalid bracket: f(a) and f(b) must have opposite signs");
        }
    }
    
    RootFinderCategory getCategory() const override {
        return RootFinderCategory::BRACKETING;
    }
    
    bool requiresDerivative() const override {
        return false;
    }
    
    bool supportsComplexRoots() const override {
        return false; // Real brackets only
    }
    
    std::pair<double, double> getBracket() const {
        return {a_, b_};
    }

    double getBracketWidth() const {
        return std::abs(b_ - a_);
    }

protected:
    double a_, b_;     ///< Bracket endpoints
    double f_a_, f_b_; ///< Function values at endpoints
    
    /**
     * @brief Update bracket to [a, c] or [c, b]
     * @param c New midpoint
     */
    void updateBracket(double c) {
        double f_c = evaluateFunction(c);
        
        if (f_a_ * f_c < 0) {
            b_ = c;
            f_b_ = f_c;
        } else {
            a_ = c;
            f_a_ = f_c;
        }
    }
};


// Abstract: Open Methods

class OpenMethod : public RootFinder {
public:

    explicit OpenMethod(Function f,
                       double x0,
                       double tolerance = 1e-6,
                       int max_iter = 1000)
        : RootFinder(std::move(f), tolerance, max_iter)
        , x0_(x0)
        , current_x_(x0) {
        
        if (!std::isfinite(x0)) {
            throw InvalidParameterException("Initial guess must be finite");
        }
    }
    
    RootFinderCategory getCategory() const override {
        return RootFinderCategory::OPEN;
    }
    
    bool supportsComplexRoots() const override {
        return false; // Override in methods that support complex
    }

    double getInitialGuess() const { return x0_; }

    double getCurrentEstimate() const { return current_x_; }

    void setCurrentEstimate(double x) { current_x_ = x; }

protected:
    double x0_;         ///< Initial guess
    double current_x_;  ///< Current estimate
};


// Abstract: Derivative-Based Methods

class DerivativeMethod : public OpenMethod {
public:

    explicit DerivativeMethod(Function f,
                             DerivativeFunction df,
                             double x0,
                             double tolerance = 1e-6,
                             int max_iter = 1000)
        : OpenMethod(std::move(f), x0, tolerance, max_iter)
        , derivative_(std::move(df))
        , derivative_evaluations_(0) {}
    
    bool requiresDerivative() const override {
        return true;
    }
  
    const DerivativeFunction& getDerivative() const {
        return derivative_;
    }

    int getDerivativeEvaluations() const {
        return derivative_evaluations_;
    }

protected:
    DerivativeFunction derivative_;
    mutable int derivative_evaluations_;

    double evaluateDerivative(double x) const {
        ++derivative_evaluations_;
        double df = derivative_(x);
        
        if (isTooSmall(df)) {
            throw NumericalInstabilityException(
                "Derivative too small (near-horizontal tangent)");
        }
        
        return df;
    }

    void resetDerivativeEvaluations() {
        derivative_evaluations_ = 0;
    }
};

// Abstract: Higher-Order Derivative Methods

class SecondDerivativeMethod : public DerivativeMethod {
public:

    explicit SecondDerivativeMethod(Function f,
                                   DerivativeFunction df,
                                   SecondDerivativeFunction ddf,
                                   double x0,
                                   double tolerance = 1e-6,
                                   int max_iter = 1000)
        : DerivativeMethod(std::move(f), std::move(df), x0, tolerance, max_iter)
        , second_derivative_(std::move(ddf)) {}
    

    const SecondDerivativeFunction& getSecondDerivative() const {
        return second_derivative_;
    }

protected:
    SecondDerivativeFunction second_derivative_;
    

    double evaluateSecondDerivative(double x) const {
        return second_derivative_(x);
    }
};

// Abstract: Polynomial Methods

class PolynomialMethod : public RootFinder {
public:

    explicit PolynomialMethod(const std::vector<double>& coefficients,
                             double tolerance = 1e-6,
                             int max_iter = 1000)
        : RootFinder(createPolynomialFunction(coefficients), tolerance, max_iter)
        , coefficients_(coefficients)
        , degree_(static_cast<int>(coefficients.size()) - 1) {
        
        if (coefficients.empty()) {
            throw InvalidParameterException("Polynomial must have at least one coefficient");
        }
    }
    
    RootFinderCategory getCategory() const override {
        return RootFinderCategory::POLYNOMIAL;
    }
    
    bool requiresDerivative() const override {
        return false; // Computed from coefficients
    }
    

    int getDegree() const { return degree_; }
    

    const std::vector<double>& getCoefficients() const {
        return coefficients_;
    }
    

    double evaluatePolynomial(double x) const {
        return evaluatePolynomialHorner(x, coefficients_);
    }

    double evaluatePolynomialDerivative(double x) const {
        auto deriv_coeffs = computeDerivativeCoefficients(coefficients_);
        return evaluatePolynomialHorner(x, deriv_coeffs);
    }

protected:
    std::vector<double> coefficients_;
    int degree_;

    static Function createPolynomialFunction(const std::vector<double>& coeffs) {
        return [coeffs](double x) {
            return evaluatePolynomialHorner(x, coeffs);
        };
    }
    

    static double evaluatePolynomialHorner(double x, const std::vector<double>& coeffs) {
        if (coeffs.empty()) return 0.0;
        
        double result = coeffs.back();
        for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
            result = result * x + coeffs[i];
        }
        return result;
    }
    
   
    static std::vector<double> computeDerivativeCoefficients(
        const std::vector<double>& coeffs) {
        
        if (coeffs.size() <= 1) return {0.0};
        
        std::vector<double> deriv(coeffs.size() - 1);
        for (size_t i = 1; i < coeffs.size(); ++i) {
            deriv[i-1] = i * coeffs[i];
        }
        return deriv;
    }
    

    std::vector<double> deflate(double root) const {
        // Synthetic division
        std::vector<double> deflated(coefficients_.size() - 1);
        deflated[deflated.size() - 1] = coefficients_.back();
        
        for (int i = static_cast<int>(deflated.size()) - 2; i >= 0; --i) {
            deflated[i] = coefficients_[i+1] + root * deflated[i+1];
        }
        
        return deflated;
    }
};


// Forward Declarations of Concrete Root-Finding Classes


// Bracketing Methods
class BisectionMethod;        // Reliable, linear convergence
class RegulaFalsiMethod;      // False position, ~1.6 order
class BrentMethod;            // Hybrid, best bracketing method
class RidderMethod;           // Exponential convergence, bracketing

// Open Methods (Derivative-Free)
class SecantMethod;           // ~1.618 order (golden ratio)
class SteffensenMethod;       // Quadratic without derivatives
class MullerMethod;           // Parabolic, finds complex roots

// Derivative-Based Methods
class NewtonRaphsonMethod;    // Classic, quadratic convergence
class ChebyshevMethod;        // Cubic convergence, needs f''

// Polynomial Specialized
class LaguerreMethod;         // For polynomial roots, cubic convergence

// Fixed-Point Methods
class FixedPointIteration;    // Simple iteration x = g(x)
class MultipointIterationMethod; // Multi-point variant

// Utility Functions


std::pair<double, double> findBracket(Function f,
                                      double x0 = 0.0,
                                      double expansion_factor = 1.6,
                                      int max_iterations = 50);


std::vector<double> findAllPolynomialRoots(
    const std::vector<double>& coefficients,
    double a, double b,
    double tolerance = 1e-6);

int estimateRootMultiplicity(Function f, double root, double tolerance = 1e-6);

} // namespace ode_solver

#endif // ODE_SOLVER_ROOT_FINDERS_HPP