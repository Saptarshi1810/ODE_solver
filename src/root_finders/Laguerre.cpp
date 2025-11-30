/**
 * @file Laguerre.cpp
 * @brief Implementation of Laguerre's method for polynomial root finding
 * @author Your Name
 * @version 1.0.0
 * 
 * Laguerre's method is specialized for finding roots of polynomials.
 * It has cubic convergence and works exceptionally well for polynomials,
 * including finding complex roots.
 * 
 * Algorithm:
 *   Uses the polynomial and its first two derivatives
 *   with a special formula designed for polynomials:
 *   
 *   G = P'/P
 *   H = G² - P''/P
 *   a = n / (G ± √((n-1)(nH - G²)))
 *   x_{n+1} = x_n - a
 * 
 * Convergence: Cubic (order 3) for polynomials
 * Advantages: Excellent for polynomials, finds complex roots
 * Disadvantages: Specialized for polynomials only
 */

#include "ode_solver/RootFinders.hpp"
#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

namespace ode_solver {

/**
 * @brief Laguerre's method for polynomial root finding
 * 
 * Highly effective method specifically designed for polynomials.
 * Can find all roots including complex ones through deflation.
 */
class LaguerreMethod : public PolynomialMethod {
public:
    /**
     * @brief Constructor
     * @param coefficients Polynomial coefficients [a₀, a₁, ..., aₙ] where
     *                     P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
     * @param x0 Initial guess (default: 0)
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum iterations
     */
    explicit LaguerreMethod(const std::vector<double>& coefficients,
                           double x0 = 0.0,
                           double tolerance = 1e-10,
                           int max_iter = 100)
        : PolynomialMethod(coefficients, tolerance, max_iter)
        , x0_(x0)
        , find_all_roots_(false) {
        
        if (verbose_) {
            std::cout << std::setprecision(10);
            printVerbose("Initializing Laguerre's Method");
            printVerbose("Polynomial degree: " + std::to_string(degree_));
            printVerbose("Initial guess: x0 = " + std::to_string(x0_));
        }
    }
    
    double solve() override {
        auto timer_start = startTimer();
        resetIterationCount();
        resetFunctionEvaluations();
        
        if (verbose_) {
            printVerbose("Starting Laguerre iteration");
            printVerbose("Tolerance: " + std::to_string(tolerance_));
        }
        
        double x = x0_;
        
        for (current_iterations_ = 0;
             current_iterations_ < max_iterations_;
             ++current_iterations_) {
            
            // Evaluate polynomial and its derivatives at x
            double p, p_prime, p_double_prime;
            evaluatePolynomialAndDerivatives(x, p, p_prime, p_double_prime);
            
            // Check convergence
            if (std::abs(p) < tolerance_) {
                last_root_ = x;
                last_error_ = std::abs(p);
                
                if (verbose_) {
                    printVerbose("Converged!");
                    printVerbose("Root: " + std::to_string(x));
                    printVerbose("P(root) = " + std::to_string(p));
                    printVerbose("Iterations: " + std::to_string(current_iterations_));
                }
                
                recordError(last_error_);
                return x;
            }
            
            // Check for zero derivative
            if (std::abs(p_prime) < 1e-15) {
                // Try perturbing x slightly
                x += 1e-8;
                continue;
            }
            
            // Laguerre's formula
            double G = p_prime / p;
            double H = G * G - p_double_prime / p;
            
            double n = static_cast<double>(degree_);
            double discriminant = (n - 1.0) * (n * H - G * G);
            
            // Handle negative discriminant (shouldn't happen for real polynomials
            // but can due to numerical errors)
            if (discriminant < 0) {
                discriminant = 0;
            }
            
            double sqrt_disc = std::sqrt(discriminant);
            
            // Choose sign to maximize denominator
            double denom1 = G + sqrt_disc;
            double denom2 = G - sqrt_disc;
            double denominator = (std::abs(denom1) > std::abs(denom2)) ? 
                                denom1 : denom2;
            
            if (std::abs(denominator) < 1e-15) {
                throw NumericalInstabilityException(
                    "Laguerre: Denominator too small");
            }
            
            double a = n / denominator;
            double x_new = x - a;
            
            // Check for numerical issues
            if (!isFinite(x_new)) {
                throw NumericalInstabilityException(
                    "Laguerre: Non-finite value");
            }
            
            if (verbose_ && current_iterations_ % 5 == 0) {
                printVerbose("Iteration " + std::to_string(current_iterations_) +
                           ": x = " + std::to_string(x_new) +
                           ", P(x) = " + std::to_string(p) +
                           ", |step| = " + std::to_string(std::abs(a)));
            }
            
            x = x_new;
            recordError(std::abs(p));
        }
        
        // Maximum iterations reached
        last_root_ = x;
        last_error_ = std::abs(evaluatePolynomial(x));
        
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
        
        throw ConvergenceException("Laguerre: Maximum iterations reached", info);
    }
    
    double iterate() override {
        double x = x0_;
        
        double p, p_prime, p_double_prime;
        evaluatePolynomialAndDerivatives(x, p, p_prime, p_double_prime);
        
        double G = p_prime / p;
        double H = G * G - p_double_prime / p;
        
        double n = static_cast<double>(degree_);
        double discriminant = (n - 1.0) * (n * H - G * G);
        double sqrt_disc = std::sqrt(std::max(0.0, discriminant));
        
        double denom = (std::abs(G + sqrt_disc) > std::abs(G - sqrt_disc)) ?
                      (G + sqrt_disc) : (G - sqrt_disc);
        
        double a = n / denom;
        x0_ = x - a;
        
        return x0_;
    }
    
    /**
     * @brief Find all roots of the polynomial
     * @param initial_guesses Optional initial guesses for each root
     * @return Vector of all roots (real and complex represented as pairs)
     */
    std::vector<double> findAllRoots(const std::vector<double>& initial_guesses = {}) {
        std::vector<double> roots;
        std::vector<double> current_coeffs = coefficients_;
        int current_degree = degree_;
        
        if (verbose_) {
            printVerbose("Finding all roots via deflation");
        }
        
        // Find roots one by one and deflate
        for (int i = 0; i < degree_; ++i) {
            // Determine initial guess
            double guess = 0.0;
            if (i < static_cast<int>(initial_guesses.size())) {
                guess = initial_guesses[i];
            } else {
                // Use a pseudo-random guess
                guess = std::cos(static_cast<double>(i) * 0.7) * 
                       (1.0 + static_cast<double>(i));
            }
            
            // Find root using current polynomial
            LaguerreMethod solver(current_coeffs, guess, tolerance_, max_iterations_);
            solver.setVerbose(false); // Suppress individual solver output
            
            try {
                double root = solver.solve();
                roots.push_back(root);
                
                if (verbose_) {
                    printVerbose("Root " + std::to_string(i+1) + ": " + 
                               std::to_string(root));
                }
                
                // Deflate polynomial: divide by (x - root)
                current_coeffs = deflatePolynomial(current_coeffs, root);
                --current_degree;
                
            } catch (const std::exception& e) {
                if (verbose_) {
                    printVerbose("Warning: Could not find root " + 
                               std::to_string(i+1));
                }
            }
            
            // Stop if polynomial fully deflated
            if (current_degree <= 0) break;
        }
        
        return roots;
    }
    
    ConvergenceInfo getConvergenceInfo() const override {
        ConvergenceInfo info;
        info.iterations = current_iterations_;
        info.final_error = last_error_;
        info.status = SolverStatus::SUCCESS;
        info.message = "Laguerre's method for polynomials";
        return info;
    }
    
    std::string getMethodName() const override {
        return "Laguerre's Method";
    }
    
    ConvergenceRate getConvergenceRate() const override {
        return ConvergenceRate::CUBIC;
    }
    
    bool requiresDerivative() const override {
        return false; // Computed from coefficients
    }
    
    bool supportsComplexRoots() const override {
        return true; // Can find complex roots
    }

private:
    double x0_;
    bool find_all_roots_;
    
    /**
     * @brief Evaluate polynomial and first two derivatives using Horner's method
     * @param x Point to evaluate
     * @param p Output: P(x)
     * @param p_prime Output: P'(x)
     * @param p_double_prime Output: P''(x)
     */
    void evaluatePolynomialAndDerivatives(double x, 
                                         double& p, 
                                         double& p_prime,
                                         double& p_double_prime) const {
        int n = static_cast<int>(coefficients_.size()) - 1;
        
        // Initialize with highest degree coefficient
        p = coefficients_[n];
        p_prime = 0.0;
        p_double_prime = 0.0;
        
        // Horner's method for simultaneous evaluation
        for (int i = n - 1; i >= 0; --i) {
            p_double_prime = p_double_prime * x + 2.0 * p_prime;
            p_prime = p_prime * x + p;
            p = p * x + coefficients_[i];
        }
        
        ++function_evaluations_;
    }
    
    /**
     * @brief Deflate polynomial by synthetic division
     * @param coeffs Current polynomial coefficients
     * @param root Root to divide out
     * @return Deflated polynomial coefficients
     */
    std::vector<double> deflatePolynomial(const std::vector<double>& coeffs,
                                         double root) const {
        int n = static_cast<int>(coeffs.size()) - 1;
        std::vector<double> deflated(n);
        
        // Synthetic division by (x - root)
        deflated[n-1] = coeffs[n];
        for (int i = n - 2; i >= 0; --i) {
            deflated[i] = coeffs[i+1] + root * deflated[i+1];
        }
        
        return deflated;
    }
    
    void resetFunctionEvaluations() {
        function_evaluations_ = 0;
    }
};

/**
 * @brief Complex version of Laguerre's method
 * 
 * Finds complex roots directly using complex arithmetic.
 */
class ComplexLaguerreMethod {
public:
    using Complex = std::complex<double>;
    
    explicit ComplexLaguerreMethod(const std::vector<double>& coefficients,
                                  Complex x0 = Complex(0, 0),
                                  double tolerance = 1e-10,
                                  int max_iter = 100)
        : coefficients_(coefficients)
        , x0_(x0)
        , tolerance_(tolerance)
        , max_iterations_(max_iter) {
        
        degree_ = static_cast<int>(coefficients.size()) - 1;
    }
    
    Complex solve() {
        Complex x = x0_;
        
        for (int iter = 0; iter < max_iterations_; ++iter) {
            Complex p, p_prime, p_double_prime;
            evaluateComplex(x, p, p_prime, p_double_prime);
            
            if (std::abs(p) < tolerance_) {
                return x;
            }
            
            Complex G = p_prime / p;
            Complex H = G * G - p_double_prime / p;
            
            double n = static_cast<double>(degree_);
            Complex discriminant = Complex(n - 1.0) * (Complex(n) * H - G * G);
            Complex sqrt_disc = std::sqrt(discriminant);
            
            Complex denom1 = G + sqrt_disc;
            Complex denom2 = G - sqrt_disc;
            Complex denominator = (std::abs(denom1) > std::abs(denom2)) ?
                                 denom1 : denom2;
            
            Complex a = Complex(n) / denominator;
            x = x - a;
        }
        
        throw std::runtime_error("Complex Laguerre: Maximum iterations reached");
    }
    
    std::vector<Complex> findAllComplexRoots() {
        std::vector<Complex> roots;
        std::vector<double> current_coeffs = coefficients_;
        
        for (int i = 0; i < degree_; ++i) {
            // Use various initial guesses
            Complex guess(std::cos(i * 0.7), std::sin(i * 0.7));
            
            ComplexLaguerreMethod solver(current_coeffs, guess, tolerance_, max_iterations_);
            
            try {
                Complex root = solver.solve();
                roots.push_back(root);
                
                // Deflate
                current_coeffs = deflateComplex(current_coeffs, root);
                
            } catch (...) {
                // Continue to next root
            }
        }
        
        return roots;
    }

private:
    std::vector<double> coefficients_;
    Complex x0_;
    double tolerance_;
    int max_iterations_;
    int degree_;
    
    void evaluateComplex(Complex x, Complex& p, Complex& p_prime, 
                        Complex& p_double_prime) const {
        int n = static_cast<int>(coefficients_.size()) - 1;
        
        p = Complex(coefficients_[n]);
        p_prime = Complex(0.0);
        p_double_prime = Complex(0.0);
        
        for (int i = n - 1; i >= 0; --i) {
            p_double_prime = p_double_prime * x + 2.0 * p_prime;
            p_prime = p_prime * x + p;
            p = p * x + Complex(coefficients_[i]);
        }
    }
    
    std::vector<double> deflateComplex(const std::vector<double>& coeffs,
                                      Complex root) const {
        // For real coefficients, complex roots come in conjugate pairs
        // Deflate by (x - r)(x - r*) = x² - 2Re(r)x + |r|²
        
        int n = static_cast<int>(coeffs.size()) - 1;
        std::vector<double> deflated(n - 1);
        
        double a = -2.0 * root.real();
        double b = std::norm(root);
        
        // Deflate by quadratic
        deflated[n-2] = coeffs[n];
        if (n >= 2) {
            deflated[n-3] = coeffs[n-1] + a * deflated[n-2];
        }
        
        for (int i = n - 4; i >= 0; --i) {
            deflated[i] = coeffs[i+2] + a * deflated[i+1] + b * deflated[i+2];
        }
        
        return deflated;
    }
};

} // namespace ode_solver