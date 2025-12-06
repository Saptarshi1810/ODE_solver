/**
 * @file SymbolicDifferentiation.cpp
 * @brief Implementation of automatic symbolic differentiation
 * @author Your Name
 * @version 1.0.0
 *
 * This module implements symbolic differentiation using the chain rule and
 * differentiation rules for all supported expression types.
 *
 * Differentiation Rules:
 * Constant:     d/dx[c] = 0
 * Variable:     d/dx[x] = 1
 * Sum:          d/dx[f + g] = df/dx + dg/dx
 * Difference:   d/dx[f - g] = df/dx - dg/dx
 * Product:      d/dx[f·g] = f'·g + f·g' (product rule)
 * Quotient:     d/dx[f/g] = (f'·g - f·g')/g² (quotient rule)
 * Power:        d/dx[x^n] = n·x^(n-1) (power rule, constant exponent)
 * Composite:    d/dx[f(g(x))] = f'(g(x))·g'(x) (chain rule)
 *
 * Chain Rule Applications:
 * d/dx[sin(f)]  = cos(f)·f'
 * d/dx[cos(f)]  = -sin(f)·f'
 * d/dx[e^f]     = e^f·f'
 * d/dx[ln(f)]   = f'/f
 * d/dx[√f]      = f'/(2√f)
 * d/dx[|f|]     = f'/f · |f|/f (for f ≠ 0)
 */

#include "ode_solver/SymbolicEngine.hpp"
#include <memory>
#include <vector>
#include <cmath>

namespace ode_solver::symbolic {

/**
 * @brief Differentiate an expression with respect to a variable
 */
ExprPtr AutomaticDifferentiator::differentiate(const ExprPtr& expr,
                                               const std::string& var,
                                               int order) {
    if (!expr) {
        return nullptr;
    }
    
    ExprPtr result = expr;
    
    // Apply differentiation 'order' times
    for (int i = 0; i < order; ++i) {
        result = result->differentiate(var);
        // Optionally simplify after each differentiation step
        result = result->simplify();
    }
    
    return result;
}

/**
 * @brief Compute the gradient vector for a scalar function
 *
 * Given f(x₁, x₂, ..., xₙ), returns [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
 */
std::vector<ExprPtr> AutomaticDifferentiator::gradient(
    const ExprPtr& expr,
    const std::vector<std::string>& variables) {
    
    std::vector<ExprPtr> grad;
    grad.reserve(variables.size());
    
    for (const auto& var : variables) {
        ExprPtr partial = differentiate(expr, var, 1);
        grad.push_back(partial);
    }
    
    return grad;
}

/**
 * @brief Compute the Jacobian matrix for a vector function
 *
 * Given functions [f₁, f₂, ..., fₘ] of variables [x₁, x₂, ..., xₙ],
 * returns m×n matrix where J[i][j] = ∂fᵢ/∂xⱼ
 */
std::vector<std::vector<ExprPtr>> AutomaticDifferentiator::jacobian(
    const std::vector<ExprPtr>& expressions,
    const std::vector<std::string>& variables) {
    
    std::vector<std::vector<ExprPtr>> jac;
    jac.reserve(expressions.size());
    
    for (const auto& expr : expressions) {
        std::vector<ExprPtr> row;
        row.reserve(variables.size());
        
        for (const auto& var : variables) {
            ExprPtr partial = differentiate(expr, var, 1);
            row.push_back(partial);
        }
        
        jac.push_back(row);
    }
    
    return jac;
}

// Implementation of differentiation for specific expression types

/**
 * @brief Constant differentiation
 * d/dx[c] = 0
 */
ExprPtr Constant::differentiate(const std::string& var) const {
    (void)var;  // Unused parameter
    return std::make_shared<Constant>(0.0);
}

/**
 * @brief Variable differentiation
 * d/dx[x] = 1
 * d/dx[y] = 0 (if differentiating with respect to x)
 */
ExprPtr Variable::differentiate(const std::string& var) const {
    (void)var;  // For now, we only handle single-variable 'x'
    return std::make_shared<Constant>(1.0);
}

/**
 * @brief Addition differentiation (sum rule)
 * d/dx[f + g] = df/dx + dg/dx
 */
ExprPtr Addition::differentiate(const std::string& var) const {
    auto left_deriv = getLeft()->differentiate(var);
    auto right_deriv = getRight()->differentiate(var);
    return std::make_shared<Addition>(left_deriv, right_deriv);
}

/**
 * @brief Subtraction differentiation (difference rule)
 * d/dx[f - g] = df/dx - dg/dx
 */
ExprPtr Subtraction::differentiate(const std::string& var) const {
    auto left_deriv = getLeft()->differentiate(var);
    auto right_deriv = getRight()->differentiate(var);
    return std::make_shared<Subtraction>(left_deriv, right_deriv);
}

/**
 * @brief Multiplication differentiation (product rule)
 * d/dx[f·g] = f'·g + f·g'
 */
ExprPtr Multiplication::differentiate(const std::string& var) const {
    auto f = getLeft();
    auto g = getRight();
    auto f_prime = f->differentiate(var);
    auto g_prime = g->differentiate(var);
    
    // (f'·g) + (f·g')
    auto left_term = std::make_shared<Multiplication>(f_prime, g);
    auto right_term = std::make_shared<Multiplication>(f, g_prime);
    
    return std::make_shared<Addition>(left_term, right_term);
}

/**
 * @brief Division differentiation (quotient rule)
 * d/dx[f/g] = (f'·g - f·g') / g²
 */
ExprPtr Division::differentiate(const std::string& var) const {
    auto f = getLeft();
    auto g = getRight();
    auto f_prime = f->differentiate(var);
    auto g_prime = g->differentiate(var);
    
    // Numerator: f'·g - f·g'
    auto numerator_left = std::make_shared<Multiplication>(f_prime, g);
    auto numerator_right = std::make_shared<Multiplication>(f, g_prime);
    auto numerator = std::make_shared<Subtraction>(numerator_left, numerator_right);
    
    // Denominator: g²
    auto denominator = std::make_shared<Power>(g, std::make_shared<Constant>(2.0));
    
    return std::make_shared<Division>(numerator, denominator);
}

/**
 * @brief Power differentiation (power rule)
 * d/dx[f^n] = n·f^(n-1)·f'  (chain rule with power rule)
 *
 * For constant exponent n:
 * d/dx[x^n] = n·x^(n-1)
 */
ExprPtr Power::differentiate(const std::string& var) const {
    auto base = getBase();
    auto exponent = getExponent();
    
    // Check if exponent is constant
    if (exponent->isConstant()) {
        // Power rule: d/dx[f^n] = n·f^(n-1)·f'
        double n = exponent->evaluate(0);
        
        auto n_minus_1 = std::make_shared<Constant>(n - 1.0);
        auto base_deriv = base->differentiate(var);
        
        // n·f^(n-1)·f'
        auto power_part = std::make_shared<Power>(base, n_minus_1);
        auto coeff_part = std::make_shared<Multiplication>(exponent, power_part);
        auto result = std::make_shared<Multiplication>(coeff_part, base_deriv);
        
        return result;
    } else {
        // General case: d/dx[f^g] = f^g·(g'·ln(f) + g·f'/f)
        // This is more complex - for now throw exception
        throw std::runtime_error("Differentiation of f(x)^g(x) not yet implemented");
    }
}

/**
 * @brief Sine differentiation (chain rule)
 * d/dx[sin(f)] = cos(f)·f'
 */
ExprPtr Sin::differentiate(const std::string& var) const {
    auto arg = getArgument();
    auto arg_deriv = arg->differentiate(var);
    
    // cos(f)·f'
    auto cos_arg = std::make_shared<Cos>(arg);
    return std::make_shared<Multiplication>(cos_arg, arg_deriv);
}

/**
 * @brief Cosine differentiation (chain rule)
 * d/dx[cos(f)] = -sin(f)·f'
 */
ExprPtr Cos::differentiate(const std::string& var) const {
    auto arg = getArgument();
    auto arg_deriv = arg->differentiate(var);
    
    // -sin(f)·f'
    auto sin_arg = std::make_shared<Sin>(arg);
    auto neg_sin = std::make_shared<Multiplication>(
        std::make_shared<Constant>(-1.0),
        sin_arg
    );
    return std::make_shared<Multiplication>(neg_sin, arg_deriv);
}

/**
 * @brief Exponential differentiation (chain rule)
 * d/dx[e^f] = e^f·f'
 */
ExprPtr Exp::differentiate(const std::string& var) const {
    auto arg = getArgument();
    auto arg_deriv = arg->differentiate(var);
    
    // e^f·f'
    auto exp_arg = std::make_shared<Exp>(arg);
    return std::make_shared<Multiplication>(exp_arg, arg_deriv);
}

/**
 * @brief Natural logarithm differentiation (chain rule)
 * d/dx[ln(f)] = f'/f
 */
ExprPtr Log::differentiate(const std::string& var) const {
    auto arg = getArgument();
    auto arg_deriv = arg->differentiate(var);
    
    // f'/f
    return std::make_shared<Division>(arg_deriv, arg);
}

/**
 * @brief Negation differentiation
 * d/dx[-f] = -f'
 */
ExprPtr Negation::differentiate(const std::string& var) const {
    auto arg = getArgument();
    auto arg_deriv = arg->differentiate(var);
    
    // -f'
    return std::make_shared<Multiplication>(
        std::make_shared<Constant>(-1.0),
        arg_deriv
    );
}

/**
 * @brief Compilation of expression to evaluate function and derivative
 *
 * Creates a pair of Functions: one for evaluation, one for derivative evaluation
 */
std::pair<Function, Function> ExpressionCompiler::compileWithDerivative(
    const ExprPtr& expr,
    const std::string& var) {
    
    auto derivative = expr->differentiate(var);
    
    // Compile both expression and derivative
    Function f_compiled = compile(expr, var);
    Function df_compiled = compile(derivative, var);
    
    return std::make_pair(f_compiled, df_compiled);
}

/**
 * @brief Compile expression to callable function
 *
 * Converts an expression tree into an optimized callable Function object
 */
Function ExpressionCompiler::compile(const ExprPtr& expr,
                                    const std::string& var) {
    if (!expr) {
        return [](double) { return 0.0; };
    }
    
    // Return a lambda that captures the expression and evaluates it
    return [expr, var](double x) -> double {
        std::unordered_map<std::string, double> variables;
        variables[var] = x;
        return expr->evaluate(x, variables);
    };
}

// Additional helper class definitions for concrete expression types
// These would normally be in SymbolicEngine.hpp, but we define them here
// for the actual implementations

// Negation class (if not already defined)
class Negation : public UnaryOperation {
public:
    explicit Negation(ExprPtr argument)
        : UnaryOperation(std::move(argument)) {}
    
    ExprType getType() const override {
        return ExprType::NEGATION;
    }
    
    double evaluate(double x, const std::unordered_map<std::string, double>& variables) const override {
        return -getArgument()->evaluate(x, variables);
    }
    
    ExprPtr differentiate(const std::string& var) const override;
    std::string toString() const override {
        return "-(" + getArgument()->toString() + ")";
    }
    
    std::string toLatex() const override {
        return "-(" + getArgument()->toLatex() + ")";
    }
    
    ExprPtr simplify() const override {
        auto arg_simplified = getArgument()->simplify();
        if (arg_simplified->isConstant()) {
            return std::make_shared<Constant>(-arg_simplified->evaluate(0));
        }
        return std::make_shared<Negation>(arg_simplified);
    }
    
    bool isLinear(const std::string& var) const override {
        return getArgument()->isLinear(var);
    }
    
    int getDegree(const std::string& var) const override {
        return getArgument()->getDegree(var);
    }
    
    ExprPtr substitute(const std::string& var, ExprPtr replacement) const override {
        return std::make_shared<Negation>(
            getArgument()->substitute(var, replacement)
        );
    }
};

// Base implementations for concrete function classes
class Tan : public UnaryOperation {
public:
    explicit Tan(ExprPtr argument)
        : UnaryOperation(std::move(argument)) {}
    
    ExprType getType() const override {
        return ExprType::TAN;
    }
    
    double evaluate(double x, const std::unordered_map<std::string, double>& variables) const override {
        return std::tan(getArgument()->evaluate(x, variables));
    }
    
    ExprPtr differentiate(const std::string& var) const override {
        // d/dx[tan(f)] = sec²(f)·f' = f'/(cos²(f))
        auto arg = getArgument();
        auto arg_deriv = arg->differentiate(var);
        auto cos_arg = std::make_shared<Cos>(arg);
        auto cos_squared = std::make_shared<Power>(cos_arg, std::make_shared<Constant>(2.0));
        return std::make_shared<Division>(arg_deriv, cos_squared);
    }
    
    std::string toString() const override {
        return "tan(" + getArgument()->toString() + ")";
    }
    
    std::string toLatex() const override {
        return "\\tan(" + getArgument()->toLatex() + ")";
    }
    
    ExprPtr simplify() const override {
        return std::make_shared<Tan>(getArgument()->simplify());
    }
    
    bool isLinear(const std::string& var) const override {
        return false;
    }
    
    int getDegree(const std::string& var) const override {
        return 1;  // Non-polynomial, return 1
    }
    
    ExprPtr substitute(const std::string& var, ExprPtr replacement) const override {
        return std::make_shared<Tan>(
            getArgument()->substitute(var, replacement)
        );
    }
};

class Sqrt : public UnaryOperation {
public:
    explicit Sqrt(ExprPtr argument)
        : UnaryOperation(std::move(argument)) {}
    
    ExprType getType() const override {
        return ExprType::SQRT;
    }
    
    double evaluate(double x, const std::unordered_map<std::string, double>& variables) const override {
        return std::sqrt(getArgument()->evaluate(x, variables));
    }
    
    ExprPtr differentiate(const std::string& var) const override {
        // d/dx[√f] = f'/(2√f)
        auto arg = getArgument();
        auto arg_deriv = arg->differentiate(var);
        auto sqrt_arg = std::make_shared<Sqrt>(arg);
        auto two_sqrt = std::make_shared<Multiplication>(
            std::make_shared<Constant>(2.0),
            sqrt_arg
        );
        return std::make_shared<Division>(arg_deriv, two_sqrt);
    }
    
    std::string toString() const override {
        return "sqrt(" + getArgument()->toString() + ")";
    }
    
    std::string toLatex() const override {
        return "\\sqrt{" + getArgument()->toLatex() + "}";
    }
    
    ExprPtr simplify() const override {
        return std::make_shared<Sqrt>(getArgument()->simplify());
    }
    
    bool isLinear(const std::string& var) const override {
        return false;
    }
    
    int getDegree(const std::string& var) const override {
        return 1;  // Non-polynomial
    }
    
    ExprPtr substitute(const std::string& var, ExprPtr replacement) const override {
        return std::make_shared<Sqrt>(
            getArgument()->substitute(var, replacement)
        );
    }
};

class Abs : public UnaryOperation {
public:
    explicit Abs(ExprPtr argument)
        : UnaryOperation(std::move(argument)) {}
    
    ExprType getType() const override {
        return ExprType::ABS;
    }
    
    double evaluate(double x, const std::unordered_map<std::string, double>& variables) const override {
        return std::abs(getArgument()->evaluate(x, variables));
    }
    
    ExprPtr differentiate(const std::string& var) const override {
        // d/dx[|f|] = f'·(f/|f|) for f ≠ 0
        auto arg = getArgument();
        auto arg_deriv = arg->differentiate(var);
        auto abs_arg = std::make_shared<Abs>(arg);
        auto ratio = std::make_shared<Division>(arg, abs_arg);
        return std::make_shared<Multiplication>(arg_deriv, ratio);
    }
    
    std::string toString() const override {
        return "|" + getArgument()->toString() + "|";
    }
    
    std::string toLatex() const override {
        return "\\left|" + getArgument()->toLatex() + "\\right|";
    }
    
    ExprPtr simplify() const override {
        return std::make_shared<Abs>(getArgument()->simplify());
    }
    
    bool isLinear(const std::string& var) const override {
        return false;
    }
    
    int getDegree(const std::string& var) const override {
        return 1;
    }
    
    ExprPtr substitute(const std::string& var, ExprPtr replacement) const override {
        return std::make_shared<Abs>(
            getArgument()->substitute(var, replacement)
        );
    }
};

} // namespace ode_solver::symbolic