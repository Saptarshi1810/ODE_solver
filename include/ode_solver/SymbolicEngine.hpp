/**
 * @file SymbolicEngine.hpp
 * @brief Symbolic expression representation, differentiation, and simplification
 * @author Your Name
 * @version 1.0.0
 * 
 * This header provides a complete symbolic mathematics engine for:
 * - Building expression trees
 * - Automatic differentiation
 * - Expression simplification
 * - Parsing mathematical expressions
 * - Converting between symbolic and numeric representations
 * 
 * Design Patterns:
 * - Composite: Expression tree structure
 * - Visitor: Expression traversal and transformation
 * - Interpreter: Expression evaluation
 */

#ifndef ODE_SOLVER_SYMBOLIC_ENGINE_HPP
#define ODE_SOLVER_SYMBOLIC_ENGINE_HPP

#include "EquationSolver.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

namespace ode_solver {
namespace symbolic {

// ============================================================================
// Forward Declarations
// ============================================================================

class Expression;
class ExpressionVisitor;
using ExprPtr = std::shared_ptr<Expression>;

// ============================================================================
// Enumeration: Expression Types
// ============================================================================

/**
 * @brief Type classification for expressions
 */
enum class ExprType {
    CONSTANT,           ///< Constant value
    VARIABLE,           ///< Variable (x)
    ADDITION,           ///< Binary addition
    SUBTRACTION,        ///< Binary subtraction
    MULTIPLICATION,     ///< Binary multiplication
    DIVISION,           ///< Binary division
    POWER,              ///< Power/exponentiation
    NEGATION,           ///< Unary negation
    SIN,                ///< Sine function
    COS,                ///< Cosine function
    TAN,                ///< Tangent function
    EXP,                ///< Exponential e^x
    LOG,                ///< Natural logarithm
    SQRT,               ///< Square root
    ABS                 ///< Absolute value
};

// ============================================================================
// Abstract Base Class: Expression
// ============================================================================

/**
 * @brief Abstract base class for all symbolic expressions
 * 
 * Represents a node in an expression tree. Provides interface for:
 * - Numerical evaluation
 * - Symbolic differentiation
 * - Expression simplification
 * - String representation
 * - Type queries
 * 
 * Thread Safety: Expression objects are immutable after construction
 * and can be safely shared across threads.
 */
class Expression : public std::enable_shared_from_this<Expression> {
public:
    virtual ~Expression() = default;
    
    // ========================================================================
    // Pure Virtual Methods
    // ========================================================================
    
    /**
     * @brief Evaluate the expression numerically at given point
     * @param x Value at which to evaluate
     * @param variables Optional variable bindings (for multi-variable expressions)
     * @return Numerical result
     * @throws std::runtime_error if evaluation fails (e.g., log of negative)
     */
    virtual double evaluate(double x, 
                           const std::unordered_map<std::string, double>& variables = {}) const = 0;
    
    /**
     * @brief Compute symbolic derivative d/dx of this expression
     * @param var Variable to differentiate with respect to (default: "x")
     * @return New expression representing the derivative
     */
    virtual ExprPtr differentiate(const std::string& var = "x") const = 0;
    
    /**
     * @brief Simplify this expression algebraically
     * @return Simplified expression (may return this if already simple)
     * 
     * Applies rules like:
     * - 0 + x → x
     * - 1 * x → x
     * - x^0 → 1
     * - Constant folding
     */
    virtual ExprPtr simplify() const = 0;
    
    /**
     * @brief Get string representation of expression
     * @return Human-readable string (e.g., "x^2 + 3*x - 5")
     */
    virtual std::string toString() const = 0;
    
    /**
     * @brief Get LaTeX representation of expression
     * @return LaTeX-formatted string
     */
    virtual std::string toLatex() const = 0;
    
    /**
     * @brief Get expression type
     * @return Type classification
     */
    virtual ExprType getType() const = 0;
    
    /**
     * @brief Accept visitor for traversal (Visitor pattern)
     * @param visitor Visitor object
     */
    virtual void accept(ExpressionVisitor& visitor) const = 0;
    
    /**
     * @brief Clone this expression (deep copy)
     * @return New independent copy
     */
    virtual ExprPtr clone() const = 0;
    
    // ========================================================================
    // Query Methods
    // ========================================================================
    
    /**
     * @brief Check if expression is constant (doesn't depend on any variables)
     * @return True if expression has no variable dependencies
     */
    virtual bool isConstant() const = 0;
    
    /**
     * @brief Check if expression is identically zero
     * @return True if expression equals 0 for all variable values
     */
    virtual bool isZero() const { return false; }
    
    /**
     * @brief Check if expression is identically one
     * @return True if expression equals 1 for all variable values
     */
    virtual bool isOne() const { return false; }
    
    /**
     * @brief Check if expression is linear in given variable
     * @param var Variable name
     * @return True if expression is linear (degree ≤ 1)
     */
    virtual bool isLinear(const std::string& var = "x") const = 0;
    
    /**
     * @brief Get polynomial degree in given variable
     * @param var Variable name
     * @return Degree (0 for constant, -1 if not polynomial)
     */
    virtual int getDegree(const std::string& var = "x") const = 0;
    
    /**
     * @brief Get all variables appearing in this expression
     * @return Set of variable names
     */
    virtual std::vector<std::string> getVariables() const = 0;
    
    /**
     * @brief Get complexity measure (number of operations)
     * @return Complexity score
     */
    virtual int getComplexity() const = 0;
    
    // ========================================================================
    // Transformation Methods
    // ========================================================================
    
    /**
     * @brief Expand expression (distribute products, combine like terms)
     * @return Expanded expression
     */
    virtual ExprPtr expand() const { return clone(); }
    
    /**
     * @brief Factor expression if possible
     * @return Factored expression
     */
    virtual ExprPtr factor() const { return clone(); }
    
    /**
     * @brief Substitute variable with another expression
     * @param var Variable name to substitute
     * @param replacement Expression to substitute in
     * @return New expression with substitution applied
     */
    virtual ExprPtr substitute(const std::string& var, ExprPtr replacement) const = 0;
    
    // ========================================================================
    // Operator Overloads (for convenient expression building)
    // ========================================================================
    
    ExprPtr operator+(const ExprPtr& other) const;
    ExprPtr operator-(const ExprPtr& other) const;
    ExprPtr operator*(const ExprPtr& other) const;
    ExprPtr operator/(const ExprPtr& other) const;
    ExprPtr operator-() const; // Unary negation
    
protected:
    // Helper for checking approximate equality
    static bool approxEqual(double a, double b, double epsilon = 1e-10) {
        return std::abs(a - b) < epsilon;
    }
};

// ============================================================================
// Abstract: Visitor Pattern for Expression Traversal
// ============================================================================

/**
 * @brief Abstract visitor for expression tree traversal
 * 
 * Implements Visitor pattern for operations on expression trees.
 * Useful for:
 * - Custom simplification rules
 * - Code generation
 * - Expression analysis
 */
class ExpressionVisitor {
public:
    virtual ~ExpressionVisitor() = default;
    
    // Visit methods for each expression type
    virtual void visitConstant(const class Constant& expr) = 0;
    virtual void visitVariable(const class Variable& expr) = 0;
    virtual void visitBinaryOp(const class BinaryOperation& expr) = 0;
    virtual void visitUnaryOp(const class UnaryOperation& expr) = 0;
    virtual void visitFunction(const class FunctionCall& expr) = 0;
};

// ============================================================================
// Abstract: Binary Operation Base
// ============================================================================

/**
 * @brief Base class for binary operations (left op right)
 */
class BinaryOperation : public Expression {
public:
    BinaryOperation(ExprPtr left, ExprPtr right)
        : left_(std::move(left)), right_(std::move(right)) {}
    
    const ExprPtr& getLeft() const { return left_; }
    const ExprPtr& getRight() const { return right_; }
    
    bool isConstant() const override {
        return left_->isConstant() && right_->isConstant();
    }
    
    std::vector<std::string> getVariables() const override {
        auto left_vars = left_->getVariables();
        auto right_vars = right_->getVariables();
        left_vars.insert(left_vars.end(), right_vars.begin(), right_vars.end());
        // Remove duplicates
        std::sort(left_vars.begin(), left_vars.end());
        left_vars.erase(std::unique(left_vars.begin(), left_vars.end()), left_vars.end());
        return left_vars;
    }
    
    int getComplexity() const override {
        return 1 + left_->getComplexity() + right_->getComplexity();
    }
    
    void accept(ExpressionVisitor& visitor) const override {
        visitor.visitBinaryOp(*this);
    }

protected:
    ExprPtr left_;
    ExprPtr right_;
};

// ============================================================================
// Abstract: Unary Operation Base
// ============================================================================

/**
 * @brief Base class for unary operations (op argument)
 */
class UnaryOperation : public Expression {
public:
    explicit UnaryOperation(ExprPtr argument)
        : argument_(std::move(argument)) {}
    
    const ExprPtr& getArgument() const { return argument_; }
    
    bool isConstant() const override {
        return argument_->isConstant();
    }
    
    std::vector<std::string> getVariables() const override {
        return argument_->getVariables();
    }
    
    int getComplexity() const override {
        return 1 + argument_->getComplexity();
    }
    
    void accept(ExpressionVisitor& visitor) const override {
        visitor.visitUnaryOp(*this);
    }

protected:
    ExprPtr argument_;
};

// ============================================================================
// Concrete Expression Classes (Forward Declarations)
// ============================================================================

/**
 * @brief Constant numerical value
 */
class Constant;

/**
 * @brief Variable symbol (e.g., "x", "y", "t")
 */
class Variable;

/**
 * @brief Addition: left + right
 */
class Addition;

/**
 * @brief Subtraction: left - right
 */
class Subtraction;

/**
 * @brief Multiplication: left * right
 */
class Multiplication;

/**
 * @brief Division: left / right
 */
class Division;

/**
 * @brief Power/Exponentiation: base ^ exponent
 */
class Power;

/**
 * @brief Unary negation: -argument
 */
class Negation;

/**
 * @brief Sine function: sin(argument)
 */
class Sin;

/**
 * @brief Cosine function: cos(argument)
 */
class Cos;

/**
 * @brief Tangent function: tan(argument)
 */
class Tan;

/**
 * @brief Exponential function: e^argument
 */
class Exp;

/**
 * @brief Natural logarithm: ln(argument)
 */
class Log;

/**
 * @brief Square root: √argument
 */
class Sqrt;

/**
 * @brief Absolute value: |argument|
 */
class Abs;

// ============================================================================
// Expression Builder Functions (Factory Methods)
// ============================================================================

/**
 * @brief Create variable expression
 * @param name Variable name (default: "x")
 * @return Expression representing the variable
 */
ExprPtr var(const std::string& name = "x");

/**
 * @brief Create constant expression
 * @param value Numerical value
 * @return Expression representing the constant
 */
ExprPtr constant(double value);

/**
 * @brief Create addition expression
 */
ExprPtr add(ExprPtr left, ExprPtr right);

/**
 * @brief Create subtraction expression
 */
ExprPtr subtract(ExprPtr left, ExprPtr right);

/**
 * @brief Create multiplication expression
 */
ExprPtr multiply(ExprPtr left, ExprPtr right);

/**
 * @brief Create division expression
 */
ExprPtr divide(ExprPtr left, ExprPtr right);

/**
 * @brief Create power expression
 */
ExprPtr pow(ExprPtr base, ExprPtr exponent);

/**
 * @brief Create negation expression
 */
ExprPtr negate(ExprPtr argument);

/**
 * @brief Create sine expression
 */
ExprPtr sin(ExprPtr argument);

/**
 * @brief Create cosine expression
 */
ExprPtr cos(ExprPtr argument);

/**
 * @brief Create tangent expression
 */
ExprPtr tan(ExprPtr argument);

/**
 * @brief Create exponential expression
 */
ExprPtr exp(ExprPtr argument);

/**
 * @brief Create logarithm expression
 */
ExprPtr log(ExprPtr argument);

/**
 * @brief Create square root expression
 */
ExprPtr sqrt(ExprPtr argument);

/**
 * @brief Create absolute value expression
 */
ExprPtr abs(ExprPtr argument);

// ============================================================================
// Expression Parser
// ============================================================================

/**
 * @brief Parser for mathematical expressions from strings
 * 
 * Supports:
 * - Basic arithmetic: +, -, *, /, ^
 * - Functions: sin, cos, tan, exp, ln, sqrt, abs
 * - Parentheses for grouping
 * - Constants: numbers, pi, e
 * - Variables: single letters or multi-character names
 * 
 * Example: "3*x^2 + 2*sin(x) - 5"
 */
class ExpressionParser {
public:
    /**
     * @brief Parse string into expression tree
     * @param expression Mathematical expression string
     * @return Parsed expression
     * @throws std::runtime_error if parsing fails
     */
    static ExprPtr parse(const std::string& expression);
    
    /**
     * @brief Set custom variable names to recognize
     * @param variables List of valid variable names
     */
    void setVariables(const std::vector<std::string>& variables);
    
    /**
     * @brief Add custom constant definition
     * @param name Constant name
     * @param value Constant value
     */
    void addConstant(const std::string& name, double value);

private:
    std::unordered_map<std::string, double> constants_;
    std::vector<std::string> variables_;
    
    // Recursive descent parser methods
    ExprPtr parseExpression(const std::string& str, size_t& pos);
    ExprPtr parseTerm(const std::string& str, size_t& pos);
    ExprPtr parseFactor(const std::string& str, size_t& pos);
    ExprPtr parsePower(const std::string& str, size_t& pos);
    ExprPtr parsePrimary(const std::string& str, size_t& pos);
    ExprPtr parseFunction(const std::string& name, const std::string& str, size_t& pos);
    
    void skipWhitespace(const std::string& str, size_t& pos);
    double parseNumber(const std::string& str, size_t& pos);
    std::string parseIdentifier(const std::string& str, size_t& pos);
};

// ============================================================================
// Utility Classes
// ============================================================================

/**
 * @brief Converts expressions to std::function for numerical use
 */
class ExpressionCompiler {
public:
    /**
     * @brief Compile expression to std::function
     * @param expr Expression to compile
     * @param var Variable name (default: "x")
     * @return Compiled function
     */
    static Function compile(const ExprPtr& expr, const std::string& var = "x");
    
    /**
     * @brief Compile expression and its derivative
     * @param expr Expression to compile
     * @param var Variable name
     * @return Pair of (function, derivative)
     */
    static std::pair<Function, DerivativeFunction> compileWithDerivative(
        const ExprPtr& expr, const std::string& var = "x");
};

/**
 * @brief Performs advanced simplification
 */
class ExpressionSimplifier {
public:
    /**
     * @brief Apply comprehensive simplification
     * @param expr Expression to simplify
     * @return Simplified expression
     */
    static ExprPtr simplify(const ExprPtr& expr);
    
    /**
     * @brief Enable/disable specific simplification rules
     */
    void enableRule(const std::string& rule_name, bool enable = true);
    
private:
    std::unordered_map<std::string, bool> rules_;
    
    ExprPtr applyAlgebraicRules(const ExprPtr& expr);
    ExprPtr applyTrigonometricRules(const ExprPtr& expr);
    ExprPtr constantFolding(const ExprPtr& expr);
};

/**
 * @brief Handles automatic differentiation
 */
class AutomaticDifferentiator {
public:
    /**
     * @brief Compute nth derivative
     * @param expr Expression to differentiate
     * @param var Variable to differentiate with respect to
     * @param order Order of derivative (1 = first derivative, 2 = second, etc.)
     * @return nth derivative
     */
    static ExprPtr differentiate(const ExprPtr& expr, 
                                 const std::string& var = "x",
                                 int order = 1);
    
    /**
     * @brief Compute gradient (vector of partial derivatives)
     * @param expr Scalar expression
     * @param variables List of variables
     * @return Vector of partial derivatives
     */
    static std::vector<ExprPtr> gradient(
        const ExprPtr& expr,
        const std::vector<std::string>& variables);
    
    /**
     * @brief Compute Jacobian matrix of vector-valued function
     * @param expressions Vector of expressions [f1, f2, ..., fn]
     * @param variables List of variables [x1, x2, ..., xm]
     * @return n×m matrix of partial derivatives
     */
    static std::vector<std::vector<ExprPtr>> jacobian(
        const std::vector<ExprPtr>& expressions,
        const std::vector<std::string>& variables);
};

} // namespace symbolic
} // namespace ode_solver

#endif // ODE_SOLVER_SYMBOLIC_ENGINE_HPP