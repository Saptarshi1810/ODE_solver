

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

// Forward Declarations


class Expression;
class ExpressionVisitor;
using ExprPtr = std::shared_ptr<Expression>;

// Enumeration: Expression Types

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

// Abstract Base Class: Expression


class Expression : public std::enable_shared_from_this<Expression> {
public:
    virtual ~Expression() = default;
    
   
    virtual double evaluate(double x, 
                           const std::unordered_map<std::string, double>& variables = {}) const = 0;
    

    virtual ExprPtr differentiate(const std::string& var = "x") const = 0;
    
 /
    virtual ExprPtr simplify() const = 0;
 
    virtual std::string toString() const = 0;
    

    virtual std::string toLatex() const = 0;

    virtual ExprType getType() const = 0;
    

    virtual void accept(ExpressionVisitor& visitor) const = 0;
    

    virtual ExprPtr clone() const = 0;

    virtual bool isConstant() const = 0;
    

    virtual bool isZero() const { return false; }

    virtual bool isOne() const { return false; }
    

    virtual bool isLinear(const std::string& var = "x") const = 0;
    

    virtual int getDegree(const std::string& var = "x") const = 0;

    virtual std::vector<std::string> getVariables() const = 0;
    
    virtual int getComplexity() const = 0;
    

  
    virtual ExprPtr expand() const { return clone(); }
    

    virtual ExprPtr factor() const { return clone(); }
    

    virtual ExprPtr substitute(const std::string& var, ExprPtr replacement) const = 0;
    

    
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


class Constant;


class Variable;


class Addition;


class Subtraction;

class Multiplication;


class Division;


class Power;


class Negation;

class Sin;

class Cos;


class Tan;

class Exp;

class Log;

class Sqrt;

class Abs;


ExprPtr var(const std::string& name = "x");

ExprPtr constant(double value);

ExprPtr add(ExprPtr left, ExprPtr right);

ExprPtr subtract(ExprPtr left, ExprPtr right);

ExprPtr multiply(ExprPtr left, ExprPtr right);


ExprPtr divide(ExprPtr left, ExprPtr right);


ExprPtr pow(ExprPtr base, ExprPtr exponent);

ExprPtr negate(ExprPtr argument);

ExprPtr sin(ExprPtr argument);

ExprPtr cos(ExprPtr argument);

ExprPtr tan(ExprPtr argument);

ExprPtr exp(ExprPtr argument);

ExprPtr log(ExprPtr argument);

ExprPtr sqrt(ExprPtr argument);

ExprPtr abs(ExprPtr argument);

class ExpressionParser {
public:

    static ExprPtr parse(const std::string& expression);

    void setVariables(const std::vector<std::string>& variables);

    void addConstant(const std::string& name, double value);

private:
    std::unordered_map<std::string, double> constants_;
    std::vector<std::string> variables_;
    
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


class ExpressionCompiler {
public:
 
    static Function compile(const ExprPtr& expr, const std::string& var = "x");

    static std::pair<Function, DerivativeFunction> compileWithDerivative(
        const ExprPtr& expr, const std::string& var = "x");
};

class ExpressionSimplifier {
public:

    static ExprPtr simplify(const ExprPtr& expr);

    void enableRule(const std::string& rule_name, bool enable = true);
    
private:
    std::unordered_map<std::string, bool> rules_;
    
    ExprPtr applyAlgebraicRules(const ExprPtr& expr);
    ExprPtr applyTrigonometricRules(const ExprPtr& expr);
    ExprPtr constantFolding(const ExprPtr& expr);
};

class AutomaticDifferentiator {
public:

    static ExprPtr differentiate(const ExprPtr& expr, 
                                 const std::string& var = "x",
                                 int order = 1);
    
 
    static std::vector<ExprPtr> gradient(
        const ExprPtr& expr,
        const std::vector<std::string>& variables);

    static std::vector<std::vector<ExprPtr>> jacobian(
        const std::vector<ExprPtr>& expressions,
        const std::vector<std::string>& variables);
};

} // namespace symbolic
} // namespace ode_solver

#endif // ODE_SOLVER_SYMBOLIC_ENGINE_HPP