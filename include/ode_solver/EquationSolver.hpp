#ifndef ODE_SOLVER_EXPRESSION_HPP
#define ODE_SOLVER_EXPRESSION_HPP

#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace ode_solver {
namespace symbolic {

// Forward Declarations
class Expression;

using ExprPtr = std::shared_ptr<Expression>;

/************************************Expressions************************************/


// Expression for mathematical operations
class Expression {
public:
    virtual ~Expression() = default;
    

    virtual double evaluate(double x) const = 0;
    
    virtual ExprPtr differentiate() const = 0;
    
    virtual std::string toString() const = 0;
    
    virtual ExprPtr simplify() const = 0;
    
    virtual bool isConstant() const = 0;

    virtual bool isZero() const { return false; }
    
    virtual bool isOne() const { return false; }
};


// Variable expression

class Variable : public Expression {
public:
    Variable() = default;
    
    double evaluate(double x) const override {
        return x;
    }
    
    ExprPtr differentiate() const override;
    
    std::string toString() const override {
        return "x";
    }
    
    ExprPtr simplify() const override {
        return std::make_shared<Variable>();
    }
    
    bool isConstant() const override { return false; }
};

// Constant expression

class Constant : public Expression {
private:
    double value_;
    
public:
    explicit Constant(double value) : value_(value) {}
    
    double getValue() const { return value_; }
    
    double evaluate([[maybe_unused]] double x) const override {
        (void)x;
        return value_;
    }
    
    ExprPtr differentiate() const override {
        return std::make_shared<Constant>(0.0);
    }
    
    std::string toString() const override {
        std::ostringstream oss;
        oss << value_;
        return oss.str();
    }
    
    ExprPtr simplify() const override {
        return std::make_shared<Constant>(value_);
    }
    
    bool isConstant() const override { return true; }
    bool isZero() const override { return value_ == 0.0; }
    bool isOne() const override { return value_ == 1.0; }
};



/************************************Binary Operations************************************/



//Addition
class Addition : public Expression {
private:
    ExprPtr left_;
    ExprPtr right_;
    
public:
    Addition(ExprPtr left, ExprPtr right) 
        : left_(std::move(left)), right_(std::move(right)) {}
    
    double evaluate(double x) const override {
        return left_->evaluate(x) + right_->evaluate(x);
    }
    
    ExprPtr differentiate() const override {
        return std::make_shared<Addition>(
            left_->differentiate(),
            right_->differentiate()
        );
    }
    
    std::string toString() const override {
        return "(" + left_->toString() + " + " + right_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto l = left_->simplify();
        auto r = right_->simplify();
        
        // 0 + expr = expr
        if (l->isZero()) return r;
        if (r->isZero()) return l;
        
        // const + const = const
        if (l->isConstant() && r->isConstant()) {
            double val = l->evaluate(0) + r->evaluate(0);
            return std::make_shared<Constant>(val);
        }
        
        return std::make_shared<Addition>(l, r);
    }
    
    bool isConstant() const override {
        return left_->isConstant() && right_->isConstant();
    }
};

// Subtraction

class Subtraction : public Expression {
private:
    ExprPtr left_;
    ExprPtr right_;

public:
    Subtraction(ExprPtr left, ExprPtr right)
        : left_(std::move(left)), right_(std::move(right)) {}
    
    double evaluate(double x) const override {
        return left_->evaluate(x) - right_->evaluate(x);
    }
    
    ExprPtr differentiate() const override {
        return std::make_shared<Subtraction>(
            left_->differentiate(),
            right_->differentiate()
        );
    }
    
    std::string toString() const override {
        return "(" + left_->toString() + " - " + right_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto l = left_->simplify();
        auto r = right_->simplify();
        
        // expr - 0 = expr
        if (r->isZero()) return l;
        // 0 - expr = -expr
        if (l->isZero()) {
            return std::make_shared<Multiplication>(
                std::make_shared<Constant>(-1.0), r
            );
        }
        
        if (l->isConstant() && r->isConstant()) {
            double val = l->evaluate(0) - r->evaluate(0);
            return std::make_shared<Constant>(val);
        }
        
        return std::make_shared<Subtraction>(l, r);
    }
    
    bool isConstant() const override {
        return left_->isConstant() && right_->isConstant();
    }
};

//Multiplication

class Multiplication : public Expression {
private:
    ExprPtr left_;
    ExprPtr right_;
    
public:
    Multiplication(ExprPtr left, ExprPtr right)
        : left_(std::move(left)), right_(std::move(right)) {}
    
    double evaluate(double x) const override {
        return left_->evaluate(x) * right_->evaluate(x);
    }
    
    // Product rule: (f*g)' = f'*g + f*g'
    ExprPtr differentiate() const override {
        return std::make_shared<Addition>(
            std::make_shared<Multiplication>(left_->differentiate(), right_),
            std::make_shared<Multiplication>(left_, right_->differentiate())
        );
    }
    
    std::string toString() const override {
        return "(" + left_->toString() + " * " + right_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto l = left_->simplify();
        auto r = right_->simplify();
        
        // 0 * expr = 0
        if (l->isZero() || r->isZero()) {
            return std::make_shared<Constant>(0.0);
        }
        // 1 * expr = expr
        if (l->isOne()) return r;
        if (r->isOne()) return l;
        
        if (l->isConstant() && r->isConstant()) {
            double val = l->evaluate(0) * r->evaluate(0);
            return std::make_shared<Constant>(val);
        }
        
        return std::make_shared<Multiplication>(l, r);
    }
    
    bool isConstant() const override {
        return left_->isConstant() && right_->isConstant();
    }
};

//Division 

class Division : public Expression {
private:
    ExprPtr left_;
    ExprPtr right_;
    
public:
    Division(ExprPtr left, ExprPtr right)
        : left_(std::move(left)), right_(std::move(right)) {}
    
    double evaluate(double x) const override {
        double denominator = right_->evaluate(x);
        if (std::abs(denominator) < 1e-12) {
            throw std::runtime_error("Division by zero in expression evaluation");
        }
        return left_->evaluate(x) / denominator;
    }
    
    // Quotient rule: (f/g)' = (f'*g - f*g') / g^2
    ExprPtr differentiate() const override {
        auto numerator = std::make_shared<Subtraction>(
            std::make_shared<Multiplication>(left_->differentiate(), right_),
            std::make_shared<Multiplication>(left_, right_->differentiate())
        );
        auto denominator = std::make_shared<Power>(right_, std::make_shared<Constant>(2.0));
        return std::make_shared<Division>(numerator, denominator);
    }
    
    std::string toString() const override {
        return "(" + left_->toString() + " / " + right_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto l = left_->simplify();
        auto r = right_->simplify();
        
        // 0 / expr = 0
        if (l->isZero()) return std::make_shared<Constant>(0.0);
        // expr / 1 = expr
        if (r->isOne()) return l;
        
        if (l->isConstant() && r->isConstant()) {
            double val = l->evaluate(0) / r->evaluate(0);
            return std::make_shared<Constant>(val);
        }
        
        return std::make_shared<Division>(l, r);
    }
    
    bool isConstant() const override {
        return left_->isConstant() && right_->isConstant();
    }
};

//Power

class Power : public Expression {
private:
    ExprPtr base_;
    ExprPtr exponent_;
    
public:
    Power(ExprPtr base, ExprPtr exponent)
        : base_(std::move(base)), exponent_(std::move(exponent)) {}
    
    double evaluate(double x) const override {
        return std::pow(base_->evaluate(x), exponent_->evaluate(x));
    }
    

    ExprPtr differentiate() const override {
        if (exponent_->isConstant()) {
            // Power rule for constant exponent
            auto n = std::make_shared<Constant>(exponent_->evaluate(0));
            auto n_minus_1 = std::make_shared<Constant>(exponent_->evaluate(0) - 1.0);
            
            return std::make_shared<Multiplication>(
                std::make_shared<Multiplication>(
                    n,
                    std::make_shared<Power>(base_, n_minus_1)
                ),
                base_->differentiate()
            );
        } else {
            // General case - more complex, requires logarithm
            throw std::runtime_error("Differentiation of f(x)^g(x) not yet implemented");
        }
    }
    
    std::string toString() const override {
        return "(" + base_->toString() + "^" + exponent_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto b = base_->simplify();
        auto e = exponent_->simplify();
        
        // expr^0 = 1
        if (e->isZero()) return std::make_shared<Constant>(1.0);
        // expr^1 = expr
        if (e->isOne()) return b;
        // 0^n = 0 (n != 0)
        if (b->isZero()) return std::make_shared<Constant>(0.0);
        // 1^n = 1
        if (b->isOne()) return std::make_shared<Constant>(1.0);
        
        if (b->isConstant() && e->isConstant()) {
            double val = std::pow(b->evaluate(0), e->evaluate(0));
            return std::make_shared<Constant>(val);
        }
        
        return std::make_shared<Power>(b, e);
    }
    
    bool isConstant() const override {
        return base_->isConstant() && exponent_->isConstant();
    }
};

// Transcendental Functions


//Sine function: sin(expr)

class Sin : public Expression {
private:
    ExprPtr argument_;
    
public:
    explicit Sin(ExprPtr argument) : argument_(std::move(argument)) {}
    
    double evaluate(double x) const override {
        return std::sin(argument_->evaluate(x));
    }
    
    // Chain rule: (sin(f))' = cos(f) * f'
    ExprPtr differentiate() const override {
        return std::make_shared<Multiplication>(
            std::make_shared<Cos>(argument_),
            argument_->differentiate()
        );
    }
    
    std::string toString() const override {
        return "sin(" + argument_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto arg = argument_->simplify();
        if (arg->isConstant()) {
            double val = std::sin(arg->evaluate(0));
            return std::make_shared<Constant>(val);
        }
        return std::make_shared<Sin>(arg);
    }
    
    bool isConstant() const override { return argument_->isConstant(); }
};


//Cosine function: cos(expr)


class Cos : public Expression {
private:
    ExprPtr argument_;
    
public:
    explicit Cos(ExprPtr argument) : argument_(std::move(argument)) {}
    
    double evaluate(double x) const override {
        return std::cos(argument_->evaluate(x));
    }
    
    // Chain rule: (cos(f))' = -sin(f) * f'
    ExprPtr differentiate() const override {
        return std::make_shared<Multiplication>(
            std::make_shared<Multiplication>(
                std::make_shared<Constant>(-1.0),
                std::make_shared<Sin>(argument_)
            ),
            argument_->differentiate()
        );
    }
    
    std::string toString() const override {
        return "cos(" + argument_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto arg = argument_->simplify();
        if (arg->isConstant()) {
            double val = std::cos(arg->evaluate(0));
            return std::make_shared<Constant>(val);
        }
        return std::make_shared<Cos>(arg);
    }
    
    bool isConstant() const override { return argument_->isConstant(); }
};

//Exponential function: exp(expr)

class Exp : public Expression {
private:
    ExprPtr argument_;
    
public:
    explicit Exp(ExprPtr argument) : argument_(std::move(argument)) {}
    
    double evaluate(double x) const override {
        return std::exp(argument_->evaluate(x));
    }
    
    // Chain rule: (e^f)' = e^f * f'
    ExprPtr differentiate() const override {
        return std::make_shared<Multiplication>(
            std::make_shared<Exp>(argument_),
            argument_->differentiate()
        );
    }
    
    std::string toString() const override {
        return "exp(" + argument_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto arg = argument_->simplify();
        if (arg->isZero()) return std::make_shared<Constant>(1.0);
        if (arg->isConstant()) {
            double val = std::exp(arg->evaluate(0));
            return std::make_shared<Constant>(val);
        }
        return std::make_shared<Exp>(arg);
    }
    
    bool isConstant() const override { return argument_->isConstant(); }
};

//Natural Log

class Log : public Expression {
private:
    ExprPtr argument_;
    
public:
    explicit Log(ExprPtr argument) : argument_(std::move(argument)) {}
    
    double evaluate(double x) const override {
        double val = argument_->evaluate(x);
        if (val <= 0.0) {
            throw std::runtime_error("Logarithm of non-positive number");
        }
        return std::log(val);
    }
    
    // Chain rule: (ln(f))' = f'/f
    ExprPtr differentiate() const override {
        return std::make_shared<Division>(
            argument_->differentiate(),
            argument_
        );
    }
    
    std::string toString() const override {
        return "ln(" + argument_->toString() + ")";
    }
    
    ExprPtr simplify() const override {
        auto arg = argument_->simplify();
        if (arg->isOne()) return std::make_shared<Constant>(0.0);
        if (arg->isConstant()) {
            double val = std::log(arg->evaluate(0));
            return std::make_shared<Constant>(val);
        }
        return std::make_shared<Log>(arg);
    }
    
    bool isConstant() const override { return argument_->isConstant(); }
};


// Helper Functions for Building Expressions

inline ExprPtr var() { return std::make_shared<Variable>(); }
inline ExprPtr constant(double val) { return std::make_shared<Constant>(val); }
inline ExprPtr operator+(ExprPtr left, ExprPtr right) { 
    return std::make_shared<Addition>(left, right); 
}
inline ExprPtr operator-(ExprPtr left, ExprPtr right) { 
    return std::make_shared<Subtraction>(left, right); 
}
inline ExprPtr operator*(ExprPtr left, ExprPtr right) { 
    return std::make_shared<Multiplication>(left, right); 
}
inline ExprPtr operator/(ExprPtr left, ExprPtr right) { 
    return std::make_shared<Division>(left, right); 
}
inline ExprPtr pow(ExprPtr base, ExprPtr exp) { 
    return std::make_shared<Power>(base, exp); 
}
inline ExprPtr sin(ExprPtr arg) { return std::make_shared<Sin>(arg); }
inline ExprPtr cos(ExprPtr arg) { return std::make_shared<Cos>(arg); }
inline ExprPtr exp(ExprPtr arg) { return std::make_shared<Exp>(arg); }
inline ExprPtr log(ExprPtr arg) { return std::make_shared<Log>(arg); }

// Implement Variable::differentiate() here to avoid forward declaration issues

inline ExprPtr Variable::differentiate() const {
    return std::make_shared<Constant>(1.0);
}

} // namespace symbolic
} // namespace ode_solver

#endif // ODE_SOLVER_EXPRESSION_HPP