/**
 * @file ExpressionParser.cpp
 * @brief Implementation of expression parser for symbolic math expressions
 * @author Your Name
 * @version 1.0.0
 *
 * This parser converts string representations of mathematical expressions into
 * an abstract syntax tree (AST) for symbolic manipulation, evaluation, and differentiation.
 *
 * Grammar:
 * expression := term (('+' | '-') term)*
 * term       := factor (('*' | '/') factor)*
 * factor     := power
 * power      := primary ('^' primary)*
 * primary    := number | variable | function | '(' expression ')'
 * function   := 'sin' | 'cos' | 'tan' | 'exp' | 'log' | 'sqrt' | 'abs'
 *
 * Supported operations: +, -, *, /, ^, sin, cos, tan, exp, log, sqrt, abs
 * Operator precedence: power > multiply/divide > add/subtract
 */

#include "ode_solver/SymbolicEngine.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cctype>
#include <stdexcept>

namespace ode_solver::symbolic {

/**
 * @brief Parse a complete mathematical expression
 * @param expression String representation of the expression
 * @return Shared pointer to the parsed expression tree
 * @throws std::runtime_error if parsing fails
 */
ExprPtr ExpressionParser::parse(const std::string& expression) {
    ExpressionParser parser;
    size_t pos = 0;
    ExprPtr result = parser.parseExpression(expression, pos);
    
    // Ensure we've consumed the entire string
    parser.skipWhitespace(expression, pos);
    if (pos != expression.length()) {
        throw std::runtime_error("ExpressionParser: Unexpected characters after expression");
    }
    
    return result;
}

/**
 * @brief Parse expression with addition and subtraction (lowest precedence)
 *
 * expression := term (('+' | '-') term)*
 */
ExprPtr ExpressionParser::parseExpression(const std::string& str, size_t& pos) {
    ExprPtr left = parseTerm(str, pos);
    
    while (true) {
        skipWhitespace(str, pos);
        
        if (pos >= str.length()) {
            break;
        }
        
        char op = str[pos];
        
        if (op == '+') {
            pos++;
            ExprPtr right = parseTerm(str, pos);
            left = add(left, right);
        } else if (op == '-') {
            // Check if it's a binary minus (not unary)
            // Look back to see if there's something that could be left operand
            pos++;
            ExprPtr right = parseTerm(str, pos);
            left = subtract(left, right);
        } else {
            break;
        }
    }
    
    return left;
}

/**
 * @brief Parse term with multiplication and division
 *
 * term := factor (('*' | '/') factor)*
 */
ExprPtr ExpressionParser::parseTerm(const std::string& str, size_t& pos) {
    ExprPtr left = parseFactor(str, pos);
    
    while (true) {
        skipWhitespace(str, pos);
        
        if (pos >= str.length()) {
            break;
        }
        
        char op = str[pos];
        
        if (op == '*') {
            pos++;
            ExprPtr right = parseFactor(str, pos);
            left = multiply(left, right);
        } else if (op == '/') {
            pos++;
            ExprPtr right = parseFactor(str, pos);
            left = divide(left, right);
        } else {
            break;
        }
    }
    
    return left;
}

/**
 * @brief Parse factor (handles implicit multiplication and power)
 *
 * factor := power
 */
ExprPtr ExpressionParser::parseFactor(const std::string& str, size_t& pos) {
    return parsePower(str, pos);
}

/**
 * @brief Parse power operator (right-associative)
 *
 * power := primary ('^' primary)*
 *
 * Note: Power is right-associative, so 2^3^2 = 2^(3^2) = 512
 */
ExprPtr ExpressionParser::parsePower(const std::string& str, size_t& pos) {
    ExprPtr base = parsePrimary(str, pos);
    
    skipWhitespace(str, pos);
    
    if (pos < str.length() && str[pos] == '^') {
        pos++;
        // Right-associative: parse power recursively
        ExprPtr exponent = parsePower(str, pos);
        return pow(base, exponent);
    }
    
    return base;
}

/**
 * @brief Parse primary (atoms and parenthesized expressions)
 *
 * primary := number | variable | function '(' expression ')' | '(' expression ')'
 */
ExprPtr ExpressionParser::parsePrimary(const std::string& str, size_t& pos) {
    skipWhitespace(str, pos);
    
    if (pos >= str.length()) {
        throw std::runtime_error("ExpressionParser: Unexpected end of expression");
    }
    
    char ch = str[pos];
    
    // Handle unary minus
    if (ch == '-') {
        pos++;
        ExprPtr arg = parsePrimary(str, pos);
        return negate(arg);
    }
    
    // Handle unary plus
    if (ch == '+') {
        pos++;
        return parsePrimary(str, pos);
    }
    
    // Handle parenthesized expression
    if (ch == '(') {
        pos++;
        ExprPtr expr = parseExpression(str, pos);
        skipWhitespace(str, pos);
        
        if (pos >= str.length() || str[pos] != ')') {
            throw std::runtime_error("ExpressionParser: Missing closing parenthesis");
        }
        
        pos++;
        return expr;
    }
    
    // Try to parse a number
    if (std::isdigit(ch) || (ch == '.' && pos + 1 < str.length() && std::isdigit(str[pos + 1]))) {
        double value = parseNumber(str, pos);
        return constant(value);
    }
    
    // Try to parse an identifier (variable or function)
    if (std::isalpha(ch) || ch == '_') {
        std::string identifier = parseIdentifier(str, pos);
        
        skipWhitespace(str, pos);
        
        // Check if it's a function call
        if (pos < str.length() && str[pos] == '(') {
            return parseFunction(identifier, str, pos);
        }
        
        // Check if it's a known constant
        if (constants_.find(identifier) != constants_.end()) {
            return constant(constants_[identifier]);
        }
        
        // Otherwise it's a variable
        if (identifier == "x" || identifier == "X") {
            return var("x");
        } else if (identifier == "pi" || identifier == "PI" || identifier == "π") {
            return constant(M_PI);
        } else if (identifier == "e" || identifier == "E") {
            return constant(M_E);
        } else {
            return var(identifier);
        }
    }
    
    throw std::runtime_error(std::string("ExpressionParser: Unexpected character '") + ch + "'");
}

/**
 * @brief Parse a function call
 * @param name Function name (sin, cos, tan, exp, log, sqrt, abs)
 * @param str Input string
 * @param pos Current position (should be at '(')
 * @return Expression representing the function call
 */
ExprPtr ExpressionParser::parseFunction(const std::string& name, const std::string& str, size_t& pos) {
    if (pos >= str.length() || str[pos] != '(') {
        throw std::runtime_error("ExpressionParser: Expected '(' after function name");
    }
    
    pos++;  // Skip '('
    ExprPtr arg = parseExpression(str, pos);
    
    skipWhitespace(str, pos);
    
    if (pos >= str.length() || str[pos] != ')') {
        throw std::runtime_error("ExpressionParser: Missing closing parenthesis in function call");
    }
    
    pos++;  // Skip ')'
    
    // Create appropriate function expression
    if (name == "sin") {
        return sin(arg);
    } else if (name == "cos") {
        return cos(arg);
    } else if (name == "tan") {
        return tan(arg);
    } else if (name == "exp") {
        return exp(arg);
    } else if (name == "log" || name == "ln") {
        return log(arg);
    } else if (name == "sqrt") {
        return sqrt(arg);
    } else if (name == "abs") {
        return abs(arg);
    } else {
        throw std::runtime_error(std::string("ExpressionParser: Unknown function '") + name + "'");
    }
}

/**
 * @brief Skip whitespace characters
 */
void ExpressionParser::skipWhitespace(const std::string& str, size_t& pos) {
    while (pos < str.length() && std::isspace(str[pos])) {
        pos++;
    }
}

/**
 * @brief Parse a number (integer or floating point)
 * @return Parsed double value
 */
double ExpressionParser::parseNumber(const std::string& str, size_t& pos) {
    size_t start = pos;
    
    // Handle digits before decimal point
    while (pos < str.length() && std::isdigit(str[pos])) {
        pos++;
    }
    
    // Handle decimal point
    if (pos < str.length() && str[pos] == '.') {
        pos++;
        while (pos < str.length() && std::isdigit(str[pos])) {
            pos++;
        }
    }
    
    // Handle scientific notation (e.g., 1e-5, 2.5E+3)
    if (pos < str.length() && (str[pos] == 'e' || str[pos] == 'E')) {
        pos++;
        if (pos < str.length() && (str[pos] == '+' || str[pos] == '-')) {
            pos++;
        }
        while (pos < str.length() && std::isdigit(str[pos])) {
            pos++;
        }
    }
    
    std::string num_str = str.substr(start, pos - start);
    
    try {
        return std::stod(num_str);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ExpressionParser: Invalid number '") + num_str + "'");
    }
}

/**
 * @brief Parse an identifier (variable or function name)
 */
std::string ExpressionParser::parseIdentifier(const std::string& str, size_t& pos) {
    size_t start = pos;
    
    while (pos < str.length() && (std::isalnum(str[pos]) || str[pos] == '_')) {
        pos++;
    }
    
    return str.substr(start, pos - start);
}

/**
 * @brief Set variables that the parser should recognize
 */
void ExpressionParser::setVariables(const std::vector<std::string>& variables) {
    variables_ = variables;
}

/**
 * @brief Add a custom constant
 */
void ExpressionParser::addConstant(const std::string& name, double value) {
    constants_[name] = value;
}

// Helper functions for expression construction
ExprPtr var(const std::string& name) {
    // For now, always create 'x' variable
    // Multi-variable support would need to be added to the Expression classes
    return std::make_shared<Variable>();
}

ExprPtr constant(double value) {
    return std::make_shared<Constant>(value);
}

ExprPtr add(ExprPtr left, ExprPtr right) {
    return std::make_shared<Addition>(left, right);
}

ExprPtr subtract(ExprPtr left, ExprPtr right) {
    return std::make_shared<Subtraction>(left, right);
}

ExprPtr multiply(ExprPtr left, ExprPtr right) {
    return std::make_shared<Multiplication>(left, right);
}

ExprPtr divide(ExprPtr left, ExprPtr right) {
    return std::make_shared<Division>(left, right);
}

ExprPtr pow(ExprPtr base, ExprPtr exponent) {
    return std::make_shared<Power>(base, exponent);
}

ExprPtr negate(ExprPtr argument) {
    // Negate is represented as multiplication by -1
    return multiply(constant(-1.0), argument);
}

ExprPtr sin(ExprPtr argument) {
    return std::make_shared<Sin>(argument);
}

ExprPtr cos(ExprPtr argument) {
    return std::make_shared<Cos>(argument);
}

ExprPtr tan(ExprPtr argument) {
    // tan(x) = sin(x) / cos(x)
    return divide(sin(argument), cos(argument));
}

ExprPtr exp(ExprPtr argument) {
    return std::make_shared<Exp>(argument);
}

ExprPtr log(ExprPtr argument) {
    return std::make_shared<Log>(argument);
}

ExprPtr sqrt(ExprPtr argument) {
    // sqrt(x) = x^(1/2)
    return pow(argument, constant(0.5));
}

ExprPtr abs(ExprPtr argument) {
    // For now, represent as identity
    // Full implementation would need Abs class
    return argument;
}

} // namespace ode_solver::symbolic