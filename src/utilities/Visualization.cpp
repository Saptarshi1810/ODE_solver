/**
 * @file Visualization.cpp
 * @brief Implementation of visualization utilities for ODE solutions
 * @author Your Name
 * @version 1.0.0
 *
 * This module provides visualization capabilities through Gnuplot integration,
 * enabling the creation of publication-quality plots. Features include:
 * - Solution trajectory plots
 * - Phase portraits (2D)
 * - 3D trajectory visualization
 * - Convergence history plots
 * - CSV data plotting
 * - Customizable styles and output formats
 */

#include "ode_solver/Utilities.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace ode_solver::utils {

/**
 * @brief Check if Gnuplot is available on the system
 *
 * @return true if gnuplot command is available
 */
bool GnuplotInterface::isAvailable() {
    try {
        int result = std::system("gnuplot --version > /dev/null 2>&1");
        return result == 0;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Plot ODE solution components
 *
 * Creates a plot showing solution components as functions of time.
 *
 * @param solution ODESolution containing time and solution data
 * @param component_names Names for each component (auto-generated if empty)
 * @param config Plot configuration
 */
void GnuplotInterface::plotSolution(const ODESolution& solution,
                                   const std::vector<std::string>& component_names,
                                   const PlotConfig& config) {
    validateGnuplotAvailable();
    
    // Validate input
    if (solution.t.empty() || solution.y.empty()) {
        throw std::runtime_error("GnuplotInterface: Empty solution data");
    }
    
    // Create temporary data file
    std::string data_file = "/tmp/ode_solution_plot.dat";
    std::ofstream data_out(data_file);
    
    if (!data_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create temporary data file");
    }
    
    // Write data to file
    for (size_t i = 0; i < solution.t.size(); ++i) {
        data_out << std::scientific << std::setprecision(10) << solution.t[i];
        
        for (size_t j = 0; j < solution.y[i].size(); ++j) {
            data_out << " " << std::scientific << std::setprecision(10)
                    << solution.y[i][j];
        }
        data_out << "\n";
    }
    data_out.close();
    
    // Create Gnuplot script
    std::string script_file = "/tmp/ode_solution_plot.gp";
    std::ofstream script_out(script_file);
    
    if (!script_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create Gnuplot script");
    }
    
    // Configure terminal
    script_out << "set terminal " << config.terminal;
    if (!config.output_file.empty()) {
        script_out << " size " << config.width << "," << config.height;
        script_out << "\nset output '" << config.output_file << "'";
    }
    script_out << "\n";
    
    // Set labels and title
    script_out << "set title '" << config.title << "'\n";
    script_out << "set xlabel '" << config.xlabel << "'\n";
    script_out << "set ylabel '" << config.ylabel << "'\n";
    
    // Set grid if requested
    if (config.grid) {
        script_out << "set grid\n";
    }
    
    // Set legend if requested
    if (config.legend) {
        script_out << "set legend top left\n";
    } else {
        script_out << "set nolegend\n";
    }
    
    // Plot commands
    script_out << "plot ";
    
    for (size_t i = 0; i < solution.y[0].size(); ++i) {
        if (i > 0) script_out << ", ";
        
        std::string component_label = (i < component_names.size())
            ? component_names[i]
            : ("y" + std::to_string(i + 1));
        
        script_out << "'" << data_file << "' using 1:" << (i + 2)
                  << " with " << config.style
                  << " title '" << component_label << "'";
    }
    script_out << "\n";
    
    script_out.close();
    
    // Execute Gnuplot
    std::string command = "gnuplot '" + script_file + "'";
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot execution failed");
    }
}

/**
 * @brief Plot data from CSV file
 *
 * @param csv_file Path to CSV file
 * @param x_column Index of column for x-axis
 * @param y_columns Indices of columns for y-axis
 * @param config Plot configuration
 */
void GnuplotInterface::plotCSV(const std::string& csv_file,
                              int x_column,
                              const std::vector<int>& y_columns,
                              const PlotConfig& config) {
    validateGnuplotAvailable();
    
    // Validate input
    if (y_columns.empty()) {
        throw std::runtime_error(
            "GnuplotInterface: No y columns specified");
    }
    
    // Create Gnuplot script
    std::string script_file = "/tmp/ode_csv_plot.gp";
    std::ofstream script_out(script_file);
    
    if (!script_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create Gnuplot script");
    }
    
    // Configure terminal
    script_out << "set terminal " << config.terminal;
    if (!config.output_file.empty()) {
        script_out << " size " << config.width << "," << config.height;
        script_out << "\nset output '" << config.output_file << "'";
    }
    script_out << "\n";
    
    // Set labels and title
    script_out << "set title '" << config.title << "'\n";
    script_out << "set xlabel '" << config.xlabel << "'\n";
    script_out << "set ylabel '" << config.ylabel << "'\n";
    
    // Set grid and legend
    if (config.grid) {
        script_out << "set grid\n";
    }
    if (config.legend) {
        script_out << "set legend top left\n";
    }
    
    // Plot commands
    script_out << "plot ";
    
    for (size_t i = 0; i < y_columns.size(); ++i) {
        if (i > 0) script_out << ", ";
        
        script_out << "'" << csv_file << "' using " << (x_column + 1)
                  << ":" << (y_columns[i] + 1)
                  << " with " << config.style
                  << " title 'Column " << y_columns[i] << "'";
    }
    script_out << "\n";
    
    script_out.close();
    
    // Execute Gnuplot
    std::string command = "gnuplot '" + script_file + "'";
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot execution failed");
    }
}

/**
 * @brief Plot phase portrait (2D phase space)
 *
 * Creates a phase portrait showing relationship between two components.
 *
 * @param solution ODE solution
 * @param component1 Index of first component for phase space
 * @param component2 Index of second component for phase space
 * @param config Plot configuration
 */
void GnuplotInterface::plotPhasePortrait(const ODESolution& solution,
                                        int component1,
                                        int component2,
                                        const PlotConfig& config) {
    validateGnuplotAvailable();
    
    // Validate components
    if (solution.y.empty() || solution.y[0].empty()) {
        throw std::runtime_error("GnuplotInterface: Empty solution data");
    }
    
    if (component1 < 0 || component2 < 0 ||
        component1 >= static_cast<int>(solution.y[0].size()) ||
        component2 >= static_cast<int>(solution.y[0].size())) {
        throw std::runtime_error(
            "GnuplotInterface: Invalid component indices");
    }
    
    // Create temporary data file
    std::string data_file = "/tmp/ode_phase_portrait.dat";
    std::ofstream data_out(data_file);
    
    if (!data_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create temporary data file");
    }
    
    // Write phase space data
    for (size_t i = 0; i < solution.y.size(); ++i) {
        data_out << std::scientific << std::setprecision(10)
                << solution.y[i][component1] << " "
                << solution.y[i][component2] << "\n";
    }
    data_out.close();
    
    // Create Gnuplot script
    std::string script_file = "/tmp/ode_phase_portrait.gp";
    std::ofstream script_out(script_file);
    
    if (!script_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create Gnuplot script");
    }
    
    // Configure terminal
    script_out << "set terminal " << config.terminal;
    if (!config.output_file.empty()) {
        script_out << " size " << config.width << "," << config.height;
        script_out << "\nset output '" << config.output_file << "'";
    }
    script_out << "\n";
    
    // Set labels and title
    script_out << "set title '" << config.title << "'\n";
    script_out << "set xlabel 'y" << (component1 + 1) << "'\n";
    script_out << "set ylabel 'y" << (component2 + 1) << "'\n";
    
    if (config.grid) {
        script_out << "set grid\n";
    }
    
    // Plot phase portrait with arrows showing direction
    script_out << "plot '" << data_file << "' with " << config.style
              << " title 'Phase Portrait'\n";
    
    script_out.close();
    
    // Execute Gnuplot
    std::string command = "gnuplot '" + script_file + "'";
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot execution failed");
    }
}

/**
 * @brief Plot 3D trajectory
 *
 * Creates a 3D visualization of solution trajectory.
 *
 * @param solution ODE solution
 * @param comp1 First component for 3D plot
 * @param comp2 Second component for 3D plot
 * @param comp3 Third component for 3D plot
 * @param config Plot configuration
 */
void GnuplotInterface::plot3DTrajectory(const ODESolution& solution,
                                       int comp1, int comp2, int comp3,
                                       const PlotConfig& config) {
    validateGnuplotAvailable();
    
    // Validate components
    if (solution.y.empty() || solution.y[0].empty()) {
        throw std::runtime_error("GnuplotInterface: Empty solution data");
    }
    
    if (comp1 < 0 || comp2 < 0 || comp3 < 0 ||
        comp1 >= static_cast<int>(solution.y[0].size()) ||
        comp2 >= static_cast<int>(solution.y[0].size()) ||
        comp3 >= static_cast<int>(solution.y[0].size())) {
        throw std::runtime_error(
            "GnuplotInterface: Invalid component indices");
    }
    
    // Create temporary data file
    std::string data_file = "/tmp/ode_3d_trajectory.dat";
    std::ofstream data_out(data_file);
    
    if (!data_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create temporary data file");
    }
    
    // Write 3D data
    for (size_t i = 0; i < solution.y.size(); ++i) {
        data_out << std::scientific << std::setprecision(10)
                << solution.y[i][comp1] << " "
                << solution.y[i][comp2] << " "
                << solution.y[i][comp3] << "\n";
    }
    data_out.close();
    
    // Create Gnuplot script
    std::string script_file = "/tmp/ode_3d_trajectory.gp";
    std::ofstream script_out(script_file);
    
    if (!script_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create Gnuplot script");
    }
    
    // Configure terminal for 3D
    script_out << "set terminal " << config.terminal;
    if (!config.output_file.empty()) {
        script_out << " size " << config.width << "," << config.height;
        script_out << "\nset output '" << config.output_file << "'";
    }
    script_out << "\n";
    
    // Set 3D view
    script_out << "set view 60, 120\n";
    
    // Set labels and title
    script_out << "set title '" << config.title << "'\n";
    script_out << "set xlabel 'y" << (comp1 + 1) << "'\n";
    script_out << "set ylabel 'y" << (comp2 + 1) << "'\n";
    script_out << "set zlabel 'y" << (comp3 + 1) << "'\n";
    
    if (config.grid) {
        script_out << "set grid\n";
    }
    
    // Plot 3D trajectory
    script_out << "splot '" << data_file << "' with " << config.style
              << " title '3D Trajectory'\n";
    
    script_out.close();
    
    // Execute Gnuplot
    std::string command = "gnuplot '" + script_file + "'";
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot execution failed");
    }
}

/**
 * @brief Plot convergence history
 *
 * Visualizes error or residual history from iterative solver.
 *
 * @param error_history Vector of error values at each iteration
 * @param config Plot configuration
 */
void GnuplotInterface::plotConvergence(const std::vector<double>& error_history,
                                      const PlotConfig& config) {
    validateGnuplotAvailable();
    
    if (error_history.empty()) {
        throw std::runtime_error(
            "GnuplotInterface: Empty error history");
    }
    
    // Create temporary data file
    std::string data_file = "/tmp/ode_convergence.dat";
    std::ofstream data_out(data_file);
    
    if (!data_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create temporary data file");
    }
    
    // Write convergence data
    for (size_t i = 0; i < error_history.size(); ++i) {
        data_out << i << " " << std::scientific << std::setprecision(10)
                << error_history[i] << "\n";
    }
    data_out.close();
    
    // Create Gnuplot script
    std::string script_file = "/tmp/ode_convergence.gp";
    std::ofstream script_out(script_file);
    
    if (!script_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create Gnuplot script");
    }
    
    // Configure terminal
    script_out << "set terminal " << config.terminal;
    if (!config.output_file.empty()) {
        script_out << " size " << config.width << "," << config.height;
        script_out << "\nset output '" << config.output_file << "'";
    }
    script_out << "\n";
    
    // Set logarithmic scale for y-axis (typical for convergence plots)
    script_out << "set logscale y\n";
    
    // Set labels and title
    script_out << "set title '" << config.title << "'\n";
    script_out << "set xlabel 'Iteration'\n";
    script_out << "set ylabel 'Error (log scale)'\n";
    
    if (config.grid) {
        script_out << "set grid\n";
    }
    
    // Plot convergence
    script_out << "plot '" << data_file << "' with " << config.style
              << " title 'Convergence'\n";
    
    script_out.close();
    
    // Execute Gnuplot
    std::string command = "gnuplot '" + script_file + "'";
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot execution failed");
    }
}

/**
 * @brief Execute a Gnuplot script file
 *
 * @param script Gnuplot script content
 * @return Command output
 */
std::string GnuplotInterface::executeScript(const std::string& script) {
    validateGnuplotAvailable();
    
    // Create temporary script file
    std::string script_file = "/tmp/ode_custom_script.gp";
    std::ofstream script_out(script_file);
    
    if (!script_out.is_open()) {
        throw std::runtime_error(
            "GnuplotInterface: Cannot create script file");
    }
    
    script_out << script;
    script_out.close();
    
    // Execute Gnuplot
    std::string command = "gnuplot '" + script_file + "'";
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot execution failed");
    }
    
    return "Script executed successfully";
}

/**
 * @brief Validate that Gnuplot is available
 *
 * @throws std::runtime_error if Gnuplot is not available
 */
void GnuplotInterface::validateGnuplotAvailable() {
    if (!isAvailable()) {
        throw std::runtime_error(
            "GnuplotInterface: Gnuplot is not installed or not in PATH");
    }
}

/**
 * @brief Build a plot command based on configuration
 *
 * @param config Plot configuration
 * @return Command string
 */
std::string GnuplotInterface::buildPlotCommand(const PlotConfig& config) {
    std::ostringstream oss;
    
    oss << "set terminal " << config.terminal;
    if (!config.output_file.empty()) {
        oss << " size " << config.width << "," << config.height;
    }
    
    oss << "\nset title '" << config.title << "'";
    oss << "\nset xlabel '" << config.xlabel << "'";
    oss << "\nset ylabel '" << config.ylabel << "'";
    
    if (config.grid) {
        oss << "\nset grid";
    }
    
    if (config.legend) {
        oss << "\nset legend top left";
    } else {
        oss << "\nset nolegend";
    }
    
    return oss.str();
}

} // namespace ode_solver::utils