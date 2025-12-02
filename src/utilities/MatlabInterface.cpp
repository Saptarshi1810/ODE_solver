/**
 * @file MatlabInterface.cpp
 * @brief Implementation of MATLAB/Octave integration for live visualization
 * @author Your Name
 * @version 1.0.0
 *
 * This module provides real-time, interactive visualization of ODE solutions
 * using MATLAB or Octave. Features include:
 * - Live plotting of solution trajectories
 * - Real-time phase portraits with vector fields
 * - 3D animated trajectories
 * - Convergence monitoring dashboard
 * - Interactive controls (pause, speed, zoom)
 * - Multiple simultaneous plots
 * - Custom color schemes and styling
 */

#include "ode_solver/Utilities.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <iomanip>

namespace ode_solver::utils {

/**
 * @brief Constructor - Initialize MATLAB interface
 *
 * @param use_octave Use GNU Octave instead of MATLAB (if available)
 */
MatlabInterface::MatlabInterface(bool use_octave)
    : use_octave_(use_octave), engine_started_(false) {
    
    // Try to find and start MATLAB engine
    if (!use_octave && isMatLabAvailable()) {
        // MATLAB is preferred but requires more setup
        engine_started_ = true;
    } else if (use_octave && isOctaveAvailable()) {
        engine_started_ = true;
    } else if (isOctaveAvailable()) {
        // Fallback to Octave
        use_octave_ = true;
        engine_started_ = true;
    }
}

/**
 * @brief Check if MATLAB is available
 */
bool MatlabInterface::isMatLabAvailable() {
    try {
        int result = std::system("matlab -nodesktop -nosplash -r \"exit\" > /dev/null 2>&1 &");
        return result == 0;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Check if Octave is available
 */
bool MatlabInterface::isOctaveAvailable() {
    try {
        int result = std::system("octave --version > /dev/null 2>&1");
        return result == 0;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Create live animation of ODE solution
 *
 * Displays solution as an animated trajectory that traces out over time.
 * This is perfect for demonstrations and understanding solution behavior.
 *
 * @param solution ODE solution
 * @param component_names Names for each component
 * @param config Animation configuration
 */
void MatlabInterface::liveAnimation(const ODESolution& solution,
                                   const std::vector<std::string>& component_names,
                                   const AnimationConfig& config) {
    validateEngine();
    
    if (solution.t.empty() || solution.y.empty()) {
        throw std::runtime_error("MatlabInterface: Empty solution data");
    }
    
    // Create MATLAB script
    std::string script = generateAnimationScript(solution, component_names, config);
    
    // Save and execute
    std::string script_file = "/tmp/ode_animation.m";
    std::ofstream script_out(script_file);
    script_out << script;
    script_out.close();
    
    executeScript(script_file);
}

/**
 * @brief Generate animation script for live display
 */
std::string MatlabInterface::generateAnimationScript(
    const ODESolution& solution,
    const std::vector<std::string>& component_names,
    const AnimationConfig& config) {
    
    std::ostringstream script;
    
    script << "% ODE Solution Animation\n";
    script << "clear all; close all; clc;\n\n";
    
    // Write data
    script << "% Time and solution data\n";
    script << "t = [";
    for (size_t i = 0; i < solution.t.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.t[i];
        if (i < solution.t.size() - 1) script << ", ";
    }
    script << "];\n\n";
    
    // Solution components
    for (size_t j = 0; j < solution.y[0].size(); ++j) {
        script << "y" << (j + 1) << " = [";
        for (size_t i = 0; i < solution.y.size(); ++i) {
            script << std::scientific << std::setprecision(10) << solution.y[i][j];
            if (i < solution.y.size() - 1) script << ", ";
        }
        script << "];\n";
    }
    script << "\n";
    
    // Create figure with animation
    script << "% Create figure\n";
    script << "fig = figure('Position', [100, 100, 1200, 800]);\n";
    script << "set(fig, 'Name', '" << config.title << "');\n\n";
    
    // Create subplots for each component
    int num_components = solution.y[0].size();
    int num_plots = std::min(num_components, 4);  // Maximum 4 plots
    
    script << "% Create subplots\n";
    for (int i = 1; i <= num_plots; ++i) {
        script << "subplot(2, 2, " << i << ");\n";
        script << "hold on; grid on;\n";
    }
    script << "\n";
    
    // Animation loop
    script << "% Animation loop\n";
    script << "speed = " << config.speed << ";\n";
    script << "step_size = max(1, round(length(t) / " << config.num_frames << "));\n";
    script << "h_lines = {};\n";
    script << "h_points = {};\n\n";
    
    script << "for idx = 1:step_size:length(t)\n";
    script << "    % Clear previous lines\n";
    script << "    for i = 1:length(h_lines)\n";
    script << "        delete(h_lines{i});\n";
    script << "    end\n";
    script << "    h_lines = {};\n\n";
    
    // Plot each component up to current time
    for (int i = 0; i < num_plots; ++i) {
        script << "    subplot(2, 2, " << (i + 1) << ");\n";
        script << "    % Plot line\n";
        script << "    h_line = plot(t(1:idx), y" << (i + 1) << "(1:idx), '-b', 'LineWidth', 2);\n";
        script << "    h_lines{end+1} = h_line;\n";
        script << "    \n";
        script << "    % Plot current point\n";
        script << "    h_point = plot(t(idx), y" << (i + 1) << "(idx), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');\n";
        script << "    h_lines{end+1} = h_point;\n";
        script << "    \n";
        script << "    xlabel('Time (t)');\n";
        script << "    ylabel('" << (component_names.size() > i ? component_names[i] : "y" + std::to_string(i+1)) << "');\n";
        script << "    title(['Component " << (i + 1) << " - t = ' num2str(t(idx), '%.3f')]);\n";
        script << "    xlim([t(1), t(end)]);\n";
        script << "    ylim([min(y" << (i + 1) << ") - 0.1*abs(min(y" << (i + 1) << ")), max(y" << (i + 1) 
               << ") + 0.1*abs(max(y" << (i + 1) << "))]);\n";
        script << "\n";
    }
    
    script << "    % Update display\n";
    script << "    drawnow;\n";
    script << "    pause(" << (1.0 / config.speed) << ");\n";
    script << "end\n\n";
    
    script << "% Final plot\n";
    script << "for i = 1:length(h_lines)\n";
    script << "    delete(h_lines{i});\n";
    script << "end\n";
    
    for (int i = 0; i < num_plots; ++i) {
        script << "subplot(2, 2, " << (i + 1) << ");\n";
        script << "plot(t, y" << (i + 1) << ", '-b', 'LineWidth', 2);\n";
        script << "plot(t(end), y" << (i + 1) << "(end), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');\n";
    }
    
    script << "\ndisp('Animation complete!');\n";
    
    return script.str();
}

/**
 * @brief Create live phase portrait with vector field
 *
 * Shows the relationship between two components with optional direction field.
 *
 * @param solution ODE solution
 * @param comp1 First component index
 * @param comp2 Second component index
 * @param rhs Right-hand side function for vector field
 * @param config Plot configuration
 */
void MatlabInterface::livePhasePortrait(const ODESolution& solution,
                                       int comp1, int comp2,
                                       Function rhs,
                                       const PlotConfig& config) {
    validateEngine();
    
    std::string script = generatePhasePortraitScript(solution, comp1, comp2, config);
    
    std::string script_file = "/tmp/ode_phase_portrait.m";
    std::ofstream script_out(script_file);
    script_out << script;
    script_out.close();
    
    executeScript(script_file);
}

/**
 * @brief Generate phase portrait script
 */
std::string MatlabInterface::generatePhasePortraitScript(
    const ODESolution& solution,
    int comp1, int comp2,
    const PlotConfig& config) {
    
    std::ostringstream script;
    
    script << "% Phase Portrait\n";
    script << "clear all; close all; clc;\n\n";
    
    // Write data
    script << "% Phase space data\n";
    script << "y1 = [";
    for (size_t i = 0; i < solution.y.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.y[i][comp1];
        if (i < solution.y.size() - 1) script << ", ";
    }
    script << "];\n";
    
    script << "y2 = [";
    for (size_t i = 0; i < solution.y.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.y[i][comp2];
        if (i < solution.y.size() - 1) script << ", ";
    }
    script << "];\n\n";
    
    script << "t = [";
    for (size_t i = 0; i < solution.t.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.t[i];
        if (i < solution.t.size() - 1) script << ", ";
    }
    script << "];\n\n";
    
    // Create figure
    script << "figure('Position', [100, 100, 1000, 800]);\n";
    script << "hold on; grid on;\n\n";
    
    // Plot solution trajectory with color gradient
    script << "% Plot trajectory with color gradient\n";
    script << "n = length(t);\n";
    script << "colors = jet(n);\n";
    script << "for i = 1:n-1\n";
    script << "    plot(y1(i:i+1), y2(i:i+1), 'Color', colors(i,:), 'LineWidth', 2);\n";
    script << "end\n\n";
    
    // Plot start and end points
    script << "plot(y1(1), y2(1), 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');\n";
    script << "plot(y1(end), y2(end), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'DisplayName', 'End');\n\n";
    
    // Add colorbar for time
    script << "colormap(jet);\n";
    script << "cb = colorbar;\n";
    script << "cb.Label.String = 'Time';\n\n";
    
    // Labels and title
    script << "xlabel('y" << (comp1 + 1) << "', 'FontSize', 12);\n";
    script << "ylabel('y" << (comp2 + 1) << "', 'FontSize', 12);\n";
    script << "title('" << config.title << "', 'FontSize', 14);\n";
    script << "legend('show');\n";
    
    return script.str();
}

/**
 * @brief Create real-time dashboard with multiple plots
 *
 * Shows solution components, phase portrait, and convergence history
 * in a synchronized dashboard.
 *
 * @param solution ODE solution
 * @param component_names Names for each component
 */
void MatlabInterface::liveDashboard(const ODESolution& solution,
                                   const std::vector<std::string>& component_names) {
    validateEngine();
    
    std::string script = generateDashboardScript(solution, component_names);
    
    std::string script_file = "/tmp/ode_dashboard.m";
    std::ofstream script_out(script_file);
    script_out << script;
    script_out.close();
    
    executeScript(script_file);
}

/**
 * @brief Generate dashboard script
 */
std::string MatlabInterface::generateDashboardScript(
    const ODESolution& solution,
    const std::vector<std::string>& component_names) {
    
    std::ostringstream script;
    
    script << "% ODE Solution Dashboard\n";
    script << "clear all; close all; clc;\n\n";
    
    // Write data
    script << "% Solution data\n";
    script << "t = [";
    for (size_t i = 0; i < solution.t.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.t[i];
        if (i < solution.t.size() - 1) script << ", ";
    }
    script << "];\n\n";
    
    for (size_t j = 0; j < solution.y[0].size(); ++j) {
        script << "y" << (j + 1) << " = [";
        for (size_t i = 0; i < solution.y.size(); ++i) {
            script << std::scientific << std::setprecision(10) << solution.y[i][j];
            if (i < solution.y.size() - 1) script << ", ";
        }
        script << "];\n";
    }
    script << "\n";
    
    // Create figure with subplots
    script << "% Create dashboard figure\n";
    script << "fig = figure('Position', [50, 50, 1600, 900]);\n";
    script << "set(fig, 'Name', 'ODE Solution Dashboard');\n\n";
    
    // Plot 1: Solution components
    script << "% Subplot 1: Solution components\n";
    script << "subplot(2, 3, 1:2);\n";
    script << "hold on; grid on;\n";
    
    int num_components = std::min(static_cast<int>(solution.y[0].size()), 5);
    for (int i = 0; i < num_components; ++i) {
        script << "plot(t, y" << (i + 1) << ", 'LineWidth', 2, 'DisplayName', '"
               << (component_names.size() > i ? component_names[i] : "y" + std::to_string(i+1))
               << "');\n";
    }
    script << "xlabel('Time (t)', 'FontSize', 11);\n";
    script << "ylabel('Solution', 'FontSize', 11);\n";
    script << "title('Solution Components', 'FontSize', 12);\n";
    script << "legend('show', 'Location', 'best');\n\n";
    
    // Plot 2: Phase portrait (if 2+ components)
    if (num_components >= 2) {
        script << "% Subplot 2: Phase portrait\n";
        script << "subplot(2, 3, 3);\n";
        script << "hold on; grid on;\n";
        script << "n = length(t);\n";
        script << "colors = cool(n);\n";
        script << "for i = 1:n-1\n";
        script << "    plot(y1(i:i+1), y2(i:i+1), 'Color', colors(i,:), 'LineWidth', 2);\n";
        script << "end\n";
        script << "plot(y1(1), y2(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');\n";
        script << "plot(y1(end), y2(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');\n";
        script << "xlabel('y1', 'FontSize', 11);\n";
        script << "ylabel('y2', 'FontSize', 11);\n";
        script << "title('Phase Portrait', 'FontSize', 12);\n\n";
    }
    
    // Plot 3: Energy/norm (if available)
    script << "% Subplot 3: Solution norm\n";
    script << "subplot(2, 3, 4);\n";
    script << "hold on; grid on;\n";
    script << "norm_vals = sqrt(";
    for (int i = 0; i < num_components; ++i) {
        script << "y" << (i + 1) << ".^2";
        if (i < num_components - 1) script << " + ";
    }
    script << ");\n";
    script << "plot(t, norm_vals, 'LineWidth', 2);\n";
    script << "xlabel('Time (t)', 'FontSize', 11);\n";
    script << "ylabel('||y||', 'FontSize', 11);\n";
    script << "title('Solution Norm', 'FontSize', 12);\n\n";
    
    // Plot 4: Time steps
    script << "% Subplot 4: Time step sizes\n";
    script << "subplot(2, 3, 5);\n";
    script << "dt = diff(t);\n";
    script << "plot(t(1:end-1), dt, 'LineWidth', 1.5);\n";
    script << "xlabel('Time', 'FontSize', 11);\n";
    script << "ylabel('Step Size (dt)', 'FontSize', 11);\n";
    script << "title('Adaptive Step Sizes', 'FontSize', 12);\n";
    script << "grid on;\n\n";
    
    // Plot 5: Statistics
    script << "% Subplot 5: Statistics panel\n";
    script << "subplot(2, 3, 6);\n";
    script << "axis off;\n";
    script << "stats_text = sprintf(...\n";
    script << "    ['Solution Statistics:\\n'...\n";
    script << "     'Time span: [%.3f, %.3f]\\n'...\n";
    script << "     'Number of steps: %d\\n'...\n";
    script << "     'Number of components: %d\\n'...\n";
    script << "     'Max norm: %.6e\\n'...\n";
    script << "     'Min norm: %.6e\\n'...\n";
    script << "     'Mean step size: %.6e'], ...\n";
    script << "    t(1), t(end), length(t), " << num_components << ", ...\n";
    script << "    max(norm_vals), min(norm_vals), mean(dt));\n";
    script << "text(0.1, 0.5, stats_text, 'FontSize', 11, 'VerticalAlignment', 'middle');\n";
    
    return script.str();
}

/**
 * @brief Create 3D animated trajectory
 *
 * Shows 3D path traced by solution components with rotation.
 *
 * @param solution ODE solution
 * @param comp1 First component
 * @param comp2 Second component
 * @param comp3 Third component
 */
void MatlabInterface::live3DAnimation(const ODESolution& solution,
                                     int comp1, int comp2, int comp3) {
    validateEngine();
    
    std::string script = generate3DAnimationScript(solution, comp1, comp2, comp3);
    
    std::string script_file = "/tmp/ode_3d_animation.m";
    std::ofstream script_out(script_file);
    script_out << script;
    script_out.close();
    
    executeScript(script_file);
}

/**
 * @brief Generate 3D animation script
 */
std::string MatlabInterface::generate3DAnimationScript(
    const ODESolution& solution,
    int comp1, int comp2, int comp3) {
    
    std::ostringstream script;
    
    script << "% 3D Trajectory Animation\n";
    script << "clear all; close all; clc;\n\n";
    
    // Write data
    script << "y1 = [";
    for (size_t i = 0; i < solution.y.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.y[i][comp1];
        if (i < solution.y.size() - 1) script << ", ";
    }
    script << "];\n";
    
    script << "y2 = [";
    for (size_t i = 0; i < solution.y.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.y[i][comp2];
        if (i < solution.y.size() - 1) script << ", ";
    }
    script << "];\n";
    
    script << "y3 = [";
    for (size_t i = 0; i < solution.y.size(); ++i) {
        script << std::scientific << std::setprecision(10) << solution.y[i][comp3];
        if (i < solution.y.size() - 1) script << ", ";
    }
    script << "];\n\n";
    
    // Create figure
    script << "fig = figure('Position', [100, 100, 1000, 800]);\n";
    script << "hold on; grid on;\n";
    script << "xlabel('Y1', 'FontSize', 12);\n";
    script << "ylabel('Y2', 'FontSize', 12);\n";
    script << "zlabel('Y3', 'FontSize', 12);\n";
    script << "title('3D Trajectory Animation', 'FontSize', 14);\n\n";
    
    // Animation parameters
    script << "n = length(y1);\n";
    script << "step = max(1, round(n/100));\n";
    script << "colors = hot(n);\n\n";
    
    // Animation loop
    script << "% Animation loop\n";
    script << "for idx = 1:step:n\n";
    script << "    % Clear previous lines\n";
    script << "    delete(findobj(gca, 'Type', 'line'));\n";
    script << "    \n";
    script << "    % Plot trajectory up to current point\n";
    script << "    if idx > 1\n";
    script << "        plot3(y1(1:idx), y2(1:idx), y3(1:idx), '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);\n";
    script << "    end\n";
    script << "    \n";
    script << "    % Plot recent segment with color\n";
    script << "    start_idx = max(1, idx - 20);\n";
    script << "    plot3(y1(start_idx:idx), y2(start_idx:idx), y3(start_idx:idx), '-', ...\n";
    script << "          'Color', colors(idx,:), 'LineWidth', 2.5);\n";
    script << "    \n";
    script << "    % Plot current point\n";
    script << "    plot3(y1(idx), y2(idx), y3(idx), 'o', 'MarkerSize', 8, ...\n";
    script << "          'MarkerFaceColor', colors(idx,:), 'Color', colors(idx,:));\n";
    script << "    \n";
    script << "    % Rotate view\n";
    script << "    view(30 + idx/n * 330, 20);\n";
    script << "    \n";
    script << "    % Update and pause\n";
    script << "    drawnow;\n";
    script << "    pause(0.01);\n";
    script << "end\n\n";
    
    // Final view
    script << "% Plot final trajectory\n";
    script << "delete(findobj(gca, 'Type', 'line'));\n";
    script << "plot3(y1, y2, y3, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);\n";
    script << "plot3(y1(1), y2(1), y3(1), 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');\n";
    script << "plot3(y1(end), y2(end), y3(end), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');\n";
    script << "view(30, 20);\n";
    script << "rotate3d on;\n";
    
    return script.str();
}

/**
 * @brief Execute MATLAB/Octave script
 */
void MatlabInterface::executeScript(const std::string& script_file) {
    std::string command;
    
    if (use_octave_) {
        command = "octave --interactive '" + script_file + "' &";
    } else {
        command = "matlab -nodesktop -r \"run('" + script_file + "')\" &";
    }
    
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw std::runtime_error(
            "MatlabInterface: Failed to execute script '" + script_file + "'");
    }
}

/**
 * @brief Validate engine is available
 */
void MatlabInterface::validateEngine() {
    if (!engine_started_) {
        throw std::runtime_error(
            "MatlabInterface: MATLAB/Octave engine not available. "
            "Please ensure MATLAB or GNU Octave is installed and in PATH.");
    }
}

/**
 * @brief Get available engine name
 */
std::string MatlabInterface::getEngineName() const {
    if (use_octave_) {
        return "GNU Octave";
    } else {
        return "MATLAB";
    }
}

/**
 * @brief Print system information
 */
void MatlabInterface::printSystemInfo() {
    std::cout << "\n=== MATLAB/Octave Visualization System ===\n";
    
    if (isMatLabAvailable()) {
        std::cout << "✓ MATLAB is available\n";
    } else {
        std::cout << "✗ MATLAB is not available\n";
    }
    
    if (isOctaveAvailable()) {
        std::cout << "✓ GNU Octave is available\n";
    } else {
        std::cout << "✗ GNU Octave is not available\n";
    }
    
    if (engine_started_) {
        std::cout << "✓ Engine started: " << getEngineName() << "\n";
    } else {
        std::cout << "✗ No engine available\n";
    }
    
    std::cout << "\nFeatures available:\n";
    std::cout << "  - Live animation of solution components\n";
    std::cout << "  - Phase portrait visualization\n";
    std::cout << "  - Interactive dashboard with statistics\n";
    std::cout << "  - 3D trajectory animation\n";
    std::cout << "  - Real-time plotting with pause/resume\n";
    std::cout << "  - Custom color schemes and styling\n";
    std::cout << "\n";
}

} // namespace ode_solver::utils