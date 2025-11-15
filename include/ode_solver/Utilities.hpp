

#ifndef ODE_SOLVER_UTILITIES_HPP
#define ODE_SOLVER_UTILITIES_HPP

#include "EquationSolver.hpp"
#include "ODESolvers.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace ode_solver {
namespace utils {


class CSVWriter {
public:

    struct Config {
        char delimiter = ',';          ///< Column delimiter (default: comma)
        int precision = 10;            ///< Decimal precision
        bool scientific = false;       ///< Use scientific notation
        bool write_header = true;      ///< Write column headers
        std::string comment_char = "#"; ///< Comment line prefix
    };
    

    explicit CSVWriter(const std::string& filename, const Config& config = Config());
    

    ~CSVWriter();
    
    // Disable copy, enable move
    CSVWriter(const CSVWriter&) = delete;
    CSVWriter& operator=(const CSVWriter&) = delete;
    CSVWriter(CSVWriter&&) noexcept = default;
    CSVWriter& operator=(CSVWriter&&) noexcept = default;
    
    void writeHeader(const std::vector<std::string>& headers);
    

    void writeRow(const std::vector<double>& row);
    

    void writeRows(const std::vector<std::vector<double>>& rows);
    
 
    void writeComment(const std::string& comment);

    void writeSolution(const ODESolution& solution,
                      const std::vector<std::string>& variable_names = {},
                      bool include_metadata = true);

    void flush();

    bool isOpen() const;

private:
    std::ofstream file_;
    Config config_;
    bool header_written_;
    
    std::string formatValue(double value) const;
    void validateOpen() const;
};


class CSVReader {
public:

    struct Config {
        char delimiter = ',';              ///< Column delimiter (0 = auto-detect)
        bool has_header = true;            ///< First row contains headers
        std::string comment_char = "#";    ///< Comment line prefix
        bool skip_empty_lines = true;      ///< Skip empty lines
    };
    

    explicit CSVReader(const std::string& filename, const Config& config = Config());

    std::vector<std::vector<double>> readAll();

    std::vector<std::vector<double>> readColumns(const std::vector<int>& column_indices);
    

    std::vector<std::vector<double>> readColumns(const std::vector<std::string>& column_names);

    const std::vector<std::string>& getHeaders() const { return headers_; }
    

    size_t getRowCount() const;
    

    size_t getColumnCount() const;

private:
    std::string filename_;
    Config config_;
    std::vector<std::string> headers_;
    
    char detectDelimiter(const std::string& line);
    std::vector<std::string> parseLine(const std::string& line);
    double parseValue(const std::string& str);
};

class GnuplotInterface {
public:
    struct PlotConfig {
        std::string title = "Plot";           ///< Plot title
        std::string xlabel = "x";             ///< X-axis label
        std::string ylabel = "y";             ///< Y-axis label
        std::string output_file = "";         ///< Output file (empty = display)
        std::string terminal = "png";         ///< Output terminal (png, pdf, svg)
        int width = 1024;                     ///< Image width
        int height = 768;                     ///< Image height
        bool grid = true;                     ///< Show grid
        bool legend = true;                   ///< Show legend
        std::string style = "lines";          ///< Plot style (lines, points, linespoints)
    };
    
 
    static bool isAvailable();
    
  
    static void plotSolution(const ODESolution& solution,
                            const std::vector<std::string>& component_names = {},
                            const PlotConfig& config = PlotConfig());
   

    static void plotCSV(const std::string& csv_file,
                       int x_column,
                       const std::vector<int>& y_columns,
                       const PlotConfig& config = PlotConfig());

    static void plotPhasePortrait(const ODESolution& solution,
                                 int component1,
                                 int component2,
                                 const PlotConfig& config = PlotConfig());

    static void plot3DTrajectory(const ODESolution& solution,
                                int comp1, int comp2, int comp3,
                                const PlotConfig& config = PlotConfig());
  
    static void plotConvergence(const std::vector<double>& error_history,
                               const PlotConfig& config = PlotConfig());
    

    static std::string executeScript(const std::string& script);

private:
    static void validateGnuplotAvailable();
    static std::string buildPlotCommand(const PlotConfig& config);
};

class SolutionAnalyzer {
public:
 
    struct Statistics {
        double mean;           ///< Average value
        double std_dev;        ///< Standard deviation
        double min;            ///< Minimum value
        double max;            ///< Maximum value
        double range;          ///< Max - min
        double median;         ///< Median value
        size_t n_samples;      ///< Number of samples
    };
    

    static Statistics computeStatistics(const ODESolution& solution, int component);
    
    static std::vector<std::pair<double, double>> detectPeaks(
        const ODESolution& solution,
        int component,
        double threshold = 0.1);
    
    static std::vector<double> detectZeroCrossings(
        const ODESolution& solution,
        int component);
    

    static std::vector<std::pair<double, double>> computeFFT(
        const ODESolution& solution,
        int component);
    
    static double estimatePeriod(const ODESolution& solution, int component);

    struct PhaseSpaceProperties {
        double trajectory_length;  ///< Total arc length
        double area_enclosed;      ///< Area enclosed by trajectory
        bool is_closed;            ///< Whether trajectory is closed loop
        double winding_number;     ///< Number of loops around origin
    };
    
    static PhaseSpaceProperties analyzePhaseSpace(
        const ODESolution& solution,
        int comp1, int comp2);

    static std::vector<std::vector<double>> interpolate(
        const ODESolution& solution,
        const std::vector<double>& time_points,
        const std::string& method = "cubic");
    
    static std::vector<double> computeDerivative(
        const ODESolution& solution,
        int component);
    
    struct ChaosMetrics {
        double lyapunov_exponent;  ///< Approximate largest Lyapunov exponent
        bool is_chaotic;           ///< True if exponent > 0
        double divergence_rate;    ///< Rate of trajectory divergence
    };
    
    static ChaosMetrics detectChaos(const ODESolution& solution);
    
    static double detectSteadyState(
        const ODESolution& solution,
        int component,
        double tolerance = 1e-3);
};


class DataConverter {
public:

    static std::string toJSON(const ODESolution& solution, bool pretty = true);
    
    static std::string toXML(const ODESolution& solution);
    

    static void toMATLAB(const ODESolution& solution,
                        const std::string& filename,
                        const std::vector<std::string>& variable_names = {});

    static void toNumPy(const ODESolution& solution,
                       const std::string& filename);
    
    static ODESolution fromJSON(const std::string& json);
};

class PerformanceTimer {
public:

    void start();
    

    double stop();
    double elapsed() const;

    void reset();
    

    bool isRunning() const { return running_; }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_ = false;
};

class Logger {
public:
    enum class Level {
        DEBUG,    ///< Detailed debugging information
        INFO,     ///< Informational messages
        WARNING,  ///< Warning messages
        ERROR     ///< Error messages
    };
    
  
    static void setLevel(Level level);
    

    static void log(Level level, const std::string& message);

    static void debug(const std::string& message) { log(Level::DEBUG, message); }
    static void info(const std::string& message) { log(Level::INFO, message); }
    static void warning(const std::string& message) { log(Level::WARNING, message); }
    static void error(const std::string& message) { log(Level::ERROR, message); }

    static void setOutputFile(const std::string& filename);

    static void setTimestamps(bool enable);

private:
    static Level min_level_;
    static std::string output_file_;
    static bool timestamps_enabled_;
    static std::ofstream log_file_;
    
    static std::string levelToString(Level level);
    static std::string getCurrentTime();
};

} // namespace utils
} // namespace ode_solver

#endif // ODE_SOLVER_UTILITIES_HPP