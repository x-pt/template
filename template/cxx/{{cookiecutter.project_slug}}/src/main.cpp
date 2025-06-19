// src/main.cpp
// This application serves as a demonstration for the C++ template.
// It includes:
// 1. A command-line interface (CLI) using cxxopts for argument parsing.
//    - By default, it prints a greeting.
//    - It accepts options like --name, --verbose, --port.
// 2. An optional HTTP server mode (activated with --mode server) using httplib.
//    - The server provides basic / and /health endpoints.
// 3. Logging capabilities using spdlog.

#include <iostream>        // For std::cout (used by cxxopts help)
#include <string>          // For std::string
#include <spdlog/spdlog.h> // For structured logging

// From third_party:
#include "cxxopts/cxxopts.hpp" // For easy command-line option parsing
#include "httplib/httplib.h"   // For creating a simple HTTP server

// It's good practice to initialize complex global objects or perform setup
// outside main if it becomes complex, but for this example, spdlog's default
// console logger is fine as is.
// Consider spdlog::stdout_color_mt("console") or similar for more control.

int main(int argc, char* argv[]) {
    // Initialize spdlog for console output.
    // Default level is info. Can be changed by the --verbose flag.
    // In a real application, consider more advanced sink configuration
    // (e.g., file logger, rotating logger).
    // spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v"); // Example custom pattern

    try {
        // --- Command-Line Option Setup ---
        // cxxopts::Options allows defining and parsing command-line arguments.
        // The first argument is the program name (for help messages), second is a short description.
        cxxopts::Options options("{{cookiecutter.project_slug}}",
                                 "A C++ application template showcasing CLI options and an optional HTTP server mode.\n"
                                 "Default run: Prints a greeting.\n"
                                 "Server mode: Runs a basic HTTP server.");

        options.add_options()
            ("h,help", "Print usage information and exit.")
            ("m,mode", "Operating mode. Use 'server' to start the HTTP server, or 'cli' for default greeting.",
             cxxopts::value<std::string>()->default_value("cli")) // Default to "cli" mode
            ("n,name", "Name to include in the greeting message.",
             cxxopts::value<std::string>()) // No default, optional
            ("v,verbose", "Enable verbose (debug level) logging.")
            ("p,port", "Port for the HTTP server if in 'server' mode.",
             cxxopts::value<int>()->default_value("8080"));

        // Parse the provided command-line arguments.
        auto result = options.parse(argc, argv);

        // --- Handle Help Option ---
        // If --help is passed, print the help message and exit successfully.
        if (result.count("help")) {
            // Using std::cout here as cxxopts::Options::help() returns a string formatted for console.
            std::cout << options.help() << std::endl;
            return 0;
        }

        // --- Verbose Mode ---
        // If --verbose is passed, set the logging level to debug.
        // This should be done early so subsequent logs respect the level.
        if (result.count("verbose")) {
            spdlog::set_level(spdlog::level::debug);
            spdlog::debug("Verbose logging enabled.");
        }

        // --- Mode Handling ---
        std::string mode = result["mode"].as<std::string>();
        spdlog::debug("Operating mode selected: {}", mode);

        if (mode == "server") {
            // --- HTTP Server Logic ---
            int port = result["port"].as<int>();
            httplib::Server svr;

            // Define the root route ("/") for the HTTP server.
            // It responds with a simple "Hello, World!" message.
            svr.Get("/", [](const httplib::Request& /*req*/, httplib::Response& res) {
                res.set_content("Hello, World from {{cookiecutter.project_slug}} server!", "text/plain");
                spdlog::info("HTTP GET request to / handled.");
            });

            // Define a health check route ("/health").
            // Useful for load balancers or uptime monitoring.
            svr.Get("/health", [](const httplib::Request& /*req*/, httplib::Response& res) {
                res.set_content("Server is healthy and running.", "text/plain");
                spdlog::info("HTTP GET request to /health (health check) handled.");
            });

            spdlog::info("Starting HTTP server on http://localhost:{}", port);
            // svr.listen blocks until the server is stopped (e.g., by another signal or error).
            // If listen fails to bind (e.g., port in use), it returns false.
            if (!svr.listen("localhost", port)) {
                spdlog::error("Failed to start HTTP server on port {}. Port might be in use.", port);
                return 1; // Indicate an error.
            }
            // Normally, execution would not reach here unless server is stopped externally.
            // For this example, server runs until Ctrl+C if not for other errors.

        } else if (mode == "cli") {
            // --- CLI Greeting Logic ---
            // This is the default mode.
            if (result.count("name")) {
                std::string name = result["name"].as<std::string>();
                spdlog::info("Hello, {}! Welcome to {{cookiecutter.project_slug}}.", name);
            } else {
                spdlog::info("Hello, World! Welcome to {{cookiecutter.project_slug}}.");
            }
            spdlog::debug("CLI mode execution finished.");
        } else {
            // An invalid mode was specified.
            spdlog::error("Invalid mode '{}' specified. Use 'cli' or 'server'.", mode);
            // Printing help might be useful here too.
            std::cerr << options.help() << std::endl;
            return 1; // Indicate an error.
        }

    } catch (const cxxopts::exceptions::exception& e) {
        // This catches errors specifically from cxxopts parsing (e.g., invalid argument type).
        spdlog::error("Error parsing command-line options: {}", e.what());
        // It's good practice to return a specific error code for option parsing errors.
        return 2;
    } catch (const std::exception& e) {
        // This is a catch-all for other standard exceptions.
        spdlog::critical("An unhandled standard exception occurred: {}", e.what());
        return 1; // General error.
    } catch (...) {
        // This catches any other unknown exceptions (non-standard C++ exceptions).
        spdlog::critical("An unknown non-standard exception occurred.");
        return 1; // General error.
    }

    // If everything executed successfully.
    spdlog::debug("Application finished successfully.");
    return 0;
}
