#include <spdlog/spdlog.h>
#include <iostream>

#include "cxxopts/cxxopts.hpp"
#include "httplib/httplib.h"

int main(int argc, char* argv[]) {
    try {
        // Define the options
        cxxopts::Options options("MyProgram", "A simple HTTP server with command-line options");

        // Add command-line options
        options.add_options()
            ("h,help", "Print usage information")
            ("m,mode", "Server mode", cxxopts::value<std::string>()->default_value(""))
            ("n,name", "Name of the user", cxxopts::value<std::string>())
            ("v,verbose", "Enable verbose mode")
            ("p,port", "Server port", cxxopts::value<int>()->default_value("8080"));

        // Parse the arguments
        auto result = options.parse(argc, argv);

        // Handle help option
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Handle name option
        if (result.count("name")) {
            std::string name = result["name"].as<std::string>();
            spdlog::info("Hello, {}!", name);
        } else {
            spdlog::info("Hello, World!");
        }

        // Handle verbose option
        if (result.count("verbose")) {
            spdlog::set_level(spdlog::level::debug);
            spdlog::debug("Verbose mode is enabled.");
        }

        // Check if server mode is specified
        std::string mode = result["mode"].as<std::string>();
        if (mode == "server") {
            // Get port from command-line or use default
            int port = result["port"].as<int>();

            // Start a simple server using cpp-httplib
            httplib::Server svr;

            // Root route
            svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
                res.set_content("Hello, World!", "text/plain");
                spdlog::info("Handled request at /");
            });

            // Health check route
            svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
                res.set_content("Server is running", "text/plain");
                spdlog::info("Health check performed");
            });

            spdlog::info("Starting server on http://localhost:{}", port);
            if (!svr.listen("localhost", port)) {
                spdlog::error("Failed to start server on port {}", port);
                return 1;
            }
        } else if (!mode.empty()) {
            spdlog::error("Invalid mode. Use --mode server to start the HTTP server.");
            return 1;
        }

    } catch (const cxxopts::exceptions::exception& e) {
        spdlog::error("Error parsing options: {}", e.what());
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        return 1;
    }

    return 0;
}
