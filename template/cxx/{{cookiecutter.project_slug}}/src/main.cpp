#include <spdlog/spdlog.h>

#include <iostream>

#include "cxxopts/cxxopts.hpp"
#include "httplib/httplib.h"

int main(int argc, char* argv[]) {
    try {
        // Define the options
        cxxopts::Options options("MyProgram", "A brief description of the program");

        options.add_options()("h,help", "Print usage information")("n,name", "Name of the user", cxxopts::value<std::string>())("v,verbose", "Enable verbose mode");

        // Parse the arguments
        auto result = options.parse(argc, argv);

        // Handle help option
        if (result.count("help")) {
            spdlog::info(options.help());
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

        // Start a simple server using cpp-httplib
        httplib::Server svr;

        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("Hello, World!", "text/plain");
            spdlog::info("Handled request at /");
        });

        spdlog::info("Starting server on http://localhost:8080");
        svr.listen("localhost", 8080);

    } catch (const cxxopts::exceptions::exception& e) {
        spdlog::error("Error parsing options: {}", e.what());
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        return 1;
    }

    return 0;
}
