-- Project Configuration
set_project("{{cookiecutter.project_slug}}")
set_version("{{cookiecutter.project_version}}")
set_languages("c++{{cookiecutter.cxx_standard_version}}")

-- Dependencies
add_requires("spdlog 1.15.0")       -- Add spdlog as a required dependency

-- Include third-party libraries
includes("third_party")

-- Define Library Target
target("{{cookiecutter.package_name}}_lib")
    set_kind("static")                -- Build as a static library
    set_targetdir("build/lib")        -- Specify output directory
    add_includedirs("include")        -- Include project's header files
    add_files("src/*.cpp")            -- Add project source files
    add_headerfiles("include/*.h")    -- Add project's header files
    add_packages("spdlog")            -- Link spdlog headers and library
    add_deps("httplib", "cxxopts")    -- Use httplib and cxxopts from third_party

-- Define Executable Target
target("{{cookiecutter.project_slug}}")
    set_kind("binary")                -- Build as an executable
    set_targetdir("build/bin")        -- Specify output directory
    add_files("src/main.cpp")         -- Add main source file
    add_deps("{{cookiecutter.package_name}}_lib") -- Link against the library
    add_packages("spdlog")            -- Include spdlog headers and library

-- Tests Configuration
includes("tests")                     -- Include the tests subdirectory
