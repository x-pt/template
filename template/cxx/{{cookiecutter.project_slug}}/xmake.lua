-- Set project name and language
set_project("{{cookiecutter.project_slug}}")
set_version("0.0.1")
set_languages("c++{{cookiecutter.cxx_standard_version}}")

-- Include directories
add_includedirs("include")

-- Add targets
target("{{cookiecutter.package_name}}_lib")
    set_targetdir("build/lib")
    set_kind("static")
    add_files("src/*.cpp")
    add_headerfiles("include/*.h")

target("{{cookiecutter.project_slug}}")
    set_targetdir("build/bin")
    set_kind("binary")
    add_files("src/main.cpp")
    add_deps("{{cookiecutter.package_name}}_lib")

-- Add tests
includes("tests")
