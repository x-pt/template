-- Specify the project details
set_project("{{cookiecutter.project_slug}}")
set_version("{{cookiecutter.project_version}}")

-- Specify languages
set_languages("cxx17")

-- Add CUDA support
add_requires("cuda")

-- Define the target for the library
target("{{cookiecutter.package_name}}_lib")
    set_targetdir("build/lib")
    set_kind("{{ cookiecutter.library_type }}") -- Can be "static" or "shared"
    set_policy("build.cuda.devlink", true)
    add_includedirs("include", {public = true})
    add_files("src/**.cpp", "src/**.cu")
    add_packages("cuda")
    add_links("cudart", "cublas")
    add_cugencodes("native")

-- Define the target for the executable
target("{{cookiecutter.project_slug}}")
    set_targetdir("build/bin")
    set_kind("binary")
    add_files("src/main.cpp")
    add_deps("{{cookiecutter.package_name}}_lib")

-- Add the tests
includes("tests")
