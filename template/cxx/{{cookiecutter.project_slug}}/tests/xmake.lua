-- Add Google Test package
add_requires("gtest")

-- Add test target
target("{{cookiecutter.project_slug}}-tests")
    set_kind("binary")
    add_files("*.cpp")
    add_deps("{{cookiecutter.package_name}}_lib")
    add_packages("gtest")
