-- Add Google Test package
add_requires("gtest", {configs = {main = true, gmock = true}})

-- Add test target
target("{{cookiecutter.project_slug}}-tests")
    set_targetdir("$(buildir)/bin")
    set_kind("binary")
    add_files("*.cpp")
    add_deps("{{cookiecutter.package_name}}_lib")
    add_packages("gtest")
    add_links("gtest_main")

    if is_plat("windows") then
        -- fixes "LINK : fatal error LNK1561: entry point must be defined"
        add_ldflags("/subsystem:console")
    end
