-- Add Google Test package
add_requires("gtest", {configs = {main = true}})

-- Add test target
target("{{cookiecutter.project_slug}}-tests")
    set_targetdir("$(buildir)/bin")
    set_kind("binary")
    add_files("*.cpp")
    add_deps("{{cookiecutter.package_name}}_lib")
    add_packages("gtest", "cuda")

    -- Define test run command
    -- after_build(function (target)
    --     os.exec("%s", target:targetfile())
    -- end)
