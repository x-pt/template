-- Add Google Test package
add_requires("gtest")

-- Add test target
target("{{cookiecutter.project_slug}}-tests")
    set_targetdir("$(buildir)/bin")
    set_kind("binary")
    add_files("*.cpp")
    add_deps("{{cookiecutter.package_name}}_lib")
    add_packages("gtest", "cuda")
    add_links("cudart", "cublas")

    -- Define test run command
    -- after_build(function (target)
    --     os.exec("%s", target:targetfile())
    -- end)
