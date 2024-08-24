-- Set project name and language
set_project("{{cookiecutter.project_slug}}")
set_languages("c++{{cookiecutter.cxx_standard_version}}")

{% if cookiecutter.cxx_project_type == "binary" -%}
-- Binary project setup
target("{{cookiecutter.project_slug}}")
    set_targetdir("build/bin")
    set_kind("binary")
    add_files("src/*.cpp")
    add_headerfiles("src/*.h")

    {%- if cookiecutter.cxx_share_enabled == "STATIC" -%}
    -- Static link settings for non-macOS platforms
    if not is_plat("macosx") then
        add_ldflags("-static", {force = true})
    end
    {%- endif %}

{%- else -%}
-- Library project setup
target("{{cookiecutter.project_slug}}")
    set_targetdir("build/lib")
    set_kind("{{cookiecutter.cxx_share_enabled | lower}}")
    add_files("src/*.cpp")
    add_headerfiles("src/*.h")
{%- endif %}
