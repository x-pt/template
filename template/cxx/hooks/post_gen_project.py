import os

cxx_build_tool = "{{cookiecutter.cxx_build_tool}}"

cmake_file = "CMakeLists.txt"
xmake_file = "xmake.lua"

if cxx_build_tool == "cmake":
    if os.path.exists(xmake_file):
        os.remove(xmake_file)
elif cxx_build_tool == "xmake":
    if os.path.exists(cmake_file):
        os.remove(cmake_file)
else:
    raise ValueError(f"Unknown cxx_build_tool: {cxx_build_tool}")
