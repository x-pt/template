import os
import subprocess

cxx_build_tool = "{{cookiecutter.cxx_build_tool}}"

cmake_root = "CMakeLists.txt"
cmake_test = "tests/CMakeLists.txt"
xmake_root = "xmake.lua"
xmake_test = "tests/xmake.lua"

if cxx_build_tool == "cmake":
    if os.path.exists(xmake_root):
        os.remove(xmake_root)
    if os.path.exists(xmake_test):
        os.remove(xmake_test)
elif cxx_build_tool == "xmake":
    if os.path.exists(cmake_root):
        os.remove(cmake_root)
    if os.path.exists(cmake_test):
        os.remove(cmake_test)
else:
    raise ValueError(f"Unknown cxx_build_tool: {cxx_build_tool}")

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True)
except subprocess.CalledProcessError:
    print("Failed to initialize Git repository. Please make sure Git is installed and try manually.")
except FileNotFoundError:
    print("Git command not found. Please install Git and try again.")
