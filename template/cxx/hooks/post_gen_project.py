import os
import subprocess

def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

cxx_build_tool = "{{cookiecutter.cxx_build_tool}}"

cmake_root = "CMakeLists.txt"
cmake_test = "tests/CMakeLists.txt"
xmake_root = "xmake.lua"
xmake_test = "tests/xmake.lua"

if cxx_build_tool == "cmake":
    remove_file_if_exists(xmake_root)
    remove_file_if_exists(xmake_test)
elif cxx_build_tool == "xmake":
    remove_file_if_exists(cmake_root)
    remove_file_if_exists(cmake_test)
else:
    raise ValueError(f"Unknown cxx_build_tool: {cxx_build_tool}")

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to initialize Git repository. Error: {e}")
except FileNotFoundError:
    print("Git command not found. Please install Git and try again.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
