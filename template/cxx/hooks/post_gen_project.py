import os
import shutil
import subprocess


def remove_path_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


cxx_build_tool = "{{cookiecutter.cxx_build_tool}}"

test_main = "tests/test_main.cpp"

cmake_root = "CMakeLists.txt"
cmake_test = "tests/CMakeLists.txt"
cmake_custom = "cmake"
cmake_third = "third_party/CMakeLists.txt"

xmake_root = "xmake.lua"
xmake_test = "tests/xmake.lua"
xmake_third = "third_party/xmake.lua"

if cxx_build_tool == "cmake":
    remove_path_if_exists(xmake_root)
    remove_path_if_exists(xmake_test)
    remove_path_if_exists(xmake_third)
    remove_path_if_exists(test_main)
elif cxx_build_tool == "xmake":
    remove_path_if_exists(cmake_root)
    remove_path_if_exists(cmake_test)
    remove_path_if_exists(cmake_custom)
    remove_path_if_exists(cmake_third)
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
