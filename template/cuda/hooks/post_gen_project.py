import os
import re
import shutil
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

# Path to the directory where this hook script is running (root of generated project)
project_dir = os.getcwd()
# Path to the source template directory for the current language
template_dir = "{{cookiecutter._template}}" # This is a special variable provided by Cookiecutter
# Path to the _shared_files directory
shared_files_dir = os.path.abspath(os.path.join(template_dir, "..", "_shared_files"))

# Files to copy
files_to_copy = ["SECURITY.md", ".editorconfig"]
for file_name in files_to_copy:
    src_path = os.path.join(shared_files_dir, file_name)
    dst_path = os.path.join(project_dir, file_name)
    try:
        shutil.copy(src_path, dst_path)
        print(f"Copied {file_name} from shared files.")
    except Exception as e:
        print(f"Error copying {file_name}: {e}")

# License selection logic
license_choice = "{{cookiecutter.license}}"
full_name = "{{cookiecutter.full_name}}"
current_year = "{% now 'local', '%Y' %}" # Jinja2 processed by Cookiecutter

mit_license_file = "LICENSE-MIT"
apache_license_file = "LICENSE-APACHE"
final_license_file = "LICENSE"

license_name = ""
license_spdx_id = "" # Not directly used in CUDA project files like CMake/XMake, but good for consistency
license_badge = ""
readme_notice = ""

if license_choice == "MIT":
    license_name = "MIT License"
    license_spdx_id = "MIT"
    license_badge = "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)"
    if os.path.exists(mit_license_file):
        shutil.move(mit_license_file, final_license_file)
    if os.path.exists(apache_license_file):
        os.remove(apache_license_file)
    readme_notice = f"This project is licensed under the {license_name} - see the [LICENSE](LICENSE) file for details."

elif license_choice == "Apache-2.0":
    license_name = "Apache License 2.0"
    license_spdx_id = "Apache-2.0"
    license_badge = "[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)"
    if os.path.exists(apache_license_file):
        shutil.move(apache_license_file, final_license_file)
    if os.path.exists(mit_license_file):
        os.remove(mit_license_file)
    readme_notice = f"This project is licensed under the {license_name} - see the [LICENSE](LICENSE) file for details."

elif license_choice == "None (Proprietary)":
    license_name = "Proprietary"
    license_spdx_id = "UNLICENSED"
    license_badge = "" # No badge for proprietary
    if os.path.exists(mit_license_file):
        os.remove(mit_license_file)
    if os.path.exists(apache_license_file):
        os.remove(apache_license_file)
    with open(final_license_file, "w") as f:
        f.write(f"Copyright (c) {current_year} {full_name}\\n")
        f.write("All rights reserved.\\n")
    readme_notice = "All rights reserved. See the [LICENSE](LICENSE) file for details."

# Update README.md
readme_path = os.path.join(project_dir, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding='utf-8') as f:
        readme_content = f.read()
    readme_content = readme_content.replace("{{ VENDORED_LICENSE_BADGE }}", license_badge if license_badge else "")
    readme_content = readme_content.replace("This project is licensed under the {{ VENDORED_LICENSE_NAME }} - see the [LICENSE](LICENSE) file for details.", readme_notice)
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme_content)
    print(f"Updated README.md for {license_choice} license.")

# For CUDA (and C++), check build files for direct references to LICENSE-MIT or LICENSE-APACHE
# This is a precautionary step, as it's unlikely they'd reference the specific template names.
# If such references exist, they should be changed to "LICENSE".
# Example for CMakeLists.txt (if it had such a reference):
# cmake_file_path = os.path.join(project_dir, "CMakeLists.txt")
# if os.path.exists(cmake_file_path):
#     with open(cmake_file_path, "r", encoding='utf-8') as f:
#         cmake_content = f.read()
#     updated_cmake_content = cmake_content.replace("LICENSE-MIT", "LICENSE").replace("LICENSE-APACHE", "LICENSE")
#     if updated_cmake_content != cmake_content:
#         with open(cmake_file_path, "w", encoding='utf-8') as f:
#             f.write(updated_cmake_content)
#         print("Updated build file references to LICENSE.")
# Similar logic could be applied for xmake.lua if needed.

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True, capture_output=True, text=True)
    print("Initialized Git repository.")
except subprocess.CalledProcessError as e:
    print(f"Failed to initialize Git repository. Error: {e.stderr}")
except FileNotFoundError:
    print("Git command not found. Please ensure Git is installed and in your PATH.")
except Exception as e:
    print(f"An unexpected error occurred during git init: {e}")

# Install pre-commit hooks
try:
    # Attempt to run pre-commit install
    # Ensure that this runs after git init and after .pre-commit-config.yaml is in place
    subprocess.run(["pre-commit", "install"], check=True, capture_output=True, text=True)
    print("Installed pre-commit hooks.")
except subprocess.CalledProcessError as e:
    print(f"Failed to install pre-commit hooks. Error: {e.stderr}")
    print("Please ensure pre-commit is installed and run 'pre-commit install' manually if needed.")
except FileNotFoundError:
    print("'pre-commit' command not found. You may need to install it (e.g., 'pip install pre-commit') and then run 'pre-commit install' manually.")
except Exception as e:
    print(f"An unexpected error occurred during pre-commit install: {e}")
