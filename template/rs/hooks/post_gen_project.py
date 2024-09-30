import os
import subprocess

# Function to remove a file if it exists
def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Get crate type from the cookiecutter template
crate_type = "{{cookiecutter.crate_type}}"

# Define paths to main.rs and lib.rs
main_rs = "src/main.rs"
lib_rs = "src/lib.rs"

# Handle crate type and remove unnecessary files
if crate_type == "bin":
    remove_file_if_exists(lib_rs)
elif crate_type == "lib":
    remove_file_if_exists(main_rs)
else:
    raise ValueError(f"Unknown crate type: {crate_type}")

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to initialize Git repository. Error: {e}")
except FileNotFoundError:
    print("Git command not found. Please install Git and try again.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
