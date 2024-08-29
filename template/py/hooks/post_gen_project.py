import subprocess

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True)
except subprocess.CalledProcessError:
    print("Failed to initialize Git repository. Please make sure Git is installed and try manually.")
except FileNotFoundError:
    print("Git command not found. Please install Git and try again.")
