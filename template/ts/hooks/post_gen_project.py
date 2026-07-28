import subprocess

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True)
except subprocess.CalledProcessError:
    print("Failed to initialize Git repository.")
except FileNotFoundError:
    print("Git not found. Please install Git and try again.")
