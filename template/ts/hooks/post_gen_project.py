import subprocess

# Initialize Git repository
try:
    subprocess.run(["git", "init"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to initialize Git repository. Error: {e}")
except FileNotFoundError:
    print("Git command not found. Please install Git and try again.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
