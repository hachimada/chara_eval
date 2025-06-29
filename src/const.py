from pathlib import Path

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    print(f"Current working directory: {Path.cwd()}")
    print(f"Root directory: {ROOT}")
