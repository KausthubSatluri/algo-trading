
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("Please use 'python src/train.py' to train or 'python src/evaluate.py' to evaluate.")
