#!/bin/bash
# written using chatGBT

set -e  # Exit on error

python3 ./main.py > results.txt

echo "Simulation complete. Results saved in results.txt"
