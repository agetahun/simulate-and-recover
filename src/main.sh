#!/bin/bash
# written using chatGBT

set -e  # Exit on error

# Ensure dependencies are installed (Docker should handle this in the image)

# Run the EZ diffusion simulation
python3 ./main.py > results.txt

echo "Simulation complete. Results saved in results.txt"
