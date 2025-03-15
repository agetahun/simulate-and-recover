import numpy as np
import os
import time
from EZdiffusion import EZDiffusion

def main():
    """Main function to run the simulation exercise"""
    start_time = time.time()

    # Initialize the EZ diffusion model
    model = EZDiffusion(seed=42)

    # Run the simulation for all sample sizes
    results = model.run_full_simulation(sample_sizes=[10, 40, 4000], iterations=1000)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/ez_diffusion_results.csv", index=False)

    # Analyze and plot results
    summary = model.analyze_results(results)
    summary.to_csv("results/ez_diffusion_summary.csv", index=False)
    model.plot_

if __name__ == "__main__":
    main()