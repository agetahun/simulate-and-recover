import numpy as np
import csv
from EZdiffusion import EZDiffusion

def calculate_average(biases, squared_errors, sample_size, iterations=1000):
    """
    Calculate the average bias and squared error for each parameter (nu, alpha, tau) for a given sample size.

    Parameters:
    biases: List of biases for each iteration.
    squared_errors: List of squared errors for each iteration.
    sample_size: The sample size used in the simulation.
    iterations: Number of iterations for the simulation.

    Returns:
    averages: A dictionary containing the average bias and squared error for each parameter.
    """
    # Compute the average bias and squared error for each parameter (nu, alpha, tau)
    avg_bias_nu = np.mean([b[0] for b in biases])
    avg_bias_alpha = np.mean([b[1] for b in biases])
    avg_bias_tau = np.mean([b[2] for b in biases])

    avg_squared_error_nu = np.mean([se[0] for se in squared_errors])
    avg_squared_error_alpha = np.mean([se[1] for se in squared_errors])
    avg_squared_error_tau = np.mean([se[2] for se in squared_errors])

    # Create a dictionary with the results
    averages = {
        'Sample Size': f"SampleSize:{sample_size}\n",
        'Average_Bias_nu': f"Average Bias (nu,alpha,tau): {avg_bias_nu,avg_bias_alpha,avg_bias_tau}\n",
        'Average_Squared_Error_nu': f"Average squared Error (nu,alpha,tau): {avg_squared_error_nu,avg_squared_error_alpha,avg_squared_error_tau}\n\n"
    }

    return averages

def write_to_csv():
    """Function to run the simulation for different sample sizes and write the average bias and squared error to a CSV file."""
    # Initialize the EZ diffusion model
    model = EZDiffusion(seed=42)

    # Define the sample sizes and the number of iterations
    sample_sizes = [10, 40, 4000]
    iterations = 1000

    # List to hold all the average results
    average_results = []

    # Loop through each sample size and run simulation
    for N in sample_sizes:
        print(f"Running simulation for N={N}")
        biases, squared_errors = model.run_simulation(N, iterations)

        # Calculate the average bias and squared error
        averages = calculate_average(biases, squared_errors, N, iterations)

        # Append the averages for this sample size to the list
        average_results.append(averages)

    # Write the average results to a CSV file
    with open("simulation_results.csv", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=average_results[0].keys())
        writer.writeheader()
        writer.writerows(average_results)
    
    print("Results have been written to simulation_results.csv")

def main():
    # Run the simulation and write the results to CSV
    write_to_csv()


if __name__ == "__main__":
    main()