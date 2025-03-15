import numpy as np
import os
import time
import pandas as pd
from EZdiffusion import EZDiffusion

def run_simulation(self, N, iterations=1000):
    """
    Run simulate-and-recover iterations.
    
    Parameters:
    N: Sample size
    iterations: Number of iterations to run
    
    Returns:
    results: DataFrame with true parameters, estimated parameters, bias, and squared error
    """
    results = []
    
    for i in range(iterations):
        # Step 1: Generate true parameters
        nu, alpha, tau = self.generate_parameters()
        
        # Step 2: Get predicted summary statistics
        R_pred, M_pred, V_pred = self.forward_equations(nu, alpha, tau)
        
        # Step 3: Simulate observed summary statistics
        R_obs, M_obs, V_obs = self.simulate_observations(R_pred, M_pred, V_pred, N)
        
        # Step 4: Recover estimated parameters
        try:
            nu_est, alpha_est, tau_est = self.inverse_equations(R_obs, M_obs, V_obs)
            
            # Step 5: Calculate bias and squared error
            b, b2 = self.compute_bias(nu, alpha, tau, nu_est, alpha_est, tau_est)
            # bias_nu = nu - nu_est
            # bias_alpha = alpha - alpha_est
            # bias_tau = tau - tau_est
            
            # squared_error = (bias_nu**2 + bias_alpha**2 + bias_tau**2)
            
            # Add to results
            # results.append({
            #     'N': N,
            #     'iteration': i,
            #     'nu_true': nu,
            #     'alpha_true': alpha,
            #     'tau_true': tau,
            #     'nu_est': nu_est,
            #     'alpha_est': alpha_est,
            #     'tau_est': tau_est,
            #     'bias_nu': bias_nu,
            #     'bias_alpha': bias_alpha,
            #     'bias_tau': bias_tau,
            #     'R_pred': R_pred,
            #     'M_pred': M_pred,
            #     'V_pred': V_pred,
            #     'R_obs': R_obs,
            #     'M_obs': M_obs,
            #     'V_obs': V_obs,
            #     'bias_magnitude': np.sqrt(bias_nu**2 + bias_alpha**2 + bias_tau**2),
            #     'squared_error': squared_error
            # })
        except Exception as e:
            print(f"Error in iteration {i} with N={N}: {e}")
    
    return pd.DataFrame(results) 

def run_full_simulation(self, sample_sizes=[10, 40, 4000], iterations=1000):
    """
    Run simulations for multiple sample sizes.
    
    Parameters:
    sample_sizes: List of sample sizes to test
    iterations: Number of iterations per sample size
    
    Returns:
    all_results: Combined DataFrame of results for all sample sizes
    """
    all_results = []
    
    for N in sample_sizes:
        print(f"Running simulation with N={N}")
        results = self.run_simulation(N, iterations)
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)

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