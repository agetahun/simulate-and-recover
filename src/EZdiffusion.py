#written using ChatGBT
"""
EZ Diffusion Model - Simulate and Recover Implementation

This script performs a simulate-and-recover exercise for the EZ diffusion model.
It tests whether parameters can be accurately recovered from simulated data.
"""

import numpy as np
from scipy.stats import binom, norm, gamma

class EZDiffusion:
    def __init__(self, seed=None):
        """Initialize the EZ Diffusion model with optional random seed."""
        if seed is not None:
            np.random.seed(seed)

        # Define parameter ranges
        N_values = [10, 40, 4000]  # Sample sizes
        total_iterations = 1000 # are these ever used?

    def generate_parameters(self):
        """Generate random parameters within realistic ranges."""
        # Generate "true" parameters
        alpha = np.random.uniform(0.5, 2.0)  # Boundary separation
        nu = np.random.uniform(0.5, 2.0)     # Drift rate
        tau = np.random.uniform(0.1, 0.5)    # Non-decision time
        
        return nu, alpha, tau

    # Function to compute predicted summary statistics
    def compute_predicted_stats(self, nu, alpha, tau):
        """
        Calculate predicted statistics using forward EZ equations.
        
        Parameters:
        nu: Drift rate
        alpha: Boundary separation
        tau: Non-decision time
        
        Returns:
        R_pred: Predicted accuracy rate
        M_pred: Predicted mean response time
        V_pred: Predicted variance of response times
        """
        # Calculate y = exp(-alpha * nu)
        y = np.exp(-alpha * nu)
        
        # Equation 1: Predicted accuracy
        R_pred = 1 / (1 + y)
        
        # Equation 2: Predicted mean RT
        M_pred = tau + (alpha / (2 * nu)) * ((1 - y) / (1 + y))
        
        # Equation 3: Predicted variance of RT
        V_pred = (alpha / (2 * nu**3)) * ((1 - 2*alpha*nu*y - y**2) / (1 + y)**2)
        
        return R_pred, M_pred, V_pred

    def simulate_observations(self, R_pred, M_pred, V_pred, N):
        """
        Simulate observed statistics from sampling distributions.
        
        Parameters:
        R_pred: Predicted accuracy
        M_pred: Predicted mean RT
        V_pred: Predicted variance of RT
        N: Sample size
        
        Returns:
        R_obs: Observed accuracy
        M_obs: Observed mean RT
        V_obs: Observed variance of RT
        """
        # Equation 7: Simulate observed number of correct trials
        T_obs = binom.rvs(N, R_pred)
        R_obs = T_obs / N
        
        # Equation 8: Simulate observed mean RT
        M_obs = norm.rvs(loc=M_pred, scale=np.sqrt(V_pred/N))
        
        # Equation 9: Simulate observed variance of RT
        # Note: We use the fact that (n-1)s²/σ² ~ Gamma((n-1)/2, 2/(n-1))
        # where s² is the sample variance and σ² is the population variance
        if N > 1:  # Need at least 2 observations for variance
            scale_factor = (N - 1) / (2 * V_pred)
            V_obs = gamma.rvs(N - 1) / (2 * scale_factor)
        else:
            V_obs = V_pred  # Just use the predicted variance if N=1
        
        return R_obs, M_obs, V_obs

    def inverse_equations(self, R_obs, M_obs, V_obs):
        """
        Recover estimated parameters using inverse EZ equations.
        
        Parameters:
        R_obs: Observed accuracy
        M_obs: Observed mean RT
        V_obs: Observed variance of RT
        
        Returns:
        nu_est: Estimated drift rate
        alpha_est: Estimated boundary separation
        tau_est: Estimated non-decision time
        """
        try:
            # Ensure R_obs is between 0 and 1
            R_obs = max(min(R_obs, 0.9999), 0.0001)
            
            # Calculate L = log(R_obs/(1-R_obs))
            L = np.log(R_obs / (1 - R_obs))
            
            # Equation 4: Estimate drift rate
            sgn = np.sign(R_obs - 0.5)
            # Handle special case to avoid division by zero
            if V_obs <= 0 or R_obs == 0.5:
                nu_est = 0  # Default value for edge cases
            else:
                # Calculate the term under the square root
                sqrt_term = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs
                nu_est = sgn * (sqrt_term ** 0.25)  # Ensure non-negative under sqrt
            
            # Equation 5: Estimate boundary separation
            alpha_est = L / nu_est if nu_est != 0 else 0
            
            # Equation 6: Estimate non-decision time
            if nu_est != 0 and alpha_est != 0:
                exp_term = np.exp(-nu_est * alpha_est)
                bracket_term = (1 - exp_term) / (1 + exp_term)
                tau_est = M_obs - (alpha_est / (2 * nu_est)) * bracket_term
            else:
                tau_est = M_obs  # Default if we can't calculate
            
            return nu_est, alpha_est, tau_est
        
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            # Return default values if calculation fails
            return 0, 0, 0
        
    def compute_bias(self, nu, alpha, tau, nu_est, alpha_est, tau_est):
        """
        Compute the estimation bias and squared error.
        
        Parameters:
        nu: Drift rate
        alpha: Boundary separation
        tau: Non-decision time
        nu_est: Estimated drift rate
        alpha_est: Estimated boundary separation
        tau_est: Estimated non-decision time
        
        Returns:
        b: estimation bias
        b2: squared error
        """
        true_params = (nu, alpha, tau)
        estimated_params = (nu_est, alpha_est, tau_est)

        # Compute the difference (b)
        b = tuple(true_params - estimated_params for true_params, estimated_params in zip(true_params, estimated_params))

        # Compute the squared error (b^2)
        b_squared = tuple(b_i ** 2 for b_i in b)

        return b, b_squared

    def run_simulation(self, N, iterations=1000):
        """
        Run simulate-and-recover iterations.
        
        Parameters:
        N: Sample size
        iterations: Number of iterations to run
        
        Returns:
        results: DataFrame with true parameters, estimated parameters, bias, and squared error
        """
        biases = []
        squared_errors = []
        
        for i in range(iterations):
            # Step 1: Generate true parameters
            nu, alpha, tau = self.generate_parameters()
            
            # Step 2: Get predicted summary statistics
            R_pred, M_pred, V_pred = self.compute_predicted_stats(nu, alpha, tau)
            
            # Step 3: Simulate observed summary statistics
            R_obs, M_obs, V_obs = self.simulate_observations(R_pred, M_pred, V_pred, N)
            
            # Step 4: Recover estimated parameters
            try:
                nu_est, alpha_est, tau_est = self.inverse_equations(R_obs, M_obs, V_obs)
                
                # Step 5: Calculate bias and squared error
                b, b_squared = self.compute_bias(nu, alpha, tau, nu_est, alpha_est, tau_est)

                # Add to results
                biases.append(b)
                squared_errors.append(b_squared)
            except Exception as e:
                print(f"Error in iteration {i} with N={N}: {e}")
        return biases, squared_errors

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
        
        return results