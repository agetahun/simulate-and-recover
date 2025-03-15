#written using ChatGBT
import numpy as np
import scipy.stats as stats

# Define parameter ranges
N_values = [10, 40, 4000]  # Sample sizes
total_iterations = 1000

# Function to compute predicted summary statistics
def compute_predicted_stats(v, alpha, tau):
    y = np.exp(-alpha * v)
    R_pred = 1 / (1 + y)
    M_pred = tau + (alpha / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (alpha / (2 * v**3)) * ((1 - 2 * alpha * v - y**2) / (1 + y)**2)

    # Ensure V_pred is non-negative
    V_pred = max(V_pred, 1e-6)  # Small positive value to prevent sqrt/gamma errors

    return R_pred, M_pred, V_pred

# Function to compute estimated parameters
def compute_estimated_params(R_obs, M_obs, V_obs):
    L = np.log(R_obs / (1 - R_obs))
    nu_est = np.sign(R_obs - 0.5) * (4 * np.sqrt(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs))
    alpha_est = L / nu_est
    tau_est = M_obs - (alpha_est / (2 * nu_est)) * ((1 - np.exp(-nu_est * alpha_est)) / (1 + np.exp(-nu_est * alpha_est)))
    return nu_est, alpha_est, tau_est

# Simulation loop
results = []
for N in N_values:
    biases = []
    squared_errors = []
    
    for _ in range(total_iterations):
        # Step 1: Generate true parameters
        v_true = np.random.uniform(0.5, 2)
        alpha_true = np.random.uniform(0.5, 2)
        tau_true = np.random.uniform(0.1, 0.5)
        
        # Step 2: Compute predicted statistics
        R_pred, M_pred, V_pred = compute_predicted_stats(v_true, alpha_true, tau_true)
        
        # Step 3: Simulate observed data
        R_obs = np.random.binomial(N, R_pred) / N
        M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
        V_obs = np.random.gamma((N - 1) / 2, (2 * V_pred) / (N - 1))
        
        # Step 4: Estimate parameters
        v_est, alpha_est, tau_est = compute_estimated_params(R_obs, M_obs, V_obs)
        
        # Step 5: Compute bias and squared error
        bias = np.array([v_true, alpha_true, tau_true]) - np.array([v_est, alpha_est, tau_est])
        squared_error = bias ** 2
        
        biases.append(bias)
        squared_errors.append(squared_error)
    
    # Store results
    avg_bias = np.mean(biases, axis=0)
    avg_squared_error = np.mean(squared_errors, axis=0)
    results.append((N, avg_bias, avg_squared_error))

# Print results
for N, bias, sq_error in results:
    print(f"N = {N}")
    print(f"Average Bias: v = {bias[0]:.4f}, alpha = {bias[1]:.4f}, tau = {bias[2]:.4f}")
    print(f"Average Squared Error: v = {sq_error[0]:.4f}, alpha = {sq_error[1]:.4f}, tau = {sq_error[2]:.4f}\n")
