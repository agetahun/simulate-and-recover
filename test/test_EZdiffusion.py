import unittest
import numpy as np
import sys
# Add the full path to the directory containing EZDiffusion.py
sys.path.append('/workspace/simulate-and-recover/src')
# Now you can import the module
from EZdiffusion import EZDiffusion

class TestEZDiffusion(unittest.TestCase):
    
    def setUp(self):
        """Initialize the EZDiffusion model and set a fixed random seed."""
        self.model = EZDiffusion(seed=42)
    
    def test_generate_parameters(self):
        """Test that parameters are within the expected ranges."""
        nu, alpha, tau = self.model.generate_parameters()
        self.assertGreaterEqual(nu, 0.5)
        self.assertLessEqual(nu, 2.0)
        self.assertGreaterEqual(alpha, 0.5)
        self.assertLessEqual(alpha, 2.0)
        self.assertGreaterEqual(tau, 0.1)
        self.assertLessEqual(tau, 0.5)

    def test_compute_predicted_stats(self):
        """Test the predicted statistics."""
        nu, alpha, tau = 1.5, 1.0, 0.2  # Sample input
        R_pred, M_pred, V_pred = self.model.compute_predicted_stats(nu, alpha, tau)
        
        # Check if predicted statistics are reasonable numbers (non-NaN, finite values)
        self.assertTrue(np.isfinite(R_pred))
        self.assertTrue(np.isfinite(M_pred))
        self.assertTrue(np.isfinite(V_pred))
        
        # Check if predicted accuracy R_pred is within [0, 1]
        self.assertGreaterEqual(R_pred, 0)
        self.assertLessEqual(R_pred, 1)

    def test_simulate_observations(self):
        """Test the simulation of observations based on predicted stats."""
        R_pred, M_pred, V_pred = 0.7, 0.6, 0.2  # Sample predicted values
        N = 100  # Sample size
        
        R_obs, M_obs, V_obs = self.model.simulate_observations(R_pred, M_pred, V_pred, N)
        
        # Check if observed statistics are reasonable numbers (non-NaN, finite values)
        self.assertTrue(np.isfinite(R_obs))
        self.assertTrue(np.isfinite(M_obs))
        self.assertTrue(np.isfinite(V_obs))
        
        # Check if observed accuracy R_obs is within [0, 1]
        self.assertGreaterEqual(R_obs, 0)
        self.assertLessEqual(R_obs, 1)
        
        # Check if observed mean RT and variance are positive
        self.assertGreater(M_obs, 0)
        self.assertGreater(V_obs, 0)

    def test_inverse_equations(self):
        """Test the recovery of parameters from simulated observations."""
        # True parameters
        nu, alpha, tau = 1.5, 1.0, 0.2
        
        # Simulate predicted statistics
        R_pred, M_pred, V_pred = self.model.compute_predicted_stats(nu, alpha, tau)
        
        # Simulate observations
        R_obs, M_obs, V_obs = self.model.simulate_observations(R_pred, M_pred, V_pred, 100)
        
        # Recover estimated parameters
        nu_est, alpha_est, tau_est = self.model.inverse_equations(R_obs, M_obs, V_obs)

        # Ensure that estimated parameters are reasonable numbers (non-NaN, finite values)
        self.assertTrue(np.isfinite(nu_est))
        self.assertTrue(np.isfinite(alpha_est))
        self.assertTrue(np.isfinite(tau_est))
        
        # Ensure that estimated parameters are close to the true parameters
        self.assertAlmostEqual(nu, nu_est, delta=0.5)
        self.assertAlmostEqual(alpha, alpha_est, delta=0.5)
        self.assertAlmostEqual(tau, tau_est, delta=0.5)

    def test_compute_bias(self):
        """Test the bias and squared error calculation."""
        nu, alpha, tau = (1.5, 1.0, 0.2)  # Sample true parameters
        nu_est, alpha_est, tau_est = (1.4, 1.05, 0.19)  # Sample estimated parameters
        
        # Compute bias and squared error
        b, b_squared = self.model.compute_bias(nu, alpha, tau, nu_est, alpha_est, tau_est)
        
        # Check that the bias and squared error are lists or arrays of the correct size
        self.assertEqual(len(b), 3)  # Should have 3 elements (nu, alpha, tau)
        self.assertEqual(len(b_squared), 3)  # Should have 3 elements (nu, alpha, tau)
        
        # Check if the bias and squared errors are reasonable (non-NaN, finite values)
        for i in range(3):
            self.assertTrue(np.isfinite(b[i]))
            self.assertTrue(np.isfinite(b_squared[i]))

    def test_run_simulation(self):
        """Test the full simulation run."""
        # Run a small simulation with N = 10 and 5 iterations to ensure it completes
        biases, squared_errors = self.model.run_simulation(N=10, iterations=5)
        
        # Check if the result is a valid non-empty array or list
        self.assertEqual(len(biases), 5)
        self.assertEqual(len(squared_errors), 5)
        
        # Check if biases and squared errors are reasonable numbers (non-NaN, finite values)
        self.assertTrue(np.all(np.isfinite(biases)))
        self.assertTrue(np.all(np.isfinite(squared_errors)))

    def test_bias_is_close_to_zero(self):
        """Test that the average bias is close to zero."""
        biases, squared_errors = self.model.run_simulation(N=100, iterations=1000)
        
        # Check that the average bias is close to 0
        avg_bias = np.mean(biases)
        self.assertAlmostEqual(avg_bias, 0, delta=0.05)  # Allow a small tolerance

    def test_bias_decreases_with_increasing_N(self):
        """Test that bias decreases as N (sample size) increases."""
        # Run simulations with different sample sizes
        biases_10, _ = self.model.run_simulation(N=10, iterations=1000)
        biases_40, _ = self.model.run_simulation(N=40, iterations=1000)
        biases_100, _ = self.model.run_simulation(N=100, iterations=1000)

        # Check that bias decreases as N increases
        avg_bias_10 = np.mean(biases_10)
        avg_bias_40 = np.mean(biases_40)
        avg_bias_100 = np.mean(biases_100)

        self.assertLess(abs(avg_bias_40), abs(avg_bias_10))
        self.assertLess(abs(avg_bias_100), abs(avg_bias_40))

    def test_bias_is_zero_when_obs_equals_pred(self):
        """Test that bias is zero when (R_obs, M_obs, V_obs) == (R_pred, M_pred, V_pred)."""
        # Generate true parameters
        nu, alpha, tau = 1.5, 1.0, 0.2  # Example values
        R_pred, M_pred, V_pred = self.model.compute_predicted_stats(nu, alpha, tau)
        
        # Simulate observations to match the predictions
        R_obs, M_obs, V_obs = R_pred, M_pred, V_pred

        # Recover estimated parameters from the observed statistics
        nu_est, alpha_est, tau_est = self.model.inverse_equations(R_obs, M_obs, V_obs)

        # Compute bias
        b, _ = self.model.compute_bias(nu, alpha, tau, nu_est, alpha_est, tau_est)

        # Assert that the bias is exactly zero for all components
        self.assertEqual(b, (0.0, 0.0, 0.0)) 
        
if __name__ == "__main__":
    unittest.main()