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
        
        # Check if observed mean RT and vari
if __name__ == "__main__":
    unittest.main()