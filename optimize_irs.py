import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

# Dataset path (ensure this matches your environment)
try:
    from config import DATASET_PATH
except ImportError:
    DATASET_PATH = "data/dataset.csv" 

def get_env_loss(env_type):
    """Function to determine signal loss based on environment type."""
    mapping = {"Urban": 12.0, "Suburban": 7.0, "Rural": 3.5, "Indoor": 18.0, "Industrial": 14.0}
    return mapping.get(env_type, 0.0)

def calculate_rssi(pattern_angle, user_data):
    """Calculate signal strength based on the reflection angle and user data."""
    distance_km = max(user_data['distance_m'] / 1000.0, 0.01)
    fspl_db = 32.44 + 20 * np.log10(distance_km) + 20 * np.log10(user_data['frequency_mhz'])
    env_loss = get_env_loss(user_data['environment_type'])
    
    # Physical simulation of IRS: 
    # Assume there is an "optimal" angle depending on user coordinates.
    # The closer the algorithm gets to this angle, the more the signal increases (up to 15 dBm).
    optimal_angle = (user_data['distance_m'] * 0.5) % 180 
    angle_difference = abs(pattern_angle - optimal_angle)
    
    # IRS Gain
    irs_gain = 15.0 - (angle_difference * 0.2)
    irs_gain = max(0, irs_gain) # Gain cannot be negative
    
    # Final RSSI equation
    rssi = (user_data['transmit_power_dbm'] 
            + user_data['antenna_gain_dbi'] 
            - fspl_db 
            - user_data['obstacle_loss_db'] 
            - env_loss 
            + irs_gain)
    return rssi

def objective_function(x, user_data):
    """Objective function: The algorithm changes x (angle) to minimize the output."""
    pattern_angle = x[0]
    rssi = calculate_rssi(pattern_angle, user_data)
    # We return the negative result because the algorithm searches for the minimum, 
    # and the lowest negative number represents the highest signal strength.
    return -rssi 

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH)
    
    # Select the first user in the data to test the algorithm
    sample_user = df.iloc[0]
    
    print("\n--- 1. Current User Data ---")
    print(f"Total distance: {sample_user['distance_m']:.2f} meters")
    print(f"Environment type: {sample_user['environment_type']}")
    print(f"Original signal strength (without optimization): {sample_user['rssi_dbm']:.2f} dBm")
    
    # IRS reflection angle bounds (from 0 to 180 degrees)
    bounds = [(0.0, 180.0)]
    
    print("\n--- 2. Running the Genetic Algorithm (Differential Evolution) ---")
    print("The algorithm is now testing thousands of angles to choose the best one...")
    
    # Run the algorithm
    result = differential_evolution(objective_function, bounds, args=(sample_user,))
    
    best_angle = result.x[0]
    best_rssi = -result.fun # Convert back to positive
    
    print("\n--- 3. Final Optimization Result ---")
    print(f"✅ Best reflection angle deduced: {best_angle:.2f} degrees")
    print(f"✅ Signal strength after optimization: {best_rssi:.2f} dBm")
    print(f"🚀 Signal improvement amount: {best_rssi - sample_user['rssi_dbm']:.2f} dB")