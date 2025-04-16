import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

class Satellite:
    def __init__(self, position, accuracy=0.1):
        """
        Initialize a satellite with a fixed position in 3D space
        
        Args:
            position (tuple): (x, y, z) coordinates in meters
            accuracy (float): Error margin in distance calculation (meters)
        """
        self.position = np.array(position)
        self.accuracy = accuracy
    
    def measure_distance(self, receiver_position):
        """
        Measure the distance from satellite to receiver with some error
        
        Args:
            receiver_position (numpy.array): 3D position of the receiver
            
        Returns:
            float: Measured distance with some error
        """
        true_distance = np.linalg.norm(self.position - receiver_position)
        # Add some random error to simulate measurement inaccuracy
        error = random.gauss(0, self.accuracy)
        return true_distance + error


class GPSReceiver:
    def __init__(self, true_position, satellites):
        """
        Initialize a GPS receiver
        
        Args:
            true_position (tuple): Actual (x, y, z) position in meters
            satellites (list): List of Satellite objects
        """
        self.true_position = np.array(true_position)
        self.satellites = satellites
        self.measured_distances = []
    
    def collect_satellite_data(self):
        """Collect distance measurements from all satellites"""
        self.measured_distances = []
        for satellite in self.satellites:
            distance = satellite.measure_distance(self.true_position)
            self.measured_distances.append(distance)
    
    def estimate_position(self, initial_guess=None):
        """
        Estimate the position using trilateration
        
        Args:
            initial_guess (numpy.array): Initial guess for position (optional)
            
        Returns:
            numpy.array: Estimated (x, y, z) position
        """
        if initial_guess is None:
            # Start in the middle of the coordinate system as a guess
            initial_guess = np.array([0, 0, 0])
        
        # Define the error function that we want to minimize
        def error_function(position):
            error = 0
            for i, satellite in enumerate(self.satellites):
                calculated_distance = np.linalg.norm(satellite.position - position)
                measured_distance = self.measured_distances[i]
                error += (calculated_distance - measured_distance) ** 2
            return error
        
        # Simple gradient descent to minimize the error
        learning_rate = 0.01
        max_iterations = 1000
        position_estimate = initial_guess.copy()
        
        for _ in range(max_iterations):
            # Calculate gradient numerically
            gradient = np.zeros(3)
            epsilon = 0.1
            
            for i in range(3):
                # Perturb position slightly in each dimension
                test_position = position_estimate.copy()
                test_position[i] += epsilon
                error_plus = error_function(test_position)
                
                test_position = position_estimate.copy()
                test_position[i] -= epsilon
                error_minus = error_function(test_position)
                
                # Approximate gradient
                gradient[i] = (error_plus - error_minus) / (2 * epsilon)
            
            # Update position estimate
            position_estimate = position_estimate - learning_rate * gradient
            
            # Check if error is small enough
            if error_function(position_estimate) < 0.01:
                break
        
        return position_estimate


def visualize_triangulation(satellites, true_position, estimated_position):
    """
    Visualize the satellites, true position, and estimated position
    
    Args:
        satellites (list): List of Satellite objects
        true_position (numpy.array): True 3D position of the receiver
        estimated_position (numpy.array): Estimated 3D position
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot satellites
    sat_positions = np.array([sat.position for sat in satellites])
    ax.scatter(sat_positions[:, 0], sat_positions[:, 1], sat_positions[:, 2], 
               c='blue', marker='^', s=100, label='Satellites')
    
    # Plot true position
    ax.scatter(true_position[0], true_position[1], true_position[2],
               c='green', marker='o', s=100, label='True Position')
    
    # Plot estimated position
    ax.scatter(estimated_position[0], estimated_position[1], estimated_position[2],
               c='red', marker='x', s=100, label='Estimated Position')
    
    # Draw lines from satellites to true position
    for sat_pos in sat_positions:
        ax.plot([sat_pos[0], true_position[0]],
                [sat_pos[1], true_position[1]],
                [sat_pos[2], true_position[2]], 'k--', alpha=0.3)
    
    # Draw spheres representing distance measurements
    # (simplified to circles at the satellite positions)
    for i, sat in enumerate(satellites):
        distance = np.linalg.norm(sat.position - true_position)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = distance * np.cos(u) * np.sin(v) + sat.position[0]
        y = distance * np.sin(u) * np.sin(v) + sat.position[1]
        z = distance * np.cos(v) + sat.position[2]
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
    
    # Calculate error
    error = np.linalg.norm(true_position - estimated_position)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'GPS Triangulation Simulation\nPosition Error: {error:.2f} meters')
    ax.legend()
    
    # Make the plot more balanced
    max_range = np.array([sat_positions[:, 0].max() - sat_positions[:, 0].min(),
                         sat_positions[:, 1].max() - sat_positions[:, 1].min(),
                         sat_positions[:, 2].max() - sat_positions[:, 2].min()]).max() / 2.0
    
    mid_x = (sat_positions[:, 0].max() + sat_positions[:, 0].min()) * 0.5
    mid_y = (sat_positions[:, 1].max() + sat_positions[:, 1].min()) * 0.5
    mid_z = (sat_positions[:, 2].max() + sat_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()


def simulate_gps_movement():
    """Run a dynamic simulation of a moving GPS receiver"""
    # Create satellites at fixed positions
    satellites = [
        Satellite((20000, 0, 20000), accuracy=50),
        Satellite((-20000, 0, 20000), accuracy=80),
        Satellite((0, 25000, 20000), accuracy=60),
        Satellite((0, -15000, 20000), accuracy=70)
    ]
    
    # Create a moving path for the receiver
    steps = 50
    t = np.linspace(0, 10, steps)
    x = 2000 * np.cos(t)
    y = 3000 * np.sin(t)
    z = 100 + t * 50
    
    # Prepare figure for animation
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    true_positions = []
    estimated_positions = []
    
    # Run simulation
    for i in range(steps):
        true_position = np.array([x[i], y[i], z[i]])
        true_positions.append(true_position)
        
        # Create receiver and estimate position
        receiver = GPSReceiver(true_position, satellites)
        receiver.collect_satellite_data()
        estimated_position = receiver.estimate_position()
        estimated_positions.append(estimated_position)
        
        # Clear and update plot
        ax.clear()
        
        # Plot satellites
        sat_positions = np.array([sat.position for sat in satellites])
        ax.scatter(sat_positions[:, 0], sat_positions[:, 1], sat_positions[:, 2], 
                   c='blue', marker='^', s=100, label='Satellites')
        
        # Plot true path and position
        true_path = np.array(true_positions)
        ax.scatter(true_position[0], true_position[1], true_position[2],
                  c='green', marker='o', s=100, label='True Position')
        ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2], 'g-', alpha=0.7)
        
        # Plot estimated path and position
        estimated_path = np.array(estimated_positions)
        ax.scatter(estimated_position[0], estimated_position[1], estimated_position[2],
                  c='red', marker='x', s=100, label='Estimated Position')
        
        if i > 0:  # Only plot once we have two points
            ax.plot(estimated_path[:, 0], estimated_path[:, 1], estimated_path[:, 2], 'r--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        
        # Calculate current error
        error = np.linalg.norm(true_position - estimated_position)
        ax.set_title(f'Moving GPS Simulation - Step {i+1}/{steps}\nCurrent Error: {error:.2f} meters')
        
        ax.legend()
        
        # Set consistent scale
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.set_zlim(0, 1000)
        
        plt.pause(0.05)  # Short pause to update the display
    
    plt.tight_layout()
    plt.show()


def main():
    print("GPS Triangulation Simulation")
    print("1. Static triangulation demonstration")
    print("2. Moving receiver simulation")
    choice = input("Select an option (1/2): ")
    
    if choice == "1":
        # Create satellites at different positions
        satellites = [
            Satellite((10000, 0, 20000)),
            Satellite((-8000, 5000, 21000)),
            Satellite((2000, 12000, 18000)),
            Satellite((-3000, -7000, 22000))
        ]
        
        # True position of the receiver
        true_position = np.array([500, 600, 200])
        
        # Create receiver and collect data
        receiver = GPSReceiver(true_position, satellites)
        receiver.collect_satellite_data()
        
        # Estimate position
        print("True position:", true_position)
        print("Collecting satellite measurements...")
        time.sleep(1)
        
        estimated_position = receiver.estimate_position()
        print("Estimated position:", estimated_position)
        
        # Calculate error
        error = np.linalg.norm(true_position - estimated_position)
        print(f"Position error: {error:.2f} meters")
        
        # Visualize
        visualize_triangulation(satellites, true_position, estimated_position)
    
    elif choice == "2":
        simulate_gps_movement()
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main() 