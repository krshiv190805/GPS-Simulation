# GPS Triangulation Simulation

This Python program simulates how GPS triangulation works using multiple satellite signals to determine a receiver's position in 3D space.

## Features

- Static triangulation demonstration: Shows how a fixed position is determined using satellite signals
- Moving receiver simulation: Animates a receiver moving along a path with real-time position estimation
- 3D visualization using Matplotlib
- Simulated measurement errors to demonstrate real-world GPS accuracy issues

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install the required packages using:
```
pip install numpy matplotlib
```

## Usage

Run the program with:
```
python gps_triangulation_simulation.py
```

When prompted, select one of the following options:
1. Static triangulation demonstration - Shows how a single position is determined
2. Moving receiver simulation - Shows a receiver moving along a path with position estimates

## How GPS Triangulation Works

GPS triangulation (more accurately called trilateration) works by:

1. Multiple satellites broadcast signals with their position and precise time
2. The receiver measures the time delay from each satellite signal
3. Using the speed of light, the receiver calculates its distance from each satellite
4. Each satellite-to-receiver distance creates a sphere of possible positions
5. The intersection of 3+ spheres pinpoints the receiver's position
6. In this simulation, we use gradient descent to find the optimal position that satisfies all distance measurements

This simulation demonstrates these principles with the following components:
- Satellite class: Represents GPS satellites with fixed positions
- GPSReceiver class: Collects distance measurements and calculates position
- Visualization functions: Show the triangulation process in 3D space 
