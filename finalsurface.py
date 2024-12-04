#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:40:19 2024

@author: isaacthompson
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_potential(grid_size, plate_positions, tol=1e-6, max_iter=10000):
    rows, cols = grid_size
    potential = np.zeros((rows, cols))
    
    # Create array to store relevant parameters in
    start_row, start_col_left, end_row, start_col_right, voltage_left, voltage_right = plate_positions
    
    # Apply boundary conditions for the plates
    potential[start_row:end_row, start_col_left] = voltage_left
    potential[start_row:end_row, start_col_right] = voltage_right
    
    # Iteratively solve using the averaging method
    for _ in range(max_iter):
        old_potential = potential.copy()
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Skip the capacitor plates
                if (start_row <= i < end_row and (j == start_col_left or j == start_col_right)):
                    continue
                potential[i, j] = 0.25 * (
                    old_potential[i+1, j] +
                    old_potential[i-1, j] +
                    old_potential[i, j+1] +
                    old_potential[i, j-1]
                )
        
        # Check for convergence
        if np.max(np.abs(potential - old_potential)) < tol:
            print(f"Converged after {_} iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")
    
    return potential

grid_size = (100, 200)  
plate_positions = (40, 60, 60, 140, -1.0, 1.0)  

potential = solve_potential(grid_size, plate_positions)

# Create the grid for the surface plot
x = np.arange(grid_size[1])
y = np.arange(grid_size[0])
X, Y = np.meshgrid(x, y)

# Create the figure for the surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, potential, cmap="seismic", edgecolor="none")
fig.colorbar(surf, ax=ax, label="Potential (V)")

# Label the axes
ax.set_title("Scalar Potential in a Rectangular 2D Box with a Parallel Plate Capacitor")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("Potential (V)")

plt.show()
plt.savefig('CapacitorLaplaceSurface.png') 
