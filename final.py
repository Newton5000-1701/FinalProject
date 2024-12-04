import numpy as np
import matplotlib.pyplot as plt

def solve_potential(grid_size, plate_positions, tol=1e-6, max_iter=10000):
    rows, cols = grid_size
    potential = np.zeros((rows, cols))
    
    # Unpack plate positions
    start_row, start_col_left, end_row, start_col_right, voltage_left, voltage_right = plate_positions
    
    # Apply boundary conditions for the plates
    potential[start_row:end_row, start_col_left] = voltage_left
    potential[start_row:end_row, start_col_right] = voltage_right
    
    # Iteratively solve using the finite difference method
    for _ in range(max_iter):
        old_potential = potential.copy()
        
        # Update interior points
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

# Parameters
grid_size = (100, 200)  # Adjusted grid for 2m x 1m
plate_positions = (40, 60, 60, 140, -1.0, 1.0)  # Adjusted to scale the plates properly

# Solve for potential
potential = solve_potential(grid_size, plate_positions)

# Plot the results
plt.figure(figsize=(8, 4))  # Adjust aspect ratio for 2:1
plt.contourf(potential, levels=50, cmap="seismic")
plt.colorbar(label="Potential (V)")
plt.title("Electric Potential in a 2m x 1m Box with a Parallel Plate Capacitor")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.gca().invert_yaxis()  # Match array indexing
plt.show()



