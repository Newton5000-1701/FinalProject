import numpy as np
import matplotlib.pyplot as plt

def solve_potential(grid_size, plate_positions, tol=1e-6, max_iter=10000):
    rows, cols = grid_size
    potential = np.zeros((rows, cols))
    
   #Create array to store relevant parameters in 
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


plt.figure(figsize=(8, 4))  
plt.contourf(potential, levels=50, cmap="seismic")
plt.colorbar(label="Potential (V)")
plt.title("Scalar Potential in a Rectangular 2D Box with a Parallel Plate Capacitor")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)") 
plt.show()

plt.savefig('CapacitorLaplace.png') 


