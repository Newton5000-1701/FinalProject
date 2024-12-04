import numpy as np
import matplotlib.pyplot as plt

def solve_laplace(a, b, grid_spacing=0.1, max_iter=10000, tol=1e-6):
    # Define the grid size based on the box dimensions and grid spacing
    Nx = int(a / grid_spacing) + 1  # Number of grid points in x-direction
    Ny = int(b / grid_spacing) + 1  # Number of grid points in y-direction

    # Create a grid to store the potential values
    V = np.zeros((Ny, Nx))

    # Set the boundary conditions: V = 0 on three sides and V = 1 on the top side
    V[0, :] = 1.0  # Top boundary: V = 1 (at y = b)
    # V = 0 on other boundaries (already initialized)

    # Iterative solver using the finite difference method
    for iteration in range(max_iter):
        V_old = V.copy()
        
        # Update the potential using the finite difference approximation
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                V[i, j] = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
        
        # Check for convergence (relative change in the potential)
        if np.max(np.abs(V - V_old)) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

    return V

# Box dimensions (in meters)
a = 2.0  # Length in x-direction (meters)
b = 1.0  # Length in y-direction (meters)

# Solve Laplace's equation
V = solve_laplace(a, b)

# Create a meshgrid for plotting
x = np.linspace(0, a, V.shape[1])
y = np.linspace(0, b, V.shape[0])
X, Y = np.meshgrid(x, y)

# Plot the result using a filled contour plot
plt.figure(figsize=(6, 6))
cp = plt.contourf(X, Y, V, 20, cmap='viridis')

# Add a color bar with the correct label
cbar = plt.colorbar(cp)
cbar.set_label('Potential (V)')

# Add axis labels with correct units
plt.title("Potential Distribution (Laplace's Equation)")
plt.xlabel('x (m)')
plt.ylabel('y (m)')

# Display the plot
plt.show()


plt.savefig('ConstantBoundariesLaplace.png') 

