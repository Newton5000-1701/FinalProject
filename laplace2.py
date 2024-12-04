#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:02:44 2024

@author: isaacthompson
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_laplace(a, b, grid_spacing=0.1, max_iter=10000, tol=1e-6):
    Nx = int(a / grid_spacing) + 1  
    Ny = int(b / grid_spacing) + 1  

    
    V = np.zeros((Ny, Nx))


    x = np.linspace(0, a, Nx)
    
    # Bottom boundary (y = 0)
    V[0, :] = np.cos(np.pi * x / a)**2

    # Top boundary (y = b) Note to self: -1 in V to use last row; DOES NOT MEAN y=-1!
    V[-1, :] = -np.sin(np.pi * x / a)**2

    # Left and Right boundaries (x = 0 and x = a) are already 0! (Dirichlet condition)

    for iteration in range(max_iter):
        V_old = V.copy()
        
        # Update the potential using the average of surrounding points
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                V[i, j] = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
        
        # Check for convergence (relative change in the potential)
        if np.max(np.abs(V - V_old)) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

    return V

# Box dimensions (in meters)
a = 2.0  
b = 1.0 

V = solve_laplace(a, b)

x = np.linspace(0, a, V.shape[1])
y = np.linspace(0, b, V.shape[0])
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(6, 6))
cp = plt.contourf(X, Y, V, 20, cmap='viridis')

cbar = plt.colorbar(cp)
cbar.set_label('Potential (V)')

plt.title("Potential Distribution From Laplace's Equation")
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.show()


plt.savefig('SinusoidalBoundariesLaplace.png') 

