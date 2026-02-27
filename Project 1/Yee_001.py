import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML

# Define some parameters
c = 3e8  # Speed of light in vacuum (m/s)
dx = 0.01  # Spatial step (m)
dy = 0.01  # Spatial step (m)

CFL = 1  # Courant-Friedrichs-Lewy number (preferably as close to 1 as possible for stability/accuracy)
dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2)))

ant = input("Do you want to see the Yee Simulation?: ")
boolse=(ant.lower() == 'y')
if boolse:
    print("Tis hier nog niet te zien jong")
