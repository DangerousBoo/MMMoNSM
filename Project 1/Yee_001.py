import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML

################################################################################################################################################
#                                                           Parameters:                                                       
################################################################################################################################################
c = 3e8  # Speed of light in vacuum (m/s)
epsilon0 = 8.854e-12  # Permittivity of free space (F/m)
mu0 = 4*np.pi*1e-7  # Permeability of free space (H/m)

dx = 0.01  # Spatial step (m)
dy = 0.01  # Spatial step (m)
L = 1.0  # Length of the simulation domain (m)
nx = int(L/dx)  # Number of grid points in x direction
ny = int(L/dy)  # Number of grid points in y direction

CFL = 1  # Courant-Friedrichs-Lewy number (preferably as close to 1 as possible for stability/accuracy)
dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2))) # Time step (s)
nt = 1000  # Number of time steps

# Some source parameters
A = 1.0  # Amplitude of the source
fc = 1e9  # Frequency of the source (Hz)
sigma = 1/(2*fc)  # Standard deviation of the source (s)
t0 = 3*sigma  # Time delay of the source (s)
    
# Source position
x0 = int(nx/2)  # Source x position (grid index)
y0 = int(ny/2)  # Source y position (grid index)
# Recorder position
x1 = int(nx/4) # Recorder x position (grid index)
y1 = int(ny/2) # Recorder y position (grid index)

eps=epsilon0*np.ones((nx-2,ny-2))
eps[int(nx/2):-1, int(ny/2):-1]*=2





ant = input("Do you want to see the Yee Simulation?: ")
boolse=(ant.lower() == 'y')
if boolse:

    # Initialize the staggered fields #NOT YET STAGGERED
    Ez = np.zeros((nx, ny))  # Electric field in z direction
    Hx = np.zeros((nx, ny))  # Magnetic field in x direction
    Hy = np.zeros((nx, ny))  # Magnetic field in y direction

    def Updater(Ez, Hx, Hy):
        # Update magnetic fields
        Hx[:, :-1] -= (dt / (mu0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += (dt / (mu0 * dx)) * (Ez[1:, :] - Ez[:-1, :])
        # Update electric field
        Ez[1:-1, 1:-1] += (dt / (eps)) * (
                (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) / dx - 
                (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2]) / dy)

    



    #---- plot of the animation ----
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.xlim([1, nx+1])
    plt.ylim([1, ny+1])
    movie = []
    timeseries = np.zeros((nt,1))
    recorder = np.zeros((nt,1))
    recorder_ref = np.zeros((nt,1))
    tmax = nt

    for it in range(0, nt):
        t = (it-1)*dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))

        bron = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*((t-t0)/sigma)**2) # update source

        Ez[x0,y0] = Ez[x0,y0] + bron # add source to field
        Updater(Ez, Hx, Hy)   # propagate over dt
        recorder[it] = Ez[x1,y1] # Store field at recorder

        artists = [
            ax.text(0.5,1.05,'%d/%d' % (it, nt), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, ),
            ax.imshow(Ez.T, vmin=-0.02*A, vmax=0.02*A),
            # ax.imshow(p_ref.T, vmin=-0.02*A, vmax=0.02*A),
            ax.plot(x0,y0,'ks',fillstyle="none")[0],
            ax.plot(x1,y1,'ro',fillstyle="none")[0],
            ]
        movie.append(artists)
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                    blit=True)
    plt.show()
