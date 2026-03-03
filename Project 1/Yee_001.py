import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML

################################################################################################################################################
#                                                           Parameters:                                                       
################################################################################################################################################
nx, ny = 100, 100  # Number of grid points in x,y direction
L = 1.0  # Length of the domain in meters
dx = np.full(nx-1, L/(nx-1))  # Spacing between Ex nodes
dy = np.full(ny-1, L/(ny-1))  # Spacing between Ey nodes
dx_d = (dx[:-1] + dx[1:]) / 2.0
dy_d = (dy[:-1] + dy[1:]) / 2.0
# Add the "Half-Cells" at the boundaries to make them (nx,1) and (1,ny) for the update equations
dx_d = np.concatenate(([dx[0]/2], (dx[:-1] + dx[1:])/2, [dx[-1]/2])) # length 100
dy_d = np.concatenate(([dy[0]/2], (dy[:-1] + dy[1:])/2, [dy[-1]/2])) # length 100

c = 3e8  # Speed of light in vacuum (m/s)

epsilon0 = 8.854e-12  # Permittivity of free space (F/m)
eps_z = epsilon0 * np.ones((nx,ny))  # Permittivity array (F/m)
eps_z[int(nx/2):-1, int(ny/2):-1]*=2


mu0 = 4*np.pi*1e-7  # Permeability of free space (H/m)
mu = mu0 * np.ones((nx, ny))  # Permeability array (H/m)
mu_x = (mu[:, :-1] + mu[:, 1:]) / 2.0
mu_y = (mu[:-1, :] + mu[1:, :]) / 2.0

sigma = np.zeros((nx, ny))  # Conductivity array (S/m)



CFL = 1  # Courant-Friedrichs-Lewy number (preferably as close to 1 as possible for stability/accuracy)
dt = CFL/(c*np.sqrt((1/np.min(dx)**2)+(1/np.min(dy)**2))) # Time step (s)
nt = 1000  # Number of time steps

# Some source parameters
A = 1.0  # Amplitude of the source
fc = 1e9  # Frequency of the source (Hz)
sig = 1/(2*fc)  # Standard deviation of the source (s)
t0 = 3*sig  # Time delay of the source (s)
    
# Source position           # TEMPORARY: source & recorder not matched to grid!!!
x0 = int(nx/2)  # Source x position (grid index)
y0 = int(ny/2)  # Source y position (grid index)
# Recorder position
x1 = int(nx/4) # Recorder x position (grid index)
y1 = int(ny/2) # Recorder y position (grid index)

# Constants for update equations
C_ez = (1 / (dx_d[:, None] * dy_d[None, :]) ) / (eps_z/dt - sigma/2)
S_ez = (eps_z/dt - sigma/2) / (eps_z/dt - sigma/2)
C_hx = (dt * dx_d[:, None]) / (dy[None, :] * mu_x) 
C_hy = (dt * dy_d[None, :]) / (dx[:, None] * mu_y)


ant = input("Do you want to see the Yee Simulation?: ")
boolse=(ant.lower() == 'y')
if boolse:
    # Initialize the staggered fields 
    Ez = np.zeros((nx, ny))  # Electric field in z direction
    Hx = np.zeros((nx, ny))  # Magnetic field in x direction
    Hy = np.zeros((nx, ny))  # Magnetic field in y direction

    def Updater(Ez, Hx, Hy):
        # Update electric field
        Ez[1:-1, 1:-1] = S_ez[1:-1, 1:-1] * Ez[1:-1, 1:-1] + C_ez[1:-1, 1:-1] * (
                (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) - 
                (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2]))
        
        # PEC unnecessary since we are only updating the interior points
        # Ez[0, :] = 0  # PEC
        # Ez[-1, :] = 0  # PEC
        # Ez[:, 0] = 0  # PEC
        # Ez[:, -1] = 0  # PEC

        # Update magnetic fields
        Hx[:, :-1] -= C_hx * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += C_hy * (Ez[1:, :] - Ez[:-1, :])


    



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

        bron = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*((t-t0)/sig)**2)
        
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
