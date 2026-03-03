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



# PML thickness in grid cells
p = 2
m = 4 # Polynomial order for scaling
eta_max = (m + 1) / (150 * np.pi * dx[0])  # Maximum stretching factor
ksi_kappa_max = 3.0



kappa_x = np.ones((nx, ny))
kappa_y = np.ones((nx, ny))
eta_x = np.zeros((nx, ny))
eta_y = np.zeros((nx, ny))

for i in range(p):
    d = (p - i) / p  # Normalized dist.
    val_k = 1.0 + (ksi_kappa_max - 1.0) * (d**m)
    val_eta = eta_max * (d**m)
    
    # Left/Right boundaries (x-stretching)
    kappa_x[i, :], kappa_x[nx-1-i, :] = val_k, val_k
    eta_x[i, :], eta_x[nx-1-i, :] = val_eta, val_eta
    
    # Top/Bottom boundaries (y-stretching)
    kappa_y[:, i], kappa_y[:, ny-1-i] = val_k, val_k
    eta_y[:, i], eta_y[:, ny-1-i] = val_eta, val_eta






c = 3e8  # Speed of light in vacuum (m/s)
epsilon0 = 8.854e-12  # Permittivity of free space (F/m)
mu0 = 4*np.pi*1e-7  # Permeability of free space (H/m)
Z0 = np.sqrt(mu0/epsilon0)  # Impedance of free space (Ohms)
gamma = 1.0  # Scaling factor for conduction current (can be adjusted for stability

eps_z = epsilon0 * np.ones((nx,ny))  # Permittivity array (F/m)
eps_z[int(nx/2):-1, int(ny/2):-1]*=2

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
    Ez_dot = np.zeros((nx, ny))  # First auxiliary electric field
    Ez_ddot = np.zeros((nx, ny))  # Second auxiliary electric field
    Jc = np.zeros((nx, ny))  # Conduction current density
    Hx_dot = np.zeros((nx, ny))  # Auxiliary magnetic field x
    Hy_dot = np.zeros((nx, ny))  # Auxiliary magnetic field y

    def Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot, Hy_dot):
        # --- 1. Update jc (Conduction current density) ---
        # Formula: (c0*gamma/dt + 0.5)*jc_new = (c0*gamma/dt - 0.5)*jc_old + Z0*sigma*ez_ddot
        Jc[:, :] = (1.0 / ((c * gamma / dt) + 0.5)) * (
            ((c * gamma / dt) - 0.5) * Jc + Z0 * sigma * Ez_ddot
        )

        # --- 2. Update ez_ddot (Second auxiliary electric field) ---
        # Formula: (ez_ddot_new - ez_ddot_old)/dt + jc = (dhy/dx - dhx/dy)
        # Target shape: (nx-2, ny-2) for interior nodes
        Ez_ddot_old = Ez_ddot.copy()
        Ez_ddot[1:-1, 1:-1] += dt * (
            (Hy[1:-1, 1:-1] - Hy[1:-1, 1:-1]).T / dx_d[1:-1, None] - 
            (Hx[1:-1, 1:-1] - Hx[1:-1, 1:-1]) / dy_d[None, 1:-1].T - Jc[1:-1, 1:-1]
        )

        # --- 3. Update ez_dot (First auxiliary electric field) ---
        # Formula: kappa_x/dt(ez_dot_new - ez_dot_old) + Z0*eta_x/2(ez_dot_new + ez_dot_old) = 1/dt(ez_ddot_new + ez_ddot_old)
        Ez_dot_old = Ez_dot.copy()
        Ez_dot[:, :] = (1.0 / (kappa_x / dt + Z0 * eta_x / 2.0)) * (
            (kappa_x / dt - Z0 * eta_x / 2.0) * Ez_dot_old + (1.0 / dt) * (Ez_ddot + Ez_ddot_old)
        )

        # --- 4. Update ez (Primary electric field) ---
        # Formula: kappa_y/dt(ez_new - ez_old) + Z0*eta_y/2(ez_new + ez_old) = 1/dt(ez_dot_new - ez_dot_old)
        Ez_old = Ez.copy()
        Ez[:, :] = (1.0 / (kappa_y / dt + Z0 * eta_y / 2.0)) * (
            (kappa_y / dt - Z0 * eta_y / 2.0) * Ez_old + (1.0 / dt) * (Ez_dot - Ez_dot_old)
        )

        # --- 5. Update hx_dot (Auxiliary magnetic field x) ---
        # Formula: kappa_y/dt(hx_dot_new - hx_dot_old) + Z0*eta_y/2(hx_dot_new + hx_dot_old) = -dez/dy
        Hx_dot_old = Hx_dot.copy()
        Hx_dot[:, :-1] = (1.0 / (kappa_y[:, :-1] / dt + Z0 * eta_y[:, :-1] / 2.0)) * (
            (kappa_y[:, :-1] / dt - Z0 * eta_y[:, :-1] / 2.0) * Hx_dot_old[:, :-1] - 
            (Ez[:, 1:] - Ez[:, :-1]) / dy[None, :]
        )

        # --- 6. Update hy_dot (Auxiliary magnetic field y) ---
        # Formula: (hy_dot_new - hy_dot_old)/dt = -dez/dx
        Hy_dot_old = Hy_dot.copy()
        Hy_dot[:-1, :] -= (dt / 1.0) * ((Ez[1:, :] - Ez[:-1, :]) / dx[:, None])

        # --- 7. Update hx (Primary magnetic field x) ---
        # Formula: (hx_new - hx_old)/dt = kappa_x/dt(hx_dot_new - hx_dot_old) + Z0*eta_x/2(hx_dot_new + hx_dot_old)
        Hx[:, :-1] += (kappa_x[:, :-1] / dt) * (Hx_dot[:, :-1] - Hx_dot_old[:, :-1]) * dt + \
                    (Z0 * eta_x[:, :-1] / 2.0) * (Hx_dot[:, :-1] + Hx_dot_old[:, :-1]) * dt

        # --- 8. Update hy (Primary magnetic field y) ---
        # Formula: kappa_x/dt(hy_new - hy_old) + Z0*eta_x/2(hy_new + hy_old) = kappa_y/dt(hy_dot_new - hy_dot_old) + Z0*eta_y/2(hy_dot_new + hy_dot_old)
        Hy[:-1, :] = (1.0 / (kappa_x[:-1, :] / dt + Z0 * eta_x[:-1, :] / 2.0)) * (
            (kappa_x[:-1, :] / dt - Z0 * eta_x[:-1, :] / 2.0) * Hy[:-1, :] + 
            (kappa_y[:-1, :] / dt) * (Hy_dot[:-1, :] - Hy_dot_old[:-1, :]) + 
            (Z0 * eta_y[:-1, :] / 2.0) * (Hy_dot[:-1, :] + Hy_dot_old[:-1, :])
        )

    



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
        Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot,Hy_dot)   # propagate over dt
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
