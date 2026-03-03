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


mu = mu0 * np.ones((nx, ny))  # Permeability array (H/m)
mu_x = (mu[:, :-1] + mu[:, 1:]) / 2.0
mu_y = (mu[:-1, :] + mu[1:, :]) / 2.0

sigma = np.zeros((nx-2, ny-2))  # Conductivity array (S/m)



CFL = 0.7  # Courant-Friedrichs-Lewy number (preferably as close to 1 as possible for stability/accuracy)
dt = CFL/(c*np.sqrt((1/np.min(dx)**2)+(1/np.min(dy)**2))) # Time step (s)
nt = 500  # Number of time steps

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


beta_xp = kappa_x / (c * dt)  + Z0 * eta_x / 2.0
beta_yp = kappa_y / (c * dt)  + Z0 * eta_y / 2.0
beta_xm = kappa_x / (c * dt)  - Z0 * eta_x / 2.0
beta_ym = kappa_y / (c * dt)  - Z0 * eta_y / 2.0
# Interpolate beta coefficients to the Hx grid (midpoints in y)
beta_yp_hx = (beta_yp[:, :-1] + beta_yp[:, 1:]) / 2.0
beta_ym_hx = (beta_ym[:, :-1] + beta_ym[:, 1:]) / 2.0
beta_yp_hy = (beta_yp[:-1, :] + beta_yp[1:, :]) / 2.0
beta_ym_hy = (beta_ym[:-1, :] + beta_ym[1:, :]) / 2.0

# Interpolate beta coefficients to the Hy grid (midpoints in x)
beta_xp_hx = (beta_xp[:, :-1] + beta_xp[:, 1:]) / 2.0
beta_xm_hx = (beta_xm[:, :-1] + beta_xm[:, 1:]) / 2.0
beta_xp_hy = (beta_xp[:-1, :] + beta_xp[1:, :]) / 2.0
beta_xm_hy = (beta_xm[:-1, :] + beta_xm[1:, :]) / 2.0

beta_z = 1.0 / (c * dt)
alpha_p = 2.0 * gamma / dt + 1.0
alpha_m = 2.0 * gamma / dt - 1.0


ant = input("Do you want to see the Yee Simulation?: ")
boolse=(ant.lower() == 'y')
if boolse:
    # Initialize the staggered fields 
    Ez = np.zeros((nx, ny))  # Electric field in z direction
    Hx = np.zeros((nx, ny-1))  # Magnetic field in x direction
    Hy = np.zeros((nx-1, ny))  # Magnetic field in y direction
    Ez_dot = np.zeros((nx, ny))  # First auxiliary electric field
    Ez_ddot = np.zeros((nx, ny))  # Second auxiliary electric field
    Jc = np.zeros((nx, ny))  # Conduction current density
    Hx_dot = np.zeros((nx, ny-1))  # Auxiliary magnetic field x
    Hy_dot = np.zeros((nx-1, ny))  # Auxiliary magnetic field y

    def Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot, Hy_dot,t,it):
        # ---- UPDATE MAGNETIC FIELDS ----:
        # Update H°x:
        Hx_dot_old = Hx_dot.copy()
        Hx_dot[:, :] = (beta_ym_hx * Hx_dot - (Ez[:, 1:] - Ez[:, :-1]) / dy[None, :]) / beta_yp_hx

        # Update Hx:
        Hx[:, :] = Hx + (beta_xp_hx * Hx_dot - beta_xm_hx * Hx_dot_old) / beta_z

        # Update H°y:
        Hy_dot_old = Hy_dot.copy() 
        Hy_dot[:, :] = Hy_dot + (Ez[1:, :] - Ez[:-1, :]) / (dx[:, None] * beta_z)

        # Update Hy:
        Hy[:, :] = (beta_xm_hy * Hy + (beta_yp_hy * Hy_dot - beta_ym_hy * Hy_dot_old)) / beta_xp_hy

        # ---- UPDATE ELECTRIC FIELD ----:
        curl_h = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx_d[1:-1, None] - \
                 (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy_d[None, 1:-1]

        # Update E°°z:
        Ez_ddot_old = Ez_ddot.copy()
        coef_n = (1.0 / dt - sigma / (2.0 * alpha_p))
        coef_p = (1.0 / dt + sigma / (2.0 * alpha_p))
        coef_j = 0.5 * (1.0 + alpha_m / alpha_p)
        Ez_ddot[1:-1, 1:-1] = (coef_n * Ez_ddot[1:-1, 1:-1] - coef_j * Jc[1:-1, 1:-1] + curl_h) / coef_p

        # add source to E°°z:
        source_val = A * np.cos(2*np.pi*fc*(t-t0)) * np.exp(-0.5*((t-t0)/sig)**2)
        Ez_ddot[x0, y0] += source_val

        # Update Jc:
        Jc[1:-1, 1:-1] = (alpha_m * Jc[1:-1, 1:-1] + \
                          sigma * (Ez_ddot[1:-1, 1:-1] + \
                                   Ez_ddot_old[1:-1, 1:-1])) / alpha_p

        # Update E°z:
        Ez_dot_old = Ez_dot.copy()
        Ez_dot[1:-1, 1:-1] = (beta_xm[1:-1, 1:-1] * Ez_dot[1:-1, 1:-1] + \
                            (Ez_ddot[1:-1, 1:-1] - Ez_ddot_old[1:-1, 1:-1]) / dt) / beta_xp[1:-1, 1:-1]

        # Update Ez:
        Ez[1:-1, 1:-1] = (beta_ym[1:-1, 1:-1] * Ez[1:-1, 1:-1] + \
                          beta_z * (Ez_dot[1:-1, 1:-1] - Ez_dot_old[1:-1, 1:-1])) / beta_yp[1:-1, 1:-1]





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
        Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot,Hy_dot,t,it)
        recorder[it] = Ez[x1,y1] # Store field at recorder

        artists = [
            ax.text(0.5,1.05,'%d/%d' % (it, nt), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, ),
            ax.imshow(Ez.T * Z0, vmin=-50*A, vmax=50*A),
            # ax.imshow(p_ref.T, vmin=-0.02*A, vmax=0.02*A),
            ax.plot(x0,y0,'ks',fillstyle="none")[0],
            ax.plot(x1,y1,'ro',fillstyle="none")[0],
            ]
        movie.append(artists)
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                    blit=True)
    plt.show()
