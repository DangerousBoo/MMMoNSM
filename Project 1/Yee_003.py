import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML
from scipy.fftpack import fft2

################################################################################################################################################
#                                                           Parameters:                                                       
################################################################################################################################################
# Some source parameters
nx, ny = 400, 200  # Number of grid points in x,y direction
n_w = 200


c = 299792458 # Speed of light in vacuum (m/s)
epsilon0 = 8.854e-12  # Permittivity of free space (F/m)
gamma = 1.0  # Scaling factor for conduction current (can be adjusted for stability
mu0 = 4*np.pi*1e-7  # Permeability of free space (H/m)
mu_r = np.ones((nx, ny))  # Permeability array (H/m)
epsilon_r = np.ones((nx, ny))
sigma = np.zeros((nx-2, ny-2))  # Conductivity array (S/m)
Z0 = np.sqrt(mu0/epsilon0)  # Impedance of free space (Ohms)

t = 150
e1 = 4.0  # e.g., Glass or dry soil
x_start = n_w - (t // 2)
x_end = n_w + (t // 2)
epsilon_r[x_start:x_end, 100:200] = e1
v_local = c / np.sqrt(epsilon_r)
Z_local = Z0 / np.sqrt(epsilon_r)

lam_c = 1 # Wavelength of the modulated sine (m)
A = 1.0  # Amplitude of the source
f_c = c/lam_c  # Frequency of the source (Hz)
a = 3 # Amount of sigmas between fc and 0 in frequency domain
sig_t = a/(2*np.pi*f_c)  # Standard deviation of the source (s)
t0 = 4*sig_t  # Time delay of the source (s)

dx_0 = lam_c / (30)
dy_0 = lam_c / (30)
dx_f = dx_0 / 4
alpha = np.sqrt(2)
n_f = int(np.ceil(np.log(dx_0/dx_f)/np.log(alpha)))


dx = np.full(nx-1, dx_0)  # Spacing between Ex nodes
dist = np.arange(-n_f + 1, n_f)
dx[n_w - n_f + 1: n_w + n_f] = dx_f * alpha ** np.abs(dist) # Widen the spacing in the middle
dy = np.full(ny-1, dy_0)  # Spacing between Ey nodes
dy_f = dy_0
dx_d = np.concatenate(([dx[0]/2], (dx[:-1] + dx[1:])/2, [dx[-1]/2])) # length 100
dy_d = np.concatenate(([dy[0]/2], (dy[:-1] + dy[1:])/2, [dy[-1]/2])) # length 100



# PML thickness in grid cells
p = 20
m = 4 # Polynomial order for scaling
eta_max = (m + 1) / (150 * np.pi * dx_0)  # Maximum stretching factor
ksi_kappa_max = 3
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


CFL = 1  # Courant-Friedrichs-Lewy number (preferably as close to 1 as possible for stability/accuracy)
dt = CFL/(c * np.sqrt((1/np.min(dx_f)**2)+(1/np.min(dy_f)**2))) # Time step (s)
nt = 1000  # Number of time steps

# s = np.linspace(0, 1e-9, 1000)  # Time array for source definition (s)
# source = lambda t : A * np.cos(2*np.pi*fc*(t-t0)) * np.exp(-0.5*((t-t0)/sig)**2)
# plt.plot(s, source(s))
# plt.show()

# Source
x0 = int(50)  # Source x position (grid index)
y0 = int(ny/2)  # Source y position (grid index)
# Recorder position
x1 = int(350) # Recorder x position (grid index)
y1 = int(ny/2) # Recorder y position (grid index)

beta_xp = kappa_x / (v_local * dt) + Z_local * eta_x / 2.0
beta_yp = kappa_y / (v_local * dt) + Z_local * eta_y / 2.0
beta_xm = kappa_x / (v_local * dt) - Z_local * eta_x / 2.0
beta_ym = kappa_y / (v_local * dt) - Z_local * eta_y / 2.0
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

beta_z_h = 1.0 / (c * dt)
beta_z_e = 1.0 / (v_local * dt)
alpha_p = 2.0 * gamma / dt + 1.0
alpha_m = 2.0 * gamma / dt - 1.0


ant = input("Do you want to see the Yee Simulation?: ")
boolse=(ant.lower() == 'y')
if boolse:
    # Initialize the staggered fields 
    Ez, Ez_dot, Ez_ddot = np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny))  # E in z direction
    Hx, Hx_dot = np.zeros((nx, ny-1)), np.zeros((nx, ny-1))  # Magnetic field in x direction
    Hy, Hy_dot = np.zeros((nx-1, ny)), np.zeros((nx-1, ny))  # Magnetic field in y direction
    Jc = np.zeros((nx, ny))  # Conduction current density

    def Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot, Hy_dot,t):
        # ---- UPDATE MAGNETIC FIELDS ----:
        # Update H°x:
        Hx_dot_old = Hx_dot.copy()
        Hx_dot[:, :] = (beta_ym_hx * Hx_dot - (Ez[:, 1:] - Ez[:, :-1]) / dy[None, :]) / beta_yp_hx

        # Update Hx:
        Hx[:, :] = Hx + (beta_xp_hx * Hx_dot - beta_xm_hx * Hx_dot_old) / beta_z_h

        # Update H°y:
        Hy_dot_old = Hy_dot.copy() 
        Hy_dot[:, :] = Hy_dot + (Ez[1:, :] - Ez[:-1, :]) / (dx[:, None] * beta_z_h)

        # Update Hy:
        Hy[:, :] = (beta_xm_hy * Hy + (beta_yp_hy * Hy_dot - beta_ym_hy * Hy_dot_old) ) / beta_xp_hy




        # ---- UPDATE ELECTRIC FIELD ----:
        curl_h = (Hy[1:,1:-1] - Hy[:-1,1:-1]) / dx_d[1:-1,None] \
                - (Hx[1:-1,1:] - Hx[1:-1,:-1]) / dy_d[None,1:-1]
        
        # Update E°°z:
        Ez_ddot_old = Ez_ddot.copy()
        coef_n = (1.0 / (v_local[1:-1, 1:-1] * dt) - Z_local[1:-1, 1:-1] * sigma / (2.0 * alpha_p))
        coef_p = (1.0 / (v_local[1:-1, 1:-1] * dt) + Z_local[1:-1, 1:-1] * sigma / (2.0 * alpha_p))
        coef_j = 0.5 * (1.0 + alpha_m / alpha_p)
        Ez_ddot[1:-1, 1:-1] = (coef_n * Ez_ddot[1:-1, 1:-1] - coef_j * Jc[1:-1, 1:-1] + curl_h) / coef_p
        
        # Update Jc:
        Jc[1:-1, 1:-1] = (alpha_m * Jc[1:-1, 1:-1] + \
                          sigma * Z_local[1:-1, 1:-1] * (Ez_ddot[1:-1, 1:-1] + Ez_ddot_old[1:-1, 1:-1])) / alpha_p
        
        # Update E°z:
        Ez_dot_old = Ez_dot.copy()
        Ez_dot[1:-1, 1:-1] = (beta_xm[1:-1, 1:-1] * Ez_dot[1:-1, 1:-1] + \
                            (Ez_ddot[1:-1, 1:-1] - Ez_ddot_old[1:-1, 1:-1]) / (v_local[1:-1, 1:-1] * dt)) / beta_xp[1:-1, 1:-1]

        # Update Ez:
        Ez[1:-1, 1:-1] = (beta_ym[1:-1, 1:-1] * Ez[1:-1, 1:-1] + \
                          beta_z_e[1:-1, 1:-1] * (Ez_dot[1:-1, 1:-1] - Ez_dot_old[1:-1, 1:-1])) / beta_yp[1:-1, 1:-1]
        
        source_val = A * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/sig_t)**2)
        Ez[x0,y0] -= dx[x0] * dy[y0] * source_val / coef_p[x0,y0]
        
        # Ez[n_w,:ny//2 - 30] = 0  # PEC
        # Ez[n_w,ny//2 -20 : ny//2 + 21] = 0  # PEC
        # Ez[n_w,ny//2 + 31:] = 0  # PEC

    #---- plot of the animation ----
    timeseries = np.zeros((nt,1))
    recorder = np.zeros((nt,1))
    nodes_x = np.concatenate(([0], np.cumsum(dx)))
    nodes_y = np.concatenate(([0], np.cumsum(dy)))
    X, Y = np.meshgrid(nodes_x, nodes_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_facecolor('black') # Helps see the wave if it's faint
    v_min, v_max = -0.0001, 0.0001
    movie = []

    for it in range(nt):
        t = it * dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))
        Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot, Hy_dot, t)
        recorder[it] = Ez[x1, y1]  # Store field at recorder

        field_data = (Z_local * Ez).T
        quad = ax.pcolormesh(X, Y, field_data, 
                            vmin=v_min, vmax=v_max, 
                            shading='auto', cmap='RdBu_r', animated=True)
        
        # Plot Source and Recorder
        src, = ax.plot(nodes_x[x0], nodes_y[y0], 'wo', ms=5, fillstyle='none')
        rec, = ax.plot(nodes_x[x1], nodes_y[y1], 'yo', ms=5, fillstyle='none')
        txt = ax.text(0.5, 1.02, f'Step {it}/{nt}', transform=ax.transAxes, color='white', ha='center')
        movie.append([quad, src, rec, txt])

    ax.set_xlim(nodes_x.min(), nodes_x.max())
    ax.set_ylim(nodes_y.min(), nodes_y.max())
    ani = ArtistAnimation(fig, movie, interval=1, blit=True)
    plt.show()
plt.plot(timeseries, recorder)        
plt.show()

def freqs(recorder, dt, c):
    # number of samples
    n = len(recorder)
    n_zero_pad = 10*n
    fs = 1.0 / dt
    df = fs / (n+n_zero_pad)
    freq = np.arange(n+n_zero_pad) * df
    freq[0] = 1e-5

    # wavenumber
    k = 2 * np.pi * freq / c

    # FFTs
    fft  = np.fft.fft(recorder.flatten(), n+n_zero_pad)
    return freq, k, fft


def FreqPlot(dt,c,recorders):
        freq, k, fft1 = freqs(recorders, dt, c)
        plt.plot(2*np.pi/c*freq, np.abs(fft1))
        plt.title(f'Recorder 1')
        plt.ylabel('|E(k)|')
        plt.grid(True)
        plt.axvline(f_c, linestyle=':', linewidth=1)
        plt.xlim(0, 2*np.pi/c*2*f_c)
        plt.xlabel(r'$kd$ []')
        plt.tight_layout()
        plt.show()

        return freq

freq1, k, fft1 = freqs(recorder, dt, c)
FreqPlot(dt,c,recorder)
