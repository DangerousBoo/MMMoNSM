import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.fftpack import fft2
from matplotlib.animation import FuncAnimation

################################################################################################################################################
#                                                           Parameters:                                                       
################################################################################################################################################
class SimulationConfig:
    """Handles all physical and numerical parameters."""
    def __init__(self):
        # Grid Dimensions
        self.nx, self.ny = 600, 200 # Number of grid points in x,y direction
        self.L = 200 # Length of the waveguide in grid points
        self.d = 20 # Half the width of the waveguide core in grid points
        self.x0, self.y0 = 50, self.ny // 2 # Source location
        self.x1 = self.nx - 50 # Recorder location
        self.y_gap_top = self.ny // 2 - self.d // 2 # Gridpoint of the top gap edge
        self.y_gap_bot = self.ny // 2 + self.d // 2 # Gridpoint of the bottom gap edge

        # Physical Constants
        self.c = 299792458
        self.epsilon0 = 8.854e-12
        self.mu0 = 4 * np.pi * 1e-7
        self.gamma = 0.0
        self.sigma = np.zeros((self.nx-2, self.ny-2))
        self.Z0 = np.sqrt(self.mu0 / self.epsilon0)

        # Material Properties
        self.eps_clad = 2.218
        self.eps_core = 2.22
        self.epsilon_r = np.ones((self.nx, self.ny))
        self.setup_waveguide()
        
        self.v_local = self.c / np.sqrt(self.epsilon_r)
        self.Z_local = self.Z0 / np.sqrt(self.epsilon_r)

        # Source Parameters
        self.lam_c = 1.0
        self.f_c = self.c / self.lam_c
        self.A = 1.0
        self.a = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t = self.a / (2 * np.pi * self.f_c)
        self.t0 = 4 * self.sig_t

        # Grid Spacing (Non-uniform x)
        self.dx_0 = self.lam_c / 25
        self.dy_0 = self.lam_c / 25
        self.dx_f = self.dx_0
        self.dy_f = self.dy_0
        self.dx = np.full(self.nx - 1, self.dx_0)
        self.dy = np.full(self.ny - 1, self.dy_0)
        
        # Grid Refinement Logic
        alpha   = np.sqrt(2)
        self.n_w = self.nx - self.L
        n_f     = int(np.ceil(np.log(self.dx_0 / self.dx_f) / np.log(alpha)))
        dist    = np.arange(-n_f + 1, n_f)

        if len(dist) > 0:
            self.dx[self.n_w - n_f + 1: self.n_w + n_f] = self.dx_0 * alpha ** np.abs(dist)

        self.dx_d = np.concatenate(([self.dx[0]/2], (self.dx[:-1] + self.dx[1:])/2, [self.dx[-1]/2]))
        self.dy_d = np.concatenate(([self.dy[0]/2], (self.dy[:-1] + self.dy[1:])/2, [self.dy[-1]/2]))

        # Time Stepping
        CFL = 1.0
        self.dt  = self.CFL / (self.c * np.sqrt((1/self.dx_0**2) + (1/self.dy_0**2)))
        self.nt  = 1500

    def setup_waveguide(self):
        self.L = 200
        self.x_start = int(self.nx - self.L) # Start of the waveguide in x direction
        self.y_start = int(self.ny // 2 - 2 * self.d) # Start of the waveguide in y direction
        self.epsilon_r[self.x_start:, self.y_start            : self.y_start +   self.d] = self.eps_clad
        self.epsilon_r[self.x_start:, self.y_start +   self.d : self.y_start + 3*self.d] = self.eps_core
        self.epsilon_r[self.x_start:, self.y_start + 3*self.d : self.y_start + 4*self.d] = self.eps_clad

################################################################################################################################################
#                                                          Yee Solver Class:
################################################################################################################################################
class YeeSolver:
    def __init__(self, config):
        self.cfg = config
        self.init_fields()
        self.init_pml()
        self.init_coefficients()

    def init_fields(self):
        self.Ez      = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Ez_dot  = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Ez_ddot = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Jc      = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Hx      = np.zeros((self.cfg.nx, self.cfg.ny-1))
        self.Hx_dot  = np.zeros((self.cfg.nx, self.cfg.ny-1))
        self.Hy      = np.zeros((self.cfg.nx-1, self.cfg.ny))
        self.Hy_dot  = np.zeros((self.cfg.nx-1, self.cfg.ny))

    def init_pml(self):
        p, m = 20, 4
        eta_max = (m + 1) / (150 * np.pi * self.cfg.dx_0)
        ksi_k_max = 3
        
        self.kx, self.ky = np.ones((self.cfg.nx, self.cfg.ny)), np.ones((self.cfg.nx, self.cfg.ny))
        self.etax, self.etay = np.zeros((self.cfg.nx, self.cfg.ny)), np.zeros((self.cfg.nx, self.cfg.ny))

        for i in range(p):
            d_pml = (p - i) / p
            val_k = 1.0 + (ksi_k_max - 1.0) * (d_pml**m)
            val_eta = eta_max * (d_pml**m)

            self.kx[i, :], self.kx[-1-i, :] = val_k, val_k
            self.ky[:, i], self.ky[:, -1-i] = val_k, val_k
            self.etax[i, :], self.etax[-1-i, :] = val_eta, val_eta
            self.etay[:, i], self.etay[:, -1-i] = val_eta, val_eta

    def init_coefficients(self):
        cfg = self.cfg
        # Update coefficients
        self.bxp = self.kx / (cfg.v_local * cfg.dt) + cfg.Z_local * self.etax / 2.0
        self.byp = self.ky / (cfg.v_local * cfg.dt) + cfg.Z_local * self.etay / 2.0
        self.bxm = self.kx / (cfg.v_local * cfg.dt) - cfg.Z_local * self.etax / 2.0
        self.bym = self.ky / (cfg.v_local * cfg.dt) - cfg.Z_local * self.etay / 2.0

        # Interpolations
        self.byp_hx = (self.byp[:, :-1] + self.byp[:, 1:]) / 2.0
        self.bym_hx = (self.bym[:, :-1] + self.bym[:, 1:]) / 2.0
        self.byp_hy = (self.byp[:-1, :] + self.byp[1:, :]) / 2.0
        self.bym_hy = (self.bym[:-1, :] + self.bym[1:, :]) / 2.0
        self.bxp_hx = (self.bxp[:, :-1] + self.bxp[:, 1:]) / 2.0
        self.bxm_hx = (self.bxm[:, :-1] + self.bxm[:, 1:]) / 2.0
        self.bxp_hy = (self.bxp[:-1, :] + self.bxp[1:, :]) / 2.0
        self.bxm_hy = (self.bxm[:-1, :] + self.bxm[1:, :]) / 2.0

        self.bz_h = 1.0 / (cfg.c * cfg.dt)
        self.bz_e = 1.0 / (cfg.v_local * cfg.dt)
        self.ap = 2.0 * cfg.gamma / cfg.dt + 1.0
        self.am = 2.0 * cfg.gamma / cfg.dt - 1.0

        sub_v = cfg.v_local[1:-1, 1:-1]
        sub_z = cfg.Z_local[1:-1, 1:-1]
        self.coef_n = (1.0 / (sub_v * cfg.dt) - sub_z * self.sigma / (2.0 * self.ap))
        self.coef_p = (1.0 / (sub_v * cfg.dt) + sub_z * self.sigma / (2.0 * self.ap))
        self.coef_j = 0.5 * (1.0 + self.am / self.ap)












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
        
        Ez[x_wall, :y_gap_top] = 0
        Ez[x_wall, y_gap_bot:] = 0

        source_val = A * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/sig_t)**2)
        Ez[x0,y0] -= dx[x0] * dy[y0] * source_val / coef_p[x0 - 1,y0 - 1]
        


    #---- plot of the animation ----
    timeseries = np.zeros((nt,1))
    recorder_plane = np.zeros((nt, ny))
    nodes_x = np.concatenate(([0], np.cumsum(dx)))
    nodes_y = np.concatenate(([0], np.cumsum(dy)))
    X, Y = np.meshgrid(nodes_x, nodes_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_facecolor('black') # Helps see the wave if it's faint
    v_min, v_max = -0.0001, 0.0001
    movie = []

    wall_mask = np.zeros((nx, ny))
    wall_x = int(n_w - L/2)
    # Top part of the wall
    wall_mask[wall_x, :int(ny//2 - 2*d)] = 1
    # Bottom part of the wall
    wall_mask[wall_x, int(ny//2 + 2*d):] = 1

    for it in range(nt):
        t = it * dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))
        Updater(Ez, Hx, Hy, Ez_dot, Ez_ddot, Jc, Hx_dot, Hy_dot, t)
        recorder_plane[it, :] = Ez[x1,:]  # Store field at recorder

        field_data = (Z_local * Ez).T
        quad = ax.pcolormesh(X, Y, field_data, 
                            vmin=v_min, vmax=v_max, 
                            shading='auto', cmap='RdBu_r', animated=True)

        # Plot Source and Recorder
        src, = ax.plot(nodes_x[x0], nodes_y[y0], 'wo', ms=5, fillstyle='none')
        txt = ax.text(0.5, 1.02, f'Step {it}/{nt}', transform=ax.transAxes, color='white', ha='center')
        movie.append([quad, src, txt])

    ax.set_xlim(nodes_x.min(), nodes_x.max())
    ax.set_ylim(nodes_y.min(), nodes_y.max())
    ani = ArtistAnimation(fig, movie, interval=1, blit=True)
    plt.show()

plt.plot(timeseries, recorder_plane[:,L//2])        
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

freq1, k, fft1 = freqs(recorder_plane[:, L//2], dt, c)
FreqPlot(dt,c,recorder_plane[:, L//2])


fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(nodes_y.min(), nodes_y.max())
max_int = np.max(recorder_plane**2)
start_step = 400  # Adjust this based on when the wave hits the plane
speed_delay = 100 # Higher = Slower (ms per frame). Try 100-200 for "slow motion"
ax.set_ylim(0, max_int * 1.1)

max_int = np.max(recorder_plane[start_step:, :]**2)
ax.set_ylim(0, max_int * 1.2)

line, = ax.plot([], [], color='red', lw=2)
ax.set_xlabel('Y Position (m)')
ax.set_ylabel('Intensity (E^2)')
ax.grid(True, alpha=0.3)

# Highlight the Core region for context
ax.axvspan(nodes_y[y_start], nodes_y[y_start + d], 
           color='yellow', alpha=0.1, label='Cladding')
ax.axvspan(nodes_y[y_start + d], nodes_y[y_start + 3*d], 
           color='blue', alpha=0.1, label='Core')
ax.axvspan(nodes_y[y_start + 3*d], nodes_y[y_start + 4*d], 
           color='yellow', alpha=0.1, label='Cladding')
ax.legend(loc='upper right')

title = ax.set_title('')

def animate(i):
    intensity = recorder_plane[i, :]**2
    line.set_data(nodes_y, intensity)
    
    current_time_ns = i * dt * 1e9
    title.set_text(f'Intensity Profile | t = {current_time_ns:.2f} ns (Step {i})')
    return line, title

anim_1d = FuncAnimation(fig, animate, 
                        frames=range(start_step, nt, 2), 
                        interval=speed_delay, 
                        blit=True)

plt.show()