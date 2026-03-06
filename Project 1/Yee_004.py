import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

################################################################################################################################################
#                                                             Parameters:                                                       
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
        self.c          = 299792458
        self.epsilon0   = 8.854e-12
        self.mu0        = 4 * np.pi * 1e-7
        self.gamma      = 0.0
        self.Z0         = np.sqrt(self.mu0 / self.epsilon0)

        # Material Properties
        self.eps_clad   = 2.218
        self.eps_core   = 2.22
        self.epsilon_r  = np.ones((self.nx, self.ny))
        self.sigma      = np.zeros((self.nx-2, self.ny-2))
        self.setup_waveguide()
        
        self.v_local = self.c / np.sqrt(self.epsilon_r)
        self.Z_local = self.Z0 / np.sqrt(self.epsilon_r)

        # Source Parameters
        self.lam_c  = 1.0
        self.f_c    = self.c / self.lam_c
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t

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
        self.dt  = CFL / (self.c * np.sqrt((1/self.dx_0**2) + (1/self.dy_0**2)))
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
        self.coef_n = (1.0 / (sub_v * cfg.dt) - sub_z * cfg.sigma / (2.0 * self.ap))
        self.coef_p = (1.0 / (sub_v * cfg.dt) + sub_z * cfg.sigma / (2.0 * self.ap))
        self.coef_j = 0.5 * (1.0 + self.am / self.ap)

    def step(self, t):
        cfg = self.cfg
        # Magnetic Field Update
        Hx_dot_old = self.Hx_dot.copy()
        self.Hx_dot = (self.bym_hx * self.Hx_dot - (self.Ez[:, 1:] - self.Ez[:, :-1]) / cfg.dy[None, :]) / self.byp_hx
        self.Hx += (self.bxp_hx * self.Hx_dot - self.bxm_hx * Hx_dot_old) / self.bz_h

        Hy_dot_old = self.Hy_dot.copy()
        self.Hy_dot += (self.Ez[1:, :] - self.Ez[:-1, :]) / (cfg.dx[:, None] * self.bz_h)
        self.Hy = (self.bxm_hy * self.Hy + (self.byp_hy * self.Hy_dot - self.bym_hy * Hy_dot_old)) / self.bxp_hy

        # Electric Field Update
        curl_h = (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / cfg.dx_d[1:-1, None] - \
                 (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / cfg.dy_d[None, 1:-1]
        
        Ez_ddot_old = self.Ez_ddot.copy()
        self.Ez_ddot[1:-1, 1:-1] = (self.coef_n * self.Ez_ddot[1:-1, 1:-1] - self.coef_j * self.Jc[1:-1, 1:-1] + curl_h) / self.coef_p
        
        self.Jc[1:-1, 1:-1] = (self.am * self.Jc[1:-1, 1:-1] + cfg.sigma * cfg.Z_local[1:-1, 1:-1] * (self.Ez_ddot[1:-1, 1:-1] + Ez_ddot_old[1:-1, 1:-1])) / self.ap
        
        Ez_dot_old = self.Ez_dot.copy()
        self.Ez_dot[1:-1, 1:-1] = (self.bxm[1:-1, 1:-1] * self.Ez_dot[1:-1, 1:-1] + 
                                   (self.Ez_ddot[1:-1, 1:-1] - Ez_ddot_old[1:-1, 1:-1]) / (cfg.v_local[1:-1, 1:-1] * cfg.dt)) / self.bxp[1:-1, 1:-1]

        self.Ez[1:-1, 1:-1] = (self.bym[1:-1, 1:-1] * self.Ez[1:-1, 1:-1] + 
                               self.bz_e[1:-1, 1:-1] * (self.Ez_dot[1:-1, 1:-1] - Ez_dot_old[1:-1, 1:-1])) / self.byp[1:-1, 1:-1]
        
        # Boundary Conditions (Wall)
        self.Ez[cfg.n_w, :cfg.y_gap_top] = 0
        self.Ez[cfg.n_w, cfg.y_gap_bot:] = 0

        # Source
        src = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        self.Ez[cfg.x0, cfg.y0] -= cfg.dx[cfg.x0] * cfg.dy[cfg.y0] * src / self.coef_p[cfg.x0-1, cfg.y0-1]

################################################################################################################################################
#                                                             Animation:
################################################################################################################################################
def run_simulation():
    config = SimulationConfig()
    solver = YeeSolver(config)
    
    history_frames = config.nt // 3
    field_history = np.zeros((history_frames, config.ny, config.nx))
    recorder_plane = np.zeros((config.nt, config.ny))
    timeseries = np.linspace(0, config.nt * config.dt, config.nt)

    show_yee = input("Do you want to see the Yee Simulation? (y/n): ").lower() == 'y'

    if show_yee:
        nodes_x = np.concatenate(([0], np.cumsum(config.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(config.dy)))
        X, Y = np.meshgrid(nodes_x, nodes_y)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        movie = []
        
        for it in tqdm(range(config.nt), desc="Running Simulation"):
            t = it * config.dt
            solver.step(t)
            if it % 5 == 0:
                field_history[it // 5] = (config.Z_local * solver.Ez).T
            recorder_plane[it, :] = solver.Ez[config.x1, :]


        quad = ax.pcolormesh(X, Y, field_history[0], shading='auto', cmap='RdBu_r', vmin=-0.0001, vmax=0.0001)
        txt = ax.text(0.5, 1.02, '', transform=ax.transAxes, color='white', ha='center')

        def update_frame(i):
            quad.set_array(field_history[i].ravel())
            current_step = i * 5
            txt.set_text(f'Step {current_step}/{config.nt} | Time: {current_step * config.dt * 1e9:.2f} ns')
            
            return quad, txt
        
        # Interval=20 means 50 frames per second. Adjust as needed.
        ani = FuncAnimation(fig, update_frame, frames=history_frames, interval=50, blit=True)
        
        plt.tight_layout()
        plt.show()

    else:
        # Run silently
        for it in tqdm(range(config.nt), desc="Running Simulation"):
            solver.step(it * config.dt)
            recorder_plane[it, :] = solver.Ez[config.x1, :]

    return config, timeseries, recorder_plane

# --- Execution ---
if __name__ == "__main__":
    cfg, times, recorder = run_simulation()
    
    # 1D plot at center of plane
    plt.figure()
    plt.plot(times, recorder[:, cfg.ny // 2])
    plt.title("Field at Recorder Plane (Center)")
    plt.show()

    # Intensity Animation
    fig, ax = plt.subplots(figsize=(8, 4))
    nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
    max_int = np.max(recorder**2)
    line, = ax.plot([], [], color='red', lw=2)
    
    ax.axvspan(nodes_y[cfg.y_start], nodes_y[cfg.y_start + cfg.d], color='yellow', alpha=0.1, label='Cladding')
    ax.axvspan(nodes_y[cfg.y_start + cfg.d], nodes_y[cfg.y_start + 3*cfg.d], color='blue', alpha=0.1, label='Core')
    ax.axvspan(nodes_y[cfg.y_start + 3*cfg.d], nodes_y[cfg.y_start + 4*cfg.d], color='yellow', alpha=0.1, label='Cladding')
    ax.legend()
    
    def animate_intensity(i):
        line.set_data(nodes_y, recorder[i, :]**2)
        ax.set_title(f'Intensity Profile | t = {i * cfg.dt * 1e9:.2f} ns')
        return line,

    ani_int = FuncAnimation(fig, animate_intensity, frames=range(250, cfg.nt, 2), interval=100)
    plt.show()