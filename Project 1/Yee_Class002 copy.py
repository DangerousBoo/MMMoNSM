import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.special as sp_special
from scipy.fft import fft, fftfreq

################################################################################################################################################
#                                                             Parameters:                                                       
################################################################################################################################################
class SimulationConfig:
    """Handles all physical and numerical parameters."""
        
    def __init__(self, **kwargs):
        
        # Physical Constants
        self.c          = 299792458
        self.epsilon0   = 8.854e-12
        self.mu0        = 4 * np.pi * 1e-7
        self.gamma      = 0.0
        self.Z0         = np.sqrt(self.mu0 / self.epsilon0)
        
        # Material Properties
        self.eps_clad   = 2.218**2
        self.eps_core   = 2.22**2

        # Source Parameters
        self.lam_c  = 1.0
        self.f_c    = self.c / self.lam_c
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t
        
        # Dimensions expressed in amount of wavelengths
        self.L_wg   = 10 * self.lam_c # Length of the waveguide in meters
        self.w_core = 3 * self.lam_c # Width of the core in meters
        self.w_clad = 3 * self.lam_c # Width of the cladding on each side in meters
        self.w_air  = 1 * self.lam_c # Width of the air region on each side next to the cladding in meters
        self.d      = 4 * self.lam_c # Distance between source and the barrier in meters
        self.t_m    = 0.01 * self.lam_c # Thickness of the barrier infront of the waveguide
        self.Ll     = 3 * self.lam_c # Length of the left region before the source in meters
        self.Lr     = 1 * self.lam_c # Length of the right region after the waveguide in meters
        self.L      = self.Ll + self.d + self.t_m + self.L_wg + self.Lr # Total length of the simulation domain in x direction in meters
        self.W      = 2 * self.w_air + 2 * self.w_clad + self.w_core # Total width of the simulation domain in y direction in meters
        
        # Grid refinement
        self.finesse = 30 # Dictates how many cells per central wavelength
        
        # Give possibility to change parameters via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.wg_type = getattr(self, "wg_type", "step") # Default waveguide type is "step"

        self.alpha = np.sqrt(2)
        
        # Build Grids
        self._build_dx()
        self._build_dy()
        
        self.nx = len(self.dx) + 1
        self.ny = len(self.dy) + 1
        
        self.epsilon_r  = np.ones((self.nx, self.ny))
        self.sigma      = np.zeros((self.nx-2, self.ny-2))
        
        self.v_local = self.c / np.sqrt(self.epsilon_r)
        self.Z_local = self.Z0 / np.sqrt(self.epsilon_r)
        
        # Grid Spacing (Non-uniform)
        self.dx_f = self.dx.min()
        self.dy_f = self.dy.min()
        
        self.x0, self.y0 = self.n_Ll, self.ny // 2 # Source location
        self.x1 = self.nx - 50 # Recorder location

        self.dx_d = np.concatenate(([self.dx[0]/2], (self.dx[:-1] + self.dx[1:])/2, [self.dx[-1]/2]))
        self.dy_d = np.concatenate(([self.dy[0]/2], (self.dy[:-1] + self.dy[1:])/2, [self.dy[-1]/2]))

        # Time Stepping
        CFL = 1.0
        self.dt  = CFL / (self.c * np.sqrt((1/self.dx.min()**2) + (1/self.dy.min()**2)))
        self.setup_waveguide()
        
    def _build_dx(self):
        """Constructs the non-uniform grid in the x-direction."""
        self.dx_0 = self.lam_c / self.finesse
        
        self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
        self.L_f_dt, self.n_f_dt = self.L_and_n_fine(self.dx_0, self.t_m, self.alpha)
        self.n_d = int(np.ceil(self.d / self.dx_0 - self.L_f_dt / self.dx_0)) + self.n_f_dt
        self.L_f_twg, self.n_f_twg = self.L_and_n_fine(self.dx_0 / np.sqrt(self.eps_core), self.t_m, self.alpha)
        self.n_wg = int(np.ceil((self.L_wg - self.L_f_twg) * np.sqrt(self.eps_core)/ self.dx_0) + self.n_f_twg)
        self.L_f_wg_Lr, self.n_f_wg_Lr = self.L_and_n_fine(self.dx_0, self.dx_0 / np.sqrt(self.eps_core), self.alpha)
        self.n_Lr = int(np.ceil(self.Lr / self.dx_0 - self.L_f_wg_Lr / self.dx_0)) + self.n_f_wg_Lr
        
        self.dx = np.concatenate([
            np.full(self.n_Ll, self.Ll / self.n_Ll),
            np.full(int(self.n_d - self.n_f_dt), self.d / (self.n_d - self.n_f_dt)),
            self.alpha ** np.arange(self.n_f_dt, -1, -1) * self.t_m,
            self.alpha ** np.arange(1, self.n_f_twg + 1) * self.t_m,
            np.full(int(self.n_wg - self.n_f_twg), self.L_wg / self.n_wg),
            self.alpha ** np.arange(1, self.n_f_wg_Lr + 1) * self.dx_0 / np.sqrt(self.eps_core),
            np.full(int(self.n_Lr - self.n_f_wg_Lr), self.Lr / (self.n_Lr - self.n_f_wg_Lr))
        ])

    def _build_dy(self):
        """Constructs the non-uniform grid in the y-direction."""
        self.dy_0 = self.lam_c / 30
        
        self.L_f_ac, self.n_f_ac = self.L_and_n_fine(self.dy_0, self.dy_0 / np.sqrt(self.eps_clad), self.alpha)
        self.n_air = int(np.ceil(self.w_air / self.dy_0 - self.L_f_ac / self.dy_0)) + self.n_f_ac
        
        if self.wg_type == "step":
            self.n_clad = int(np.ceil(self.w_clad / self.dy_0 * np.sqrt(self.eps_clad)))
            self.n_core = int(np.ceil(self.w_core / self.dy_0 * np.sqrt(self.eps_core)))
            dy_mid = [
                np.full(self.n_clad, self.w_clad / self.n_clad),
                np.full(self.n_core, self.w_core / self.n_core),
                np.full(self.n_clad, self.w_clad / self.n_clad)
            ]
        else:
            self.deps_max = 0.01 # percentage of (self.eps_core - self.eps_clad)
            self.a_eps = 2 * np.sqrt(self.eps_core) * (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) / self.w_core ** 2
            self.b_eps = (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) ** 2 / self.w_core ** 4
            self.dy_core = min(np.abs(self.deps_max * (self.eps_core-self.eps_clad) / (self.a_eps * self.w_core + 1/2 * self.b_eps * self.w_core ** 3)), self.dy_0 / np.sqrt(self.eps_core))
            
            self.L_f_cc, self.n_f_cc = self.L_and_n_fine(self.dy_0 / np.sqrt(self.eps_clad), self.dy_core, self.alpha)
            self.n_clad = int(np.ceil((self.w_clad - self.L_f_cc) / self.dy_0 * np.sqrt(self.eps_clad))) + self.n_f_cc
            self.n_core = int(np.ceil(self.w_core / self.dy_core))
            
            dy_mid = [
                np.full(self.n_clad - self.n_f_cc, (self.w_clad - self.L_f_cc) / (self.n_clad - self.n_f_cc)),
                self.alpha ** np.arange(self.n_f_cc, 0, -1) * self.dy_core,
                np.full(self.n_core, self.w_core / self.n_core),
                self.alpha ** np.arange(1, self.n_f_cc + 1) * self.dy_core,
                np.full(self.n_clad - self.n_f_cc, (self.w_clad - self.L_f_cc) / (self.n_clad - self.n_f_cc))
            ]
            
        self.dy = np.concatenate([
            np.full(int(self.n_air - self.n_f_ac), self.w_air / (self.n_air - self.n_f_ac)),
            self.alpha ** np.arange(self.n_f_ac, 0, -1) * self.dy_0 / np.sqrt(self.eps_clad),
            *dy_mid,
            self.alpha ** np.arange(1, self.n_f_ac + 1) * self.dy_0 / np.sqrt(self.eps_clad),
            np.full(int(self.n_air - self.n_f_ac), self.w_air / (self.n_air - self.n_f_ac))
        ])
        
    def L_and_n_fine(self, d_coarse, d_fine, alpha = np.sqrt(2)):
        
        n_f = int(np.ceil(np.log(d_coarse / d_fine) / np.log(alpha))) - 1
        L_f = d_fine * (alpha**(n_f-1) - alpha) / (alpha - 1)
        
        return L_f, n_f
    
    def setup_waveguide(self):
        self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air):int(-self.n_air)] = self.eps_clad
        if self.wg_type == "step":
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = self.eps_core
        else:
            eps_val = lambda y: self.eps_core + self.a_eps * (y-self.W/2)**2 + self.b_eps * (y-self.W/2)**4 
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = eps_val(self.dy_core * np.arange(self.n_air + self.n_clad, (self.n_air + self.n_clad + self.n_core) + 1))
            
        self.sigma[int(self.n_Ll + self.n_d - 1),:int(self.n_air + self.n_clad)] = 3.5e7 #conductivity of aluminum
        self.sigma[int(self.n_Ll + self.n_d - 1), - int(self.n_air + self.n_clad):] = 3.5e7
        
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
        self.ix = slice(1,-1)
        self.iy = slice(1,-1)
        self.Ez      = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Ez_dot  = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Ez_ddot = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Jc      = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Hx      = np.zeros((self.cfg.nx, self.cfg.ny-1))
        self.Hx_dot  = np.zeros((self.cfg.nx, self.cfg.ny-1))
        self.Hy      = np.zeros((self.cfg.nx-1, self.cfg.ny))
        self.Hy_dot  = np.zeros((self.cfg.nx-1, self.cfg.ny))

        self.Hx_dot_old = np.zeros_like(self.Hx_dot)
        self.Hy_dot_old = np.zeros_like(self.Hy_dot)
        self.Ez_ddot_old = np.zeros_like(self.Ez_ddot)
        self.Ez_dot_old = np.zeros_like(self.Ez_dot)

    def init_pml(self):
        p, m = 20, 4
        eta_max = (m + 1) / (150 * np.pi * min([self.cfg.dx_f, self.cfg.dy_f]))
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

        self.inv_dx, self.inv_dy = 1.0 / cfg.dx, 1.0 / cfg.dy
        self.inv_dx_d, self.inv_dy_d = 1.0 / cfg.dx_d, 1.0 / cfg.dy_d


    def step(self, t):
        cfg = self.cfg
        ix, iy = self.ix, self.iy

        # Magnetic Field Update
        self.Hx_dot_old[:] = self.Hx_dot
        diff_Ez_y = self.Ez[:, 1:] - self.Ez[:, :-1]
        self.Hx_dot = (self.bym_hx * self.Hx_dot - diff_Ez_y * self.inv_dy[None, :]) / self.byp_hx
        self.Hx += (self.bxp_hx * self.Hx_dot - self.bxm_hx * self.Hx_dot_old) / self.bz_h

        self.Hy_dot_old[:] = self.Hy_dot
        diff_Ez_x = self.Ez[1:, :] - self.Ez[:-1, :]
        self.Hy_dot += diff_Ez_x * self.inv_dx[:, None] / self.bz_h
        self.Hy = (self.bxm_hy * self.Hy + (self.byp_hy * self.Hy_dot - self.bym_hy * self.Hy_dot_old)) / self.bxp_hy

        # Electric Field Update
        diff_Hy_x = self.Hy[1:, iy] - self.Hy[:-1, iy]
        diff_Hx_y = self.Hx[ix, 1:] - self.Hx[ix, :-1]
        curl_h = diff_Hy_x * self.inv_dx_d[ix, None] - diff_Hx_y * self.inv_dy_d[None, iy]
        
        self.Ez_ddot_old[:] = self.Ez_ddot
        self.Ez_ddot[ix, iy] = (self.coef_n * self.Ez_ddot[ix, iy] - self.coef_j * self.Jc[ix, iy] + curl_h) / self.coef_p
        
        avg_Ez_ddot = self.Ez_ddot[ix, iy] + self.Ez_ddot_old[ix, iy]
        self.Jc[ix, iy] = (self.am * self.Jc[ix, iy] + cfg.sigma * cfg.Z_local[ix, iy] * avg_Ez_ddot) / self.ap
        
        self.Ez_dot_old[:] = self.Ez_dot
        diff_Ez_ddot = self.Ez_ddot[ix, iy] - self.Ez_ddot_old[ix, iy]
        self.Ez_dot[ix, iy] = (self.bxm[ix, iy] * self.Ez_dot[ix, iy] + diff_Ez_ddot / (cfg.v_local[ix, iy] * cfg.dt)) / self.bxp[ix, iy]

        diff_Ez_dot = self.Ez_dot[ix, iy] - self.Ez_dot_old[ix, iy]
        self.Ez[ix, iy] = (self.bym[ix, iy] * self.Ez[ix, iy] + self.bz_e[ix, iy] * diff_Ez_dot) / self.byp[ix, iy]
        
        # Source
        src = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        self.Ez[cfg.x0, cfg.y0] -= cfg.dx[cfg.x0] * cfg.dy[cfg.y0] * src / self.coef_p[cfg.x0-1, cfg.y0-1]

################################################################################################################################################
#                                                          Simulation Runner:
################################################################################################################################################
class SimulationRunner:
    @staticmethod
    def execute(**kwargs):
        """
        Runs the simulation with optional parameter overrides.
        Example: results = SimulationRunner.execute(nt=2000, d=10)
        """
        # 1. Create config with possible overrides
        config = SimulationConfig(**kwargs)
        
        # 2. Initialize Solver
        solver = YeeSolver(config)
        
        # 3. Setup Data Storage
        history_frames = int(np.ceil(config.nt / 3))
        field_history = np.zeros((history_frames, config.nx, config.ny), dtype=np.float32)
        recorder_plane = np.zeros((config.nt, config.ny), dtype=np.float32)
        
        # 4. Run Simulation Loop
        for it in tqdm(range(config.nt), desc=f"Simulating (nt={config.nt})"):
            t = it * config.dt
            solver.step(t)
            
            if it % 3 == 0:
                field_history[it // 3] = solver.Ez
            
            recorder_plane[it, :] = solver.Ez[config.x1, :]
            
        # Return everything needed for plotting/analysis
        return {
            "config": config,
            "history": field_history,
            "recorder": recorder_plane,
            "times": np.arange(config.nt) * config.dt
        }

    @staticmethod
    def plot_2d_animation(results, interval=100):
        """Triggers the 2D Field Animation."""
        cfg = results["config"]
        hist = results["history"]
        
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        X, Y = np.meshgrid(nodes_x, nodes_y)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        
        # Initial plot:
        quad = ax.pcolormesh(X, Y, (hist[0] * cfg.Z_local).T, 
                             shading='nearest', cmap='RdBu_r', vmin=-1e-4, vmax=1e-4)
        time_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, color='white', ha='center')

        def update(i):
            frame_data = (hist[i] * cfg.Z_local).T
            quad.set_array(frame_data.ravel())
            time_text.set_text(f'Step {i*5}/{cfg.nt} | {i*5*cfg.dt*1e9:.2f} ns')
            return quad, time_text

        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval, blit=True)
        plt.show()

    @staticmethod
    def plot_1d_intensity(results, interval=50):
        """Triggers the 1D Intensity Animation at the recorder plane."""
        cfg = results["config"]
        rec = results["recorder"]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        
        fig, ax = plt.subplots(figsize=(8, 4))
        max_int = np.max(rec**2)
        line, = ax.plot([], [], color='red', lw=2)
        
        ax.set_xlim(nodes_y.min(), nodes_y.max())
        ax.set_ylim(0, max_int * 1.2)
        
        # Waveguide:
        ax.axvspan(nodes_y[int(cfg.n_air)             ], nodes_y[int(cfg.n_air + cfg.n_clad)], color='yellow', alpha=0.1)
        ax.axvspan(nodes_y[int(cfg.n_air + cfg.n_clad)], nodes_y[int(cfg.n_air + cfg.n_clad + cfg.n_core)], color='blue'  , alpha=0.1)
        ax.axvspan(nodes_y[int(cfg.n_air + cfg.n_clad + cfg.n_core)], nodes_y[int(cfg.n_air + 2 * cfg.n_clad + cfg.n_core)], color='yellow', alpha=0.1)
        time_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontweight='bold')

        def update(i):
            line.set_data(nodes_y, rec[i, :]**2)
            time_text.set_text(f'Intensity | t = {i * cfg.dt * 1e9:.3f} ns')
            return line, time_text

        ani = FuncAnimation(fig, update, frames=range(3 * cfg.nt // 4, cfg.nt, 2), interval = interval, blit=True)
        plt.show()

    @staticmethod
    def verify_with_hankel(results, src_pos, obs_pos):
        cfg = results["config"]
        ez_obs_data = results["recorder"][:, obs_pos[1]]
        
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        
        dx_m = nodes_x[obs_pos[0]] - nodes_x[src_pos[0]]
        dy_m = nodes_y[obs_pos[1]] - nodes_y[src_pos[1]]
        r = np.sqrt(dx_m**2 + dy_m**2)
        
        if r == 0:
            raise ValueError("r must be greater than zero")

        # Setup time and frequency 
        t = np.arange(cfg.nt) * cfg.dt
        freqs = fftfreq(cfg.nt, cfg.dt)
        band_idx = np.where((freqs > cfg.f_c * 0.2) & (freqs < cfg.f_c * 1.8))[0]
        f_valid = freqs[band_idx]
        omega = 2 * np.pi * f_valid
        k0 = omega / cfg.c
        
        # FFT of simulation data and the source function
        src_time = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        Ez_sim_f = fft(ez_obs_data) * cfg.dt
        Ez_sim_valid = Ez_sim_f[band_idx]
        J_src_f = fft(src_time) * cfg.dt
        J_src_valid = J_src_f[band_idx]
        H_sim = Ez_sim_valid / J_src_valid
        H_sim_corrected = H_sim * np.exp(1j * omega * cfg.dt)

        # Analytical Solution
        H_analytical = -(omega * cfg.mu0 / 4) * sp_special.hankel2(0, k0 * r)
        

        # Plotting
        plt.figure(figsize=(12, 5))
        # Magnitude Plot
        plt.subplot(1, 2, 1)
        plt.plot(f_valid, np.abs(H_sim_corrected) / np.max(np.abs(H_sim_corrected)), label='Simulation', lw=2)
        plt.plot(f_valid, np.abs(H_analytical) / np.max(np.abs(H_analytical)), '--', label='Analytical', lw=2)
        plt.title('Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Phase Plot
        plt.subplot(1, 2, 2)
        plt.plot(f_valid,  np.unwrap(np.angle(H_sim_corrected)), label='Simulation Phase', lw=2)
        plt.plot(f_valid, np.unwrap(np.angle(H_analytical)), '--', label='Analytical Phase', lw=2)
        plt.title('Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_grid_spacing(results):
        """Visualizes the dx and dy spacing to verify refinement."""
        cfg = results["config"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot DX spacing
        ax1.plot(cfg.dx, 'o-', markersize=2, label='dx spacing')
        ax1.set_title("X-Grid Spacing (dx)")
        ax1.set_xlabel("Cell Index (i)")
        ax1.set_ylabel("Spacing (m)")
        ax1.grid(True, alpha=0.3)
        
        # Plot DY spacing
        ax2.plot(cfg.dy, 'o-', markersize=2, color='orange', label='dy spacing')
        ax2.set_title("Y-Grid Spacing (dy)")
        ax2.set_xlabel("Cell Index (j)")
        ax2.set_ylabel("Spacing (m)")
        ax2.grid(True, alpha=0.3)
        
        # Highlight waveguide region in dy plot
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        ax2.axvspan(int(cfg.n_air), int(cfg.n_air + cfg.n_clad), color='yellow', alpha=0.1)
        ax2.axvspan(int(cfg.n_air + cfg.n_clad),int(cfg.n_air + cfg.n_clad + cfg.n_core), color='blue'  , alpha=0.1)
        ax2.axvspan(int(cfg.n_air + cfg.n_clad + cfg.n_core), int(cfg.n_air + 2 * cfg.n_clad + cfg.n_core), color='yellow', alpha=0.1)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mesh(results, zoom_x=(400, 600), zoom_y=None):
        """Plots the actual grid lines (the mesh) to see the refinement."""
        cfg = results["config"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        
    
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.vlines(nodes_x, ymin=nodes_y.min(), ymax=nodes_y.max(), colors='black', lw=0.5, alpha=0.5)
        ax.hlines(nodes_y, xmin=nodes_x.min(), xmax=nodes_x.max(), colors='blue', lw=0.5, alpha=0.5)

        ax.set_title("FDTD Mesh Visualization")
        ax.set_xlabel("X-position (m)")
        ax.set_ylabel("Y-position (m)")
        ax.legend()
        plt.show()

    @classmethod
    def run_full_analysis(cls, speed=40, **kwargs):
        data = cls.execute(**kwargs)
        ms_interval = int(1000 / speed)
        cls.plot_2d_animation(data, interval=ms_interval)
        cls.plot_1d_intensity(data, interval=ms_interval)
        cls.plot_grid_spacing(data)
        cls.plot_mesh(data)
        
        cfg = data["config"]
        cls.verify_with_hankel(data, (cfg.x0, cfg.y0), (cfg.x1, cfg.y0))
        
        return data






if __name__ == "__main__":
    print("Starting simulation...")
    results = SimulationRunner.run_full_analysis(speed=2000, nt=1500, wg_type="grin", finesse=10, eps_core=1.11, eps_clad=1.1)
    print("Simulation finished.")

