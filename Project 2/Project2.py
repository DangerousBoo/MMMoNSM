import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from tqdm import tqdm

class SimulationConfig:
    """Handles physical constants, geometry, and simulation variables for 1D FDTD."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Physical Constants
        self.hbar = 1.054571817e-34  # J*s
        self.m_e = 9.1093837015e-31  # kg
        self.e = 1.602176634e-19     # C
        
        # Effective mass configuration
        self.m_star = getattr(self, "m_star", 0.023 * self.m_e)
        
        # Domain Dimensions (1D target)
        self.dx = getattr(self, "dx", 0.2e-9) # 0.2 nm spatial step
        
        # Geometry lengths
        self.L_buffer_left = getattr(self, "L_buffer_left", 200e-9)
        self.L_absorb = getattr(self, "L_absorb", 50e-9)
        self.L_barrier1 = 10e-9
        self.L_well = 30e-9
        self.L_barrier2 = 10e-9
        self.L_buffer_right = getattr(self, "L_buffer_right", 200e-9)
        
        self.L_total = (self.L_absorb + self.L_buffer_left + self.L_barrier1 + 
                        self.L_well + self.L_barrier2 + self.L_buffer_right + self.L_absorb)
        
        self.nx = int(np.ceil(self.L_total / self.dx))
        self.x = np.linspace(0, self.L_total, self.nx)
        
        # Define layer boundaries
        self.x_abs1 = self.L_absorb
        self.x_bar1 = self.x_abs1 + self.L_buffer_left
        self.x_well = self.x_bar1 + self.L_barrier1
        self.x_bar2 = self.x_well + self.L_well
        self.x_buf2 = self.x_bar2 + self.L_barrier2
        self.x_abs2 = self.x_buf2 + self.L_buffer_right
        
        # Indices and nodes
        self.i_bar1 = int(self.x_bar1 / self.dx)
        self.i_well = int(self.x_well / self.dx)
        self.i_bar2 = int(self.x_bar2 / self.dx)
        self.i_buf2 = int(self.x_buf2 / self.dx)
        self.i_abs1 = int(self.x_abs1 / self.dx)
        self.i_abs2 = int(self.x_abs2 / self.dx)
        
        self.n_layer = self.i_abs1  # Number of nodes in absorbing layer
        
        # Energy and Potentials
        self.V0 = getattr(self, "V0", 0.2 * self.e) # 0.6 eV 
        self.V_DC = getattr(self, "V_DC", 0.0) # Bias voltage
        
        # Initialize Potentials
        self.U_R = np.zeros(self.nx)
        self.U_I = np.zeros(self.nx)
        self._build_potentials()
        
        # Stability / Time Step Calculation
        dt_max = self.hbar / (self.hbar**2 / (self.m_star * self.dx**2) + np.max(self.U_R)/2)
        # 4th-order laplacian has a stricter stability limit!
        CFL = 0.7
        self.dt = CFL * dt_max
        self.T_total = getattr(self, "T_total", 300e-15) # 300 femtoseconds
        self.nt = int(np.ceil(self.T_total / self.dt))
        
        # Initial Wavepacket Setup
        self.E_target = getattr(self, "E_target", 0.2 * self.e)
        K_E = self.E_target - self.U_R[0]  # Kinetic energy in the left buffer
        self.k_x = np.sqrt(2 * self.m_star * max(0, K_E)) / self.hbar
        self.x_0 = self.x_abs1 + self.L_buffer_left * 0.3
        self.sigma_x = getattr(self, "sigma_x", 15e-9)

    def _build_potentials(self):
        self.U_R[self.i_bar1:self.i_well] = self.V0 # InP Barrier 1
        self.U_R[self.i_bar2:self.i_buf2] = self.V0 # InP Barrier 2
        
        # Bias (V_DC applied to the left contact, dropping only across the well)
        if self.V_DC != 0:
            bias_energy = -self.e * self.V_DC
            
            # The left potential (buffer + barrier 1) is shifted by the offset
            self.U_R[:self.i_well] += bias_energy
            
            # The well is the sloped curve connecting the two (from bias_energy up to 0)
            slope = (0.0 - bias_energy) / (self.x_bar2 - self.x_well)
            for i in range(self.i_well, self.i_bar2):
                self.U_R[i] += bias_energy + slope * (self.x[i] - self.x_well)
            
            # The right potential (barrier 2 + buffer right) stays at 0
            
        # # 3. Absorbing boundaries (U_I)
        # m_poly = 3
        # sigma_abs = 2.0 * self.V0 
        
        # for i in range(self.n_layer):
        #     damping = sigma_abs * ( (self.n_layer - i) / self.n_layer )**m_poly
        #     self.U_I[i] = damping  # Left absorbing layer
        #     self.U_I[-1 - i] = damping # Right absorbing layer


class SchrodingerSolver:
    def __init__(self, config):
        self.cfg = config
        self.init_fields()
        self.init_coefficients()
        
    def init_fields(self):
        # psi_R is evaluated at n - 1/2 ..., psi_I at n...
        self.psi_R = np.zeros(self.cfg.nx)
        self.psi_I = np.zeros(self.cfg.nx)
        
        C_norm = (2 * np.pi * self.cfg.sigma_x**2)**(-0.25)
        envelope = C_norm * np.exp(- (self.cfg.x - self.cfg.x_0)**2 / (4 * self.cfg.sigma_x**2))
        phase = self.cfg.k_x * self.cfg.x
        self.psi_I[:] = envelope * np.sin(phase)
        
        # propagate back dt/2 to set psi_R at t = -dt/2, with psi(t) ~ exp(-j E t / hbar)
        phase_backward = phase - (self.cfg.E_target * (-self.cfg.dt/2) / self.cfg.hbar)
        self.psi_R[:] = envelope * np.cos(phase_backward)

    def init_coefficients(self):
        self.laplacian_factor = self.cfg.hbar**2 / (2 * self.cfg.m_star * self.cfg.dx**2)
        denom = 2 * self.cfg.hbar + self.cfg.dt * self.cfg.U_I
        self.c_A = (2 * self.cfg.hbar - self.cfg.dt * self.cfg.U_I) / denom
        self.c_B = (2 * self.cfg.dt) / denom
        self.order = 4 # use 4th-order spatial derivatives

    def step(self):
        lap_psi_I = np.zeros_like(self.psi_I)
        if self.order == 4:
            lap_psi_I[2:-2] = (-self.psi_I[4:] + 16*self.psi_I[3:-1] - 30*self.psi_I[2:-2] + 16*self.psi_I[1:-3] - self.psi_I[:-4]) / 12.0
            lap_psi_I[1] = self.psi_I[2] - 2*self.psi_I[1] + self.psi_I[0]
            lap_psi_I[-2] = self.psi_I[-1] - 2*self.psi_I[-2] + self.psi_I[-3]
        else:
            lap_psi_I[1:-1] = self.psi_I[2:] - 2 * self.psi_I[1:-1] + self.psi_I[:-2]

        H_R_psi_I = - self.laplacian_factor * lap_psi_I + self.cfg.U_R * self.psi_I
        self.psi_R = self.c_A * self.psi_R + self.c_B * H_R_psi_I

        lap_psi_R = np.zeros_like(self.psi_R)
        if self.order == 4:
            lap_psi_R[2:-2] = (-self.psi_R[4:] + 16*self.psi_R[3:-1] - 30*self.psi_R[2:-2] + 16*self.psi_R[1:-3] - self.psi_R[:-4]) / 12.0
            lap_psi_R[1] = self.psi_R[2] - 2*self.psi_R[1] + self.psi_R[0]
            lap_psi_R[-2] = self.psi_R[-1] - 2*self.psi_R[-2] + self.psi_R[-3]
        else:
            lap_psi_R[1:-1] = self.psi_R[2:] - 2 * self.psi_R[1:-1] + self.psi_R[:-2]
        
        H_R_psi_R = - self.laplacian_factor * lap_psi_R + self.cfg.U_R * self.psi_R
        self.psi_I = self.c_A * self.psi_I - self.c_B * H_R_psi_R

    def get_probability_density(self):
        return self.psi_R**2 + self.psi_I**2


class SimulationRunner:
    @staticmethod
    def execute(frame_skip=100, record_ix=None, **kwargs):
        config = SimulationConfig(**kwargs)
        solver = SchrodingerSolver(config)
        
        n_frames = int(np.ceil(config.nt / frame_skip))
        history_density = np.zeros((n_frames, config.nx), dtype=np.float32)
        
        if record_ix is None:
            record_ix = int(config.x_bar2 / config.dx) + int(50e-9 / config.dx)
            
        time_signal_R = np.zeros(config.nt)
        time_signal_I = np.zeros(config.nt)
        
        frame_idx = 0
        for it in tqdm(range(config.nt), desc=f"Simulating (nt={config.nt})"):
            solver.step()
            
            time_signal_R[it] = solver.psi_R[record_ix]
            time_signal_I[it] = solver.psi_I[record_ix]
            
            if it % frame_skip == 0 and frame_idx < n_frames:
                history_density[frame_idx] = solver.get_probability_density()
                frame_idx += 1
                
        return {
            "config": config,
            "history": history_density,
            "frame_skip": frame_skip,
            "times": np.arange(config.nt) * config.dt,
            "record_ix": record_ix,
            "time_signal_R": time_signal_R,
            "time_signal_I": time_signal_I
        }
        
    @staticmethod
    def plot_animation(results, fps=30):
        cfg = results["config"]
        hist = results["history"]
        x_nm = cfg.x * 1e9
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot probability density
        line, = ax1.plot([], [], color='blue', lw=2, label=r'$|\psi|^2$')
        ax1.set_xlim(x_nm.min(), x_nm.max())
        ax1.set_ylim(0, np.max(hist) * 1.1)
        ax1.set_xlabel("Position (nm)")
        ax1.set_ylabel(r"Probability Density $|\psi|^2$", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot potential energy scaled to eV
        ax2 = ax1.twinx()
        U_R_eV = cfg.U_R / cfg.e
        ax2.plot(x_nm, U_R_eV, color='red', lw=1.5, ls='--', label='Potential Energy (U_R)')
        ax2.set_ylabel("Potential Energy (eV)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        y_min = min(-0.1, np.min(U_R_eV) * 1.2)
        y_max = max(cfg.V0/cfg.e, np.max(U_R_eV)) * 1.5
        ax2.set_ylim(y_min, y_max)
        
        # Highlight regions
        ax2.axvspan(cfg.x_bar1*1e9, cfg.x_bar2*1e9, color='gray', alpha=0.1, label='Double Barrier')
        
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='black', fontweight='bold')
        prob_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, color='black')
        
        fs = results["frame_skip"]
        def update(i):
            line.set_data(x_nm, hist[i])
            time_text.set_text(f'Time: {i*fs*cfg.dt*1e15:.2f} fs')
            prob = np.sum(hist[i]) * cfg.dx
            prob_text.set_text(f'Total Probability = {prob:.4f}')
            return line, time_text, prob_text
            
        interval_ms = max(1, int(1000 / fps))
        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval_ms, blit=True)
        results["ani"] = ani
        plt.show()

results = SimulationRunner.execute(V_DC=0.0, T_total=1.0e-12, E_target=0.15*1.602e-19, frame_skip=200)
SimulationRunner.plot_animation(results)
