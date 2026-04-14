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
        self.hbar   = 1.054571817e-34 
        self.m_e    = 9.1093837015e-31 
        self.e      = 1.602176634e-19  
        self.m_star = getattr(self, "m_star", 0.023 * self.m_e)
        
        # Domain Dimensions (1D target)
        self.dx         = getattr(self, "dx", 0.2e-9)
        self.L_buffer   = getattr(self, "L_buffer", 200e-9)
        self.L_absorb   = getattr(self, "L_absorb", 50e-9)
        self.L_barrier1 = getattr(self, "L_barrier1", 10e-9)
        self.L_well     = getattr(self, "L_well", 30e-9)
        self.L_barrier2 = getattr(self, "L_barrier2", 10e-9)
        self.L_total    = (self.L_absorb + self.L_buffer + self.L_barrier1 + self.L_well 
                        + self.L_barrier2 + self.L_buffer + self.L_absorb)

        self.nx = int(np.ceil(self.L_total / self.dx))
        self.x  = np.linspace(0, self.L_total, self.nx)
        
        # Define layer boundaries
        self.x_abs1 = self.L_absorb
        self.x_bar1 = self.x_abs1 + self.L_buffer
        self.x_well = self.x_bar1 + self.L_barrier1
        self.x_bar2 = self.x_well + self.L_well
        self.x_buf2 = self.x_bar2 + self.L_barrier2
        self.x_abs2 = self.x_buf2 + self.L_absorb
        
        # Indices and nodes
        def get_idx(length): return int(length / self.dx)
        
        self.i_abs1 = get_idx(self.L_absorb)
        self.i_bar1 = self.i_abs1 + get_idx(self.L_buffer)
        self.i_well = self.i_bar1 + get_idx(self.L_barrier1)
        self.i_bar2 = self.i_well + get_idx(self.L_well)
        self.i_buf2 = self.i_bar2 + get_idx(self.L_barrier2)
        
        self.n_layer = self.i_abs1  # Number of nodes in absorbing layer
        
        # Transversal Energy
        self.Ly = getattr(self, "Ly", 40e-9)
        self.Lz = getattr(self, "Lz", 40e-9)
        self.n_y = getattr(self, "n_y", 1)
        self.n_z = getattr(self, "n_z", 1)
        self.E_trans = (self.hbar**2 / (2 * self.m_star)) * ((np.pi * self.n_y / self.Ly)**2 + (np.pi * self.n_z / self.Lz)**2)
        
        # Energy and Potentials
        self.V0 = getattr(self, "V0", 0.2 * self.e) # 0.6 eV 
        self.V_DC = getattr(self, "V_DC", 0.4 ) # Bias voltage
        
        # Initialize Potentials
        self.U_R = np.zeros(self.nx)
        self.U_I = np.zeros(self.nx)
        self._build_potentials()
        self.U_R += self.E_trans # Include transversal energy
        
        # Stability / Time Step
        dt_max = self.hbar / (self.hbar**2 / (self.m_star * self.dx**2) + np.max(self.U_R)/2)
        self.dt = kwargs.get("dt", 0.7 * dt_max) # Allow dt override to match domains
        self.T_total = getattr(self, "T_total", 100e-15)
        self.nt = int(np.ceil(self.T_total / self.dt))
        
        # Initial Wavepacket Setup
        self.E_target = getattr(self, "E_target", 0.2 * self.e)
        K_E = self.E_target - self.U_R[0]  # Kinetic energy in the left buffer
        self.k_x = np.sqrt(2 * self.m_star * max(0, K_E)) / self.hbar
        self.x_0 = self.x_abs1 + self.L_buffer * 0.3
        self.sigma_x = getattr(self, "sigma_x", 15e-9)

    def _build_potentials(self):
        self.U_R[self.i_bar1:self.i_well] = self.V0 # InP Barrier 1
        self.U_R[self.i_bar2:self.i_buf2] = self.V0 # InP Barrier 2
        
        if self.V_DC != 0:
            bias = -self.e * self.V_DC
            self.U_R[:self.i_well] += bias
            x_well = self.x[self.i_well:self.i_bar2]
            self.U_R[self.i_well:self.i_bar2] += bias * (1 - (x_well - self.x_well) / (self.x_bar2 - self.x_well))
            
        i_arr = np.arange(self.n_layer)
        dist_factor = ((self.n_layer - i_arr) / self.n_layer)**3
        self.U_I[:self.n_layer] = 2.0 * self.V0 * dist_factor
        self.U_I[-self.n_layer:] = 2.0 * self.V0 * dist_factor[::-1]

class SchrodingerSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.order = 4

        denom = (1 + 0.5 * cfg.dt * cfg.U_I / cfg.hbar)
        self.c_A = (1 - 0.5 * cfg.dt * cfg.U_I / cfg.hbar) / denom
        self.c_B = (cfg.dt / cfg.hbar) / denom
        self.lap_factor = (cfg.hbar * cfg.dt) / (2 * cfg.m_star * cfg.dx**2) / denom
        
        env = (2 * np.pi * cfg.sigma_x**2)**(-0.25) * np.exp(- (cfg.x - cfg.x_0)**2 / (4 * cfg.sigma_x**2))
        phase = cfg.k_x * cfg.x
        
        self.psi_I = env * np.sin(phase)
        self.psi_R = env * np.cos(phase + cfg.E_target * cfg.dt / (2 * cfg.hbar)) # figure out why this term has to be here, its 4am so too late to think i just vibe code

    def _lap(self, psi):
        lap = np.zeros_like(psi)
        if self.order == 4:
            lap[2:-2] = (-psi[4:] + 16*psi[3:-1] - 30*psi[2:-2] + 16*psi[1:-3] - psi[:-4]) / 12.0
            lap[1], lap[-2] = psi[2] - 2*psi[1] + psi[0], psi[-1] - 2*psi[-2] + psi[-3]
        else:
            lap[1:-1] = psi[2:] - 2*psi[1:-1] + psi[:-2]
        return lap

    def step(self):
        self.psi_R = (self.c_A * self.psi_R - self.lap_factor * self._lap(self.psi_I) + self.c_B * (self.cfg.U_R) * self.psi_I)
        self.psi_I = (self.c_A * self.psi_I + self.lap_factor * self._lap(self.psi_R) - self.c_B * (self.cfg.U_R) * self.psi_R)

    @property
    def density(self):
        return self.psi_R**2 + self.psi_I**2


class SimulationRunner:
    @staticmethod
    def execute(frame_skip=100, record_ix=None, **kwargs):
        cfg = SimulationConfig(**kwargs)
        solver = SchrodingerSolver(cfg)
        
        n_frames = int(np.ceil(cfg.nt / frame_skip))
        history = np.zeros((n_frames, cfg.nx), dtype=np.float32)
        record_ix = record_ix or int(cfg.x_buf2 / cfg.dx) + int(20e-9 / cfg.dx)
            
        sig_R, sig_I = np.zeros(cfg.nt), np.zeros(cfg.nt)
        current_sig = np.zeros(cfg.nt)
        curr_prefactor = (cfg.e * cfg.hbar) / (cfg.m_star * cfg.dx)
        
        frame_idx = 0
        for it in tqdm(range(cfg.nt), desc=f"Simulating (nt={cfg.nt})"):
            solver.step()
            sig_R[it], sig_I[it] = solver.psi_R[record_ix], solver.psi_I[record_ix]
            current_sig[it] = curr_prefactor * (solver.psi_R[record_ix] * solver.psi_I[record_ix+1] - solver.psi_R[record_ix+1] * solver.psi_I[record_ix])
            
            if not it % frame_skip and frame_idx < n_frames:
                history[frame_idx] = solver.density
                frame_idx += 1
                
        return {
            "config": cfg,
            "history": history,
            "frame_skip": frame_skip,
            "times": np.arange(cfg.nt) * cfg.dt,
            "record_ix": record_ix,
            "time_signal_R": sig_R,
            "time_signal_I": sig_I,
            "current_signal": current_sig
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
        
        # Plot potential energy
        ax2 = ax1.twinx()
        U_R_eV = cfg.U_R / cfg.e
        ax2.plot(x_nm, U_R_eV, color='red', lw=1.5, ls='--', label='Potential Energy (U_R)')
        ax2.set_ylabel("Potential Energy (eV)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        y_min = min(-0.1, np.min(U_R_eV) * 1.2)
        y_max = max(cfg.V0/cfg.e, np.max(U_R_eV)) * 1.5
        ax2.set_ylim(y_min, y_max)
        
        ax2.axvspan(cfg.x_bar1*1e9, cfg.x_buf2*1e9, color='gray', alpha=0.1, label='Double Barrier')
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
        
        # Plot current
        fig_curr = plt.figure(figsize=(8, 4))
        plt.plot(results["times"] * 1e15, results["current_signal"], 'g-', lw=2)
        plt.xlabel("Time (fs)")
        plt.ylabel("Quantum Current")
        plt.title("Current vs Time at right of double barrier")
        plt.grid(True)
        
        plt.show()

results_barrier = SimulationRunner.execute(V_DC=0.2, T_total=250.0e-15, E_target=0.35*1.602e-19, frame_skip=200)
results_free = SimulationRunner.execute(V0=0.0, V_DC=0.0, T_total=250.0e-15, E_target=0.35*1.602e-19, frame_skip=200, dt=results_barrier["config"].dt)





# testjen , lett gwn wat AI code vanaf hier dus skip dit dankoe
dt = results_barrier["config"].dt
nt = results_barrier["config"].nt

freqs = fftfreq(nt, dt)
pos_mask = freqs > 0
freqs = freqs[pos_mask]

fft_J_bar = fft(results_barrier["current_signal"])[pos_mask]
fft_J_free = fft(results_free["current_signal"])[pos_mask]

T = np.abs(fft_J_bar) / (np.abs(fft_J_free) + 1e-20)
E_eV = (2 * np.pi * results_barrier["config"].hbar * freqs) / results_barrier["config"].e

plt.figure(figsize=(8, 4))
plt.plot(E_eV, T, 'm-', lw=2)
signal_power = np.abs(fft_J_free)
valid_idx = signal_power > np.max(signal_power) * 1e-3
if np.any(valid_idx):
    plt.xlim(E_eV[valid_idx].min(), E_eV[valid_idx].max())
plt.ylim(0, 1.2)
plt.title("Transmission Spectrum $T(E)$ (via FFT of $J$)")
plt.xlabel("Energy (eV)")
plt.ylabel("Transmission Coefficient T")
plt.grid(True)

SimulationRunner.plot_animation(results_barrier)
