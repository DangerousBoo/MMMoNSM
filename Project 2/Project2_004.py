import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from tqdm import tqdm
import concurrent.futures
from functools import partial

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
        self.dx         = getattr(self, "dx", 0.5e-9)
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
        self.V0 = getattr(self, "V0", 0.6) * self.e # Evaluate V0 kwarg input as eV
        self.V_DC = getattr(self, "V_DC", 0.4 ) # Bias voltage
        
        # Initialize Potentials
        self.U_R = np.zeros(self.nx)
        self.U_I = np.zeros(self.nx)
        self._build_potentials()
        self.U_R += self.E_trans # Include transversal energy
        
        # Stability / Time Step
        self.order = getattr(self, "order", 4)
        if self.order == 4:
            dt_max = 2 / ((8/3 * self.hbar / self.m_star) / self.dx**2 + np.max(self.U_R) / self.hbar)
        elif self.order == 2:
            dt_max = 2 / ((2 * self.hbar / self.m_star) / self.dx**2 + np.max(self.U_R) / self.hbar)
        else:
            raise ValueError(f"Unsupported finite difference order: {self.order}")

        self.dt = kwargs.get("dt", 0.7 * dt_max) # Allow dt override to match domains
        self.T_total = getattr(self, "T_total", 100e-15)
        self.nt = int(np.ceil(self.T_total / self.dt))
        
        # Initial Wavepacket Setup
        self.E_target = getattr(self, "E_target", 0.2) * self.e # input E
        K_E = self.E_target - self.U_R[0]  # Kinetic energy in the left buffer
        self.k_x = np.sqrt(2 * self.m_star * max(0, K_E)) / self.hbar
        self.x_0 = self.x_abs1 + self.L_buffer * 0.3
        self.sigma_x = getattr(self, "sigma_x", 15e-9)

    def _build_potentials(self):
        # self.U_R[self.i_bar1:self.i_well] = self.V0 # InP Barrier 1
        # self.U_R[self.i_bar2:self.i_buf2] = self.V0 # InP Barrier 2
        # Als je de lijntjes hierboven gebruikt krijg je pure poep (alles shifted)
        def compute_potential(x_start, x_end, V_0):
            left_edge = np.maximum(self.x - self.dx/2, x_start)
            right_edge = np.minimum(self.x + self.dx/2, x_end)
            overlap = np.maximum(0, right_edge - left_edge)
            return (overlap / self.dx) * V_0

        self.U_R += compute_potential(self.x_bar1, self.x_well, self.V0)
        self.U_R += compute_potential(self.x_bar2, self.x_buf2, self.V0)

        if self.V_DC != 0:
            bias = -self.e * self.V_DC
            self.U_R[:self.i_well] += bias
            x_well = self.x[self.i_well:self.i_bar2]
            self.U_R[self.i_well:self.i_bar2] += bias * (1 - (x_well - self.x_well) / (self.x_bar2 - self.x_well))
            
        i_arr = np.arange(self.n_layer)
        dist_factor = ((self.n_layer - i_arr) / self.n_layer)**3
        abs_V = self.V0 if self.V0 != 0 else 0.2 * self.e # laatste is vr de free space sim
        self.U_I[:self.n_layer] = 2.0 * abs_V * dist_factor
        self.U_I[-self.n_layer:] = 2.0 * abs_V * dist_factor[::-1]

class SchrodingerSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.order = cfg.order

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

class TransmissionAnalyzer:
    """Computes and plots both numerical FDTD and analytical TMM transmission spectra."""
    @staticmethod
    def get_analytical_T(E_eV_arr, cfg):
        E = E_eV_arr * cfg.e
        k_f = np.sqrt(2 * cfg.m_star * (E - cfg.E_trans) + 0j) / cfg.hbar
        k_b = np.sqrt(2 * cfg.m_star * (E - (cfg.V0 + cfg.E_trans)) + 0j) / cfg.hbar
        
        N = len(E_eV_arr)
        
        def intf(k1, k2):
            M = np.zeros((N, 2, 2), dtype=np.complex128)
            ratio = k2 / k1
            M[:, 0, 0] = 1 + ratio
            M[:, 0, 1] = 1 - ratio
            M[:, 1, 0] = 1 - ratio
            M[:, 1, 1] = 1 + ratio
            return 0.5 * M
            
        def prop(k, d):
            M = np.zeros((N, 2, 2), dtype=np.complex128)
            M[:, 0, 0] = np.exp(-1j * k * d)
            M[:, 1, 1] = np.exp(1j * k * d)
            return M
            
        M1 = intf(k_f, k_b)
        M2 = prop(k_b, cfg.L_barrier1)
        M3 = intf(k_b, k_f)
        M4 = prop(k_f, cfg.L_well)
        M5 = intf(k_f, k_b)
        M6 = prop(k_b, cfg.L_barrier2)
        M7 = intf(k_b, k_f)
        
        M = M1 @ M2 @ M3 @ M4 @ M5 @ M6 @ M7
        return 1.0 / np.abs(M[:, 0, 0])**2

    @staticmethod
    def plot_transmission(results_barrier, results_free):
        cfg = results_barrier["config"]
        psi_t_bar = results_barrier["time_signal_R"] + 1j * results_barrier["time_signal_I"]
        psi_t_free = results_free["time_signal_R"] + 1j * results_free["time_signal_I"]
        
        N_pad = cfg.nt * 8
        fft_bar = fft(psi_t_bar, n=N_pad)
        fft_free = fft(psi_t_free, n=N_pad)
        freqs = fftfreq(N_pad, cfg.dt)
        E_all = -(2 * np.pi * cfg.hbar * freqs) / cfg.e
        
        pos_mask = E_all > 0
        E_eV = E_all[pos_mask]
        Psi_bar = fft_bar[pos_mask]
        Psi_free = fft_free[pos_mask]
        
        U_obs_bar = cfg.U_R[results_barrier["record_ix"]]
        U_obs_free = results_free["config"].U_R[results_free["record_ix"]]
        E_J = E_eV * cfg.e 
        
        valid_E = (E_J > U_obs_bar) & (E_J > U_obs_free)
        E_eV_plot = E_eV[valid_E]
        
        k_bar = np.sqrt(2 * cfg.m_star * (E_J[valid_E] - U_obs_bar))
        k_free = np.sqrt(2 * cfg.m_star * (E_J[valid_E] - U_obs_free))  
        T = (k_bar / k_free) * (np.abs(Psi_bar[valid_E])**2 / np.abs(Psi_free[valid_E])**2)
        
        plt.figure(figsize=(8, 4))
        plt.plot(E_eV_plot, T, 'm-', lw=2, label="FDTD Simulation")
        
        # Unconditionally overlay analytical curve for comparison
        T_analy = TransmissionAnalyzer.get_analytical_T(E_eV_plot, cfg)
        plt.plot(E_eV_plot, T_analy, 'k--', lw=1.5, label="Analytical (V_DC=0)")
        
        k_0 = cfg.k_x
        sigma_k = 1.0 / (2.0 * cfg.sigma_x)
        sigma_E_eV = ((cfg.hbar**2 * k_0 / cfg.m_star) * sigma_k) / cfg.e
        E_center_eV = cfg.E_target / cfg.e
        
        E_min = E_center_eV - 3 * sigma_E_eV
        E_max = E_center_eV + 3 * sigma_E_eV
        plt.axvline(E_min, color='r', linestyle='--', alpha=0.6, label=r'$\pm 3\sigma_E$ width')
        plt.axvline(E_max, color='r', linestyle='--', alpha=0.6)
        
        plt.xlim(E_min, E_max)
        plt.ylim(0, 1.1)
        plt.title("Transmission Spectrum $T(E)$")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission Coefficient T")
        plt.legend()
        plt.grid(True)
        plt.show()

        
class SimulationRunner:
    @staticmethod
    def execute(frame_skip=100, record_ix=None, disable_tqdm=False, record_history=True, **kwargs):
        cfg = SimulationConfig(**kwargs)
        solver = SchrodingerSolver(cfg)
        
        n_frames = int(np.ceil(cfg.nt / frame_skip)) if record_history else 0
        history = np.zeros((n_frames, cfg.nx), dtype=np.float32) if record_history else None
        record_ix = record_ix or int(cfg.x_buf2 / cfg.dx) + int(20e-9 / cfg.dx)
            
        sig_R, sig_I = np.zeros(cfg.nt), np.zeros(cfg.nt)
        frame_idx = 0
        
        for it in tqdm(range(cfg.nt), desc=f"Simulating (nt={cfg.nt})", disable=disable_tqdm):
            solver.step()
            sig_R[it], sig_I[it] = solver.psi_R[record_ix], solver.psi_I[record_ix]
            
            if record_history and not it % frame_skip and frame_idx < n_frames:
                history[frame_idx] = solver.density
                frame_idx += 1
                
        return {
            "config": cfg,
            "history": history,
            "frame_skip": frame_skip,
            "record_ix": record_ix,
            "time_signal_R": sig_R,
            "time_signal_I": sig_I
        }
        
    @staticmethod
    def plot_animation(results, fps=30):
        cfg = results["config"]
        hist = results["history"]
        x_nm = cfg.x * 1e9
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot probability density
        line_psi, = ax1.plot([], [], color='blue', lw=2, label=r'$|\psi|^2$')
        ax1.set_xlim(x_nm.min(), x_nm.max())
        ax1.set_ylim(0, np.max(hist) * 1.1)
        ax1.set_xlabel("Position (nm)")
        ax1.set_ylabel(r"Probability Density $|\psi|^2$", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot recorder position
        record_x_nm = cfg.x[results["record_ix"]] * 1e9
        ax1.plot(record_x_nm, 0, 'ro', markersize=8, label='Recorder')
        
        # Plot potential energy
        ax2 = ax1.twinx()
        U_R_eV = cfg.U_R / cfg.e
        ax2.plot(x_nm, U_R_eV, color='red', lw=1.5, ls='--', label='Potential Energy')
        ax2.set_ylabel("Potential Energy (eV)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        y_min = min(-0.1, np.min(U_R_eV) * 1.2)
        y_max = max(cfg.V0/cfg.e, np.max(U_R_eV)) * 1.5
        ax2.set_ylim(y_min, y_max)
        
        ax2.axvspan(cfg.x_bar1*1e9, cfg.x_buf2*1e9, color='gray', alpha=0.1, label='Double Barrier')
        
        # Unify legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='black', fontweight='bold')
        prob_text = ax1.text(0.02, 0.88, '', transform=ax1.transAxes, color='black')
        
        fs = results["frame_skip"]
        def update(i):
            line_psi.set_data(x_nm, hist[i])
            time_text.set_text(f'Time: {i*fs*cfg.dt*1e15:.2f} fs')
            prob = np.sum(hist[i]) * cfg.dx
            prob_text.set_text(f'Total Probability = {prob:.4f}')
            return line_psi, time_text, prob_text
            
        interval_ms = max(1, int(1000 / fps))
        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval_ms, blit=True)
        results["ani"] = ani
        plt.show()

class IVCharacteristic:
    @staticmethod
    def _run_bias(V, base_kwargs):
        e = 1.602176634e-19
        h = 6.62607015e-34
        k_B = 1.380649e-23
        Temp = 4.0 # 4K for sharp Fermi edge
        mu_L = 22.436e-3 * e 
        mu_R = mu_L - e * V
        
        cfg_dummy = SimulationConfig(**base_kwargs)
        total_current = 0.0
        
        # Double sum over transversal modes
        for ny in range(1, 6):
            for nz in range(1, 6):
                E_trans = (cfg_dummy.hbar**2 / (2 * cfg_dummy.m_star)) * ((np.pi * ny / cfg_dummy.Ly)**2 + (np.pi * nz / cfg_dummy.Lz)**2)
                
                # Skip simulating modes that are absolutely empty (above Fermi limit)
                if E_trans > mu_L + 6 * k_B * Temp:
                    continue
                    
                mode_kwargs = {**base_kwargs, "n_y": ny, "n_z": nz}
                
                # Free-space normalization for this specific transversal mode
                # Since V0=0.6 barrier imposes a mathematically smaller dt_max than free-space, use the barrier dt!
                safe_dt = SimulationConfig(**{**mode_kwargs, "V_DC": V}).dt
                
                res_f = SimulationRunner.execute(**{**mode_kwargs, "V_DC": 0.0, "V0": 0.0, "dt": safe_dt}, disable_tqdm=True, record_history=False)
                res_b = SimulationRunner.execute(**{**mode_kwargs, "V_DC": V, "dt": safe_dt}, disable_tqdm=True, record_history=False)
                
                cfg = res_b["config"]
                psi_bar = res_b["time_signal_R"] + 1j * res_b["time_signal_I"]
                psi_free = res_f["time_signal_R"] + 1j * res_f["time_signal_I"]
                
                N_pad = cfg.nt * 8
                fft_bar = fft(psi_bar, n=N_pad)
                fft_free = fft(psi_free, n=N_pad)
                freqs = fftfreq(N_pad, cfg.dt)
                
                E_total_J = -(2 * np.pi * cfg.hbar * freqs)
                pos_mask = E_total_J > 0
                
                E_J = E_total_J[pos_mask]
                Psi_b = fft_bar[pos_mask]
                Psi_f = fft_free[pos_mask]
                
                U_obs_bar = cfg.U_R[res_b["record_ix"]]
                U_obs_free = res_f["config"].U_R[res_f["record_ix"]]
                
                valid = (E_J > U_obs_bar) & (E_J > U_obs_free)
                E_J = E_J[valid]
                
                k_bar = np.sqrt(2 * cfg.m_star * (E_J - U_obs_bar))
                k_free = np.sqrt(2 * cfg.m_star * (E_J - U_obs_free))
                
                T_E = (k_bar / k_free) * (np.abs(Psi_b[valid])**2 / np.abs(Psi_f[valid])**2)
                
                # Integration mask for Landauer window
                int_mask = (E_J > cfg.E_trans) & (E_J < mu_L + 6*k_B*Temp)
                E_int = E_J[int_mask]
                T_int = T_E[int_mask]
                
                def fermi(E, mu):
                    return 1.0 / (np.exp(np.clip((E - mu)/(k_B * Temp), -100, 100)) + 1.0)
                    
                f_L = fermi(E_int, mu_L)
                f_R = fermi(E_int, mu_R)
                
                I_mode = (2 * e / h) * np.trapezoid(T_int * (f_L - f_R), E_int)
                total_current += I_mode
                
        return -total_current

    @staticmethod
    def plot_IV(V_dc_arr, base_kwargs):
        print(f"Executing {len(V_dc_arr)} barrier biases with full (N,M) Landauer double summation...")
        func = partial(IVCharacteristic._run_bias, base_kwargs=base_kwargs)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            currents = list(tqdm(executor.map(func, V_dc_arr), total=len(V_dc_arr), desc="Extracting IV Curve"))
            
        plt.figure(figsize=(7, 4))
        plt.semilogy(V_dc_arr, currents, 'r-o', lw=2)
        # plt.plot(V_dc_arr, currents, 'r-o', lw=2)
        plt.title("Resonant Tunneling Diode I-V Characteristic")
        plt.xlabel("$V_{DC}$ (V)")
        plt.ylabel("Current (A)")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # === RUN EXPERIMENT ===
    # Als je T_tot te laag neemt dan zie je zwakkere versies van de piekjes (ik denk Q factor van de caviteit gwn)
    
    # # 1. Single Voltage Spectrum (Uncomment to view)
    results_barrier = SimulationRunner.execute(n_y=1, n_z=1, V0=0.6, V_DC=-0.2, T_total=1000.0e-15, E_target=0.55, frame_skip=500)
    results_free = SimulationRunner.execute(n_y=1, n_z=1, V0=0.0, V_DC=0.0, T_total=1000.0e-15, E_target=0.55, frame_skip=500, dt=results_barrier["config"].dt)
    TransmissionAnalyzer.plot_transmission(results_barrier, results_free)
    SimulationRunner.plot_animation(results_barrier)
    
    # 2. Extract I-V Curve showing Negative Differential Resistance
    # V_DC sweep from 0 to 100 mV (where NDR usually occurs for this well geometry)
    voltages = np.linspace(0, 0.05, 100)
    base_sim_kwargs = {
        "V0": 0.6, "T_total": 10000.0e-15, 
        "E_target": 0.022, # Centered near Fermi level (mu_L) to maximize resolution
        "frame_skip": 1000 # Only doing integration, not viewing animation
    }
    IVCharacteristic.plot_IV(voltages, base_sim_kwargs)
