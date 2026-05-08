from multiprocessing import synchronize
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
        self.__dict__.update(kwargs)
        
        # Physical Constants
        self.hbar   = 1.054571817e-34 
        self.m_e    = 9.1093837015e-31 
        self.e      = 1.602176634e-19  
        self.m_star = kwargs.get("m_star", 0.023 * self.m_e)
        
        # Domain Dimensions
        self.dx         = kwargs.get("dx", 0.5e-9)
        self.L_barriers = np.asarray(kwargs.get("L_barriers", [10e-9, 10e-9]))
        self.L_wells    = np.asarray(kwargs.get("L_wells", [30e-9]))
        self.L_buffer   = kwargs.get("L_buffer", 200e-9)
        self.L_absorb   = kwargs.get("L_absorb", 50e-9)
        self.L_total = (2 * self.L_absorb + 2 * self.L_buffer + 
                        np.sum(self.L_barriers) + np.sum(self.L_wells))
        
        self.nx = int(np.ceil(self.L_total / self.dx))
        self.x  = np.linspace(0, self.L_total, self.nx)
        
        # Define layer boundaries
        self.x_abs1 = self.L_absorb
        
        # Calculate barrier and well positions
        current_x = self.x_abs1 + self.L_buffer
        x_bars, x_wells = [], []
        
        for i in range(len(self.L_barriers)):
            x_bars.append(current_x)
            current_x += self.L_barriers[i]
            if i < len(self.L_wells):
                x_wells.append(current_x)
                current_x += self.L_wells[i]

        self.x_bars  = np.array(x_bars)
        self.x_wells = np.array(x_wells)
        self.x_buf2 = self.x_bars[-1] + self.L_barriers[-1] if len(self.L_barriers) > 0 else current_x
        self.x_abs2 = self.x_buf2 + self.L_buffer

        # Indices
        self.n_layer = np.round(self.x_abs1 / self.dx).astype(int) # Number of nodes in absorbing layer
        self.i_bars  = np.round(self.x_bars / self.dx).astype(int)
        self.i_buf2  = np.round(self.x_buf2 / self.dx).astype(int)

        # Transversal Energy
        self.Ly = getattr(self, "Ly", 40e-9)
        self.Lz = getattr(self, "Lz", 40e-9)
        self.n_y = getattr(self, "n_y", 1)
        self.n_z = getattr(self, "n_z", 1)
        self.E_trans = (self.hbar**2 / (2 * self.m_star)) * ((np.pi * self.n_y / self.Ly)**2 + (np.pi * self.n_z / self.Lz)**2)
        
        # Energy and Potentials
        self.V0 = np.atleast_1d(np.asarray(getattr(self, "V0", 0.6), dtype=float))* self.e # Evaluate V0 kwarg input as eV
        n_repeats = int(np.ceil(len(self.L_barriers) / self.V0.size))
        self.V0_barriers = np.tile(self.V0, n_repeats)[:len(self.L_barriers)]
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

        self.dt = kwargs.get("dt", 1 * dt_max) # Allow dt override to match domains
        self.T_total = getattr(self, "T_total", 100e-15)
        self.nt = int(np.ceil(self.T_total / self.dt))
        
        # Initial Wavepacket Setup
        # E_target = kinetic energy at the injection point (bias-independent).
        # This ensures k_x > 0 regardless of the applied V_DC.
        self.E_target  = getattr(self, "E_target", 0.2) * self.e  # KE at injection point
        self.x_0       = self.x_abs1 + self.L_buffer * 0.3
        self.sigma_x   = getattr(self, "sigma_x", 15e-9)
        i_x0           = int(self.x_0 / self.dx)
        self.total_E   = self.E_target + self.U_R[i_x0]  # True total energy = KE + U(x_0)
        self.k_x       = np.sqrt(2 * self.m_star * self.E_target) / self.hbar  # Always real & positive

    def _build_potentials(self):
        def add_barrier_potential(x_start, x_end, V_0):
            i_start = int(np.round(x_start / self.dx))
            i_end = int(np.round(x_end / self.dx))
            self.U_R[i_start:i_end] += V_0

        # Make the barriers
        for i in range(len(self.L_barriers)):
            add_barrier_potential(self.x_bars[i], self.x_bars[i] + self.L_barriers[i], self.V0_barriers[i])

        if self.V_DC != 0:
            # U = q·V = (-e)·V_DC; right contact is ground (V=0), left contact at V_DC
            bias = -self.e * self.V_DC
            self.U_R[:self.i_bars[0]] += bias
            
            # Device region: linear tilt from V_DC (left) to 0 (right/ground)
            x_dev = self.x[self.i_bars[0]:self.i_buf2]
            tilt = bias * (1.0 - (x_dev - self.x_bars[0]) / (self.x_buf2 - self.x_bars[0]))
            self.U_R[self.i_bars[0]:self.i_buf2] += tilt
            
        i_arr = np.arange(self.n_layer)
        dist_factor = ((self.n_layer - i_arr) / self.n_layer)**3
        abs_V = np.max(self.V0_barriers) if np.any(self.V0_barriers != 0) else 0.2 * self.e
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
        # The half-step phase offset compensates for the leapfrog staggering:
        # psi_R is evaluated at t=+dt/2 relative to psi_I at t=0, so we advance
        # the phase by (total_E * dt/2) / hbar to keep them in sync.
        self.psi_R = env * np.cos(phase + cfg.total_E * cfg.dt / (2 * cfg.hbar))

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
        k_f = np.sqrt(2 * cfg.m_star * (E - cfg.E_trans) + 0j) / cfg.hbar # k in free space
        
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
        
        M = np.eye(2, dtype=np.complex128)[None, :, :].repeat(N, axis=0)
        
        for i in range(len(cfg.L_barriers)):
            k_b_i = np.sqrt(2 * cfg.m_star * (E - (cfg.V0_barriers[i] + cfg.E_trans)) + 0j) / cfg.hbar
            M = M @ intf(k_f, k_b_i)
            M = M @ prop(k_b_i, cfg.L_barriers[i])
            M = M @ intf(k_b_i, k_f)
            if i < len(cfg.L_wells):
                M = M @ prop(k_f, cfg.L_wells[i])
            
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
        
        T_analy = TransmissionAnalyzer.get_analytical_T(E_eV_plot, cfg)
        plt.plot(E_eV_plot, T_analy, 'k--', lw=1.5, label="Analytical (V_DC=0)")
        
        k_0 = cfg.k_x
        sigma_k = 1.0 / (2.0 * cfg.sigma_x)
        sigma_E_eV = ((cfg.hbar**2 * k_0 / cfg.m_star) * sigma_k) / cfg.e
        E_center_eV = cfg.total_E / cfg.e  # plot window centered on total energy of the packet
        
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
        y_max = max(np.max(cfg.V0) / cfg.e, np.max(U_R_eV)) * 1.5
        ax2.set_ylim(y_min, y_max)
        
        ax2.axvspan(cfg.x_bars[0]*1e9, cfg.x_buf2*1e9, color='gray', alpha=0.1, label='Device Region')
        
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
    def _run_bias(V, base_kwargs): # Removed the pre-computed fft_free and U_obs_free
        e = 1.602176634e-19
        h = 6.62607015e-34
        k_B = 1.380649e-23
        Temp = 277.0 # 4K for sharp Fermi edge
        mu_L = 22.436e-3 * e 
        mu_R = mu_L - e * V
        
        # 1. Run the barrier simulation for this specific bias
        cfg_barrier = {**base_kwargs, "n_y": 0, "n_z": 0, "V_DC": V}
        res_b = SimulationRunner.execute(**cfg_barrier, disable_tqdm=True, record_history=False)
        
        # 2. Run the free space simulation for the EXACT SAME bias (V0=0.0 removes barriers)
        cfg_free = {**base_kwargs, "n_y": 0, "n_z": 0, "V_DC": V, "V0": 0.0}
        res_f = SimulationRunner.execute(**cfg_free, disable_tqdm=True, record_history=False)
        
        cfg = res_b["config"]
        
        # Extract signals for both
        psi_bar = res_b["time_signal_R"] + 1j * res_b["time_signal_I"]
        psi_free = res_f["time_signal_R"] + 1j * res_f["time_signal_I"]
        
        # Use 16x padding for incredibly smooth energy resolution
        N_pad = cfg.nt * 16
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
                
        # Calculate transmission perfectly aligned in energy
        T_E = (k_bar / k_free) * (np.abs(Psi_b[valid])**2 / np.abs(Psi_f[valid])**2)
        
        # Sort to ensure np.trapezoid integrates forward
        sort_idx = np.argsort(E_J)
        E_J = E_J[sort_idx]
        T_E = T_E[sort_idx]

        total_current = 0.0
        cfg_base = SimulationConfig(**base_kwargs)

        def fermi(E, mu):
            return 1.0 / (np.exp(np.clip((E - mu)/(k_B * Temp), -100, 100)) + 1.0)

        for ny in range(1,15):
            for nz in range(1,15):
                E_trans = (cfg_base.hbar**2 / (2 * cfg_base.m_star)) * ((np.pi * ny / cfg_base.Ly)**2 + (np.pi * nz / cfg_base.Lz)**2)
                if E_trans > mu_L + 6 * k_B * Temp:
                    continue

                E_total = E_trans + E_J
                f_L = fermi(E_total, mu_L)
                f_R = fermi(E_total, mu_R)
                
                I_mode = (2 * e / h) * np.trapezoid(T_E * (f_L - f_R), E_J)
                total_current += I_mode
                
        return total_current

    @staticmethod
    def plot_IV(V_dc_arr, base_kwargs):
        # Initialize a dummy config to grab the globally safe dt for all runs
        # We need a stable dt so arrays match up during FFTs
        cfg_dummy = SimulationConfig(**base_kwargs, n_y=0, n_z=0, V_DC=max(V_dc_arr))
        safe_dt = cfg_dummy.dt
        base_kwargs["dt"] = safe_dt 
        
        print(f"Executing {len(V_dc_arr)} biases (Barrier & Free Space) with optimized Landauer factorization...")

        # We no longer pass pre-computed reference data to the partial function
        func = partial(IVCharacteristic._run_bias, base_kwargs=base_kwargs)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            currents = list(tqdm(executor.map(func, V_dc_arr), total=len(V_dc_arr), desc="Extracting IV Curve"))
            
        plt.figure(figsize=(8, 5))
        plt.semilogy(V_dc_arr * 1000, np.array(currents) * 1e6, 'r-o', lw=2)
        plt.title("Resonant Tunneling Diode I-V Characteristic")
        plt.xlabel("$V_{DC}$ (mV)")
        plt.ylabel("Current (µA)")
        plt.grid(True, which="both", ls="--")
        plt.show()

if __name__ == '__main__':
    # === RUN EXPERIMENT ===
    # Als je T_tot te laag neemt dan zie je zwakkere versies van de piekjes (ik denk Q factor van de caviteit gwn)
    
    # # 1. Single Voltage Spectrum (Uncomment to view)
    # Double_barrier = True
    # if Double_barrier:
    #     results_barrier = SimulationRunner.execute(n_y=1, 
    #     n_z=1, 
    #     V0=0.6, 
    #     V_DC=-0.0, 
    #     T_total=10000.0e-15, 
    #     E_target=0.5, 
    #     frame_skip=500)

    #     results_free = SimulationRunner.execute(n_y=1, 
    #     n_z=1, 
    #     V0=0.0, 
    #     V_DC=0.0, 
    #     T_total=10000.0e-15, 
    #     E_target=0.5, 
    #     frame_skip=500, 
    #     dt=results_barrier["config"].dt)
    #     TransmissionAnalyzer.plot_transmission(results_barrier, results_free)
    #     SimulationRunner.plot_animation(results_barrier)
    
    
    # Three_barriers = True
    # if Three_barriers:
    #     results_barrier = SimulationRunner.execute(L_barriers = [5e-9, 5e-9, 5e-9],
    #     L_wells = [15e-9, 15e-9], 
    #     n_y=1, 
    #     n_z=1, 
    #     V0=0.6, 
    #     V_DC=0.0, 
    #     T_total=5000.0e-15, 
    #     E_target=0.35, 
    #     frame_skip=500)

    #     results_free = SimulationRunner.execute(L_barriers = [5e-9, 5e-9, 5e-9],
    #     L_wells = [15e-9, 15e-9], 
    #     n_y=1, 
    #     n_z=1, 
    #     V0=0.0, 
    #     V_DC=0.0, 
    #     T_total=5000.0e-15, 
    #     E_target=0.35, 
    #     frame_skip=500, 
    #     dt=results_barrier["config"].dt)
    #     TransmissionAnalyzer.plot_transmission(results_barrier, results_free)
    #     SimulationRunner.plot_animation(results_barrier)
    
    # 2. Extract I-V Curve showing Negative Differential Resistance
    # V_DC sweep from 0 to 100 mV (where NDR usually occurs for this well geometry)
    do_IV_curve = True
    if do_IV_curve:
        voltages = np.linspace(0.1, 0.15, 75)
        base_sim_kwargs = {
            "V0": 0.6, "T_total": 10000.0e-15, 
            "E_target": 0.022346, # Centered near Fermi level (mu_L) to maximize resolution
            "frame_skip": 1000 # Only doing integration, not viewing animation
        }
        IVCharacteristic.plot_IV(voltages, base_sim_kwargs)
