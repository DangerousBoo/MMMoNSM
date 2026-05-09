import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from scipy.optimize import brentq
from tqdm import tqdm
import concurrent.futures
from functools import partial

class SimulationConfig:
    """Handles physical constants, geometry, and simulation variables for 1D FDTD."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
        # Physical Constants
        self.hbar   = 1.054571817e-34
        self.e      = 1.602176634e-19
        self.m_star = kwargs.get("m_star", 0.023 * 9.1093837015e-31)
        
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
        self.Ly = kwargs.get("Ly", 40e-9)
        self.Lz = kwargs.get("Lz", 40e-9)
        self.n_y = kwargs.get("n_y", 1)
        self.n_z = kwargs.get("n_z", 1)
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

        self.dt      = kwargs.get("dt", dt_max)
        self.T_total = kwargs.get("T_total", 100e-15)
        self.nt      = int(np.ceil(self.T_total / self.dt))
        
        # Wavepacket: E_target is kinetic energy at injection (bias-independent, always > 0)
        self.E_target = kwargs.get("E_target", 0.2) * self.e
        self.sigma_x  = kwargs.get("sigma_x", 15e-9)
        self.x_0      = self.x_abs1 + self.L_buffer * 0.3
        i_x0          = int(self.x_0 / self.dx)
        self.total_E  = self.E_target + self.U_R[i_x0]
        self.k_x      = np.sqrt(2 * self.m_star * self.E_target) / self.hbar

        # Absorbers must be built after E_target is known so W_max scales correctly
        self._build_absorbers()

    def _build_absorbers(self):
        lambda_dB = 2 * np.pi / self.k_x
        n_layer  = int(np.ceil(4 * lambda_dB / self.dx))

        i_arr = np.arange(n_layer)
        dist_factor = ((n_layer - i_arr) / n_layer) ** 4
        W_max = 2.0 * self.E_target
        self.U_I[:n_layer]  = W_max * dist_factor
        self.U_I[-n_layer:] = W_max * dist_factor[::-1]

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
            

class SchrodingerSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.order = cfg.order

        denom = (1 + 0.5 * cfg.dt * cfg.U_I / cfg.hbar)
        self.c_A = (1 - 0.5 * cfg.dt * cfg.U_I / cfg.hbar) / denom
        self.c_B = (cfg.dt / cfg.hbar) / denom
        self.c_lap = (cfg.hbar * cfg.dt) / (2 * cfg.m_star * cfg.dx**2) / denom
        
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
        self.psi_R = (self.c_A * self.psi_R - self.c_lap * self._lap(self.psi_I) + self.c_B * (self.cfg.U_R) * self.psi_I)
        self.psi_I = (self.c_A * self.psi_I + self.c_lap * self._lap(self.psi_R) - self.c_B * (self.cfg.U_R) * self.psi_R)

    @property
    def density(self):
        return self.psi_R**2 + self.psi_I**2

class TransmissionAnalyzer:
    """Computes and plots both numerical FDTD and analytical TMM transmission spectra."""
    @staticmethod
    def _finite_well_levels(cfg, E_low_eV, E_high_eV):
        """Approximate quasi-bound levels with a finite square-well model."""
        if len(cfg.L_wells) == 0 or len(cfg.L_barriers) < 2:
            return np.array([], dtype=float)

        levels_eV = []
        eps = 1e-12

        for j, Lw in enumerate(cfg.L_wells):
            if j >= len(cfg.x_wells) or j + 1 >= len(cfg.L_barriers): continue
            iw = (np.round(cfg.x_wells[j] / cfg.dx).astype(int), np.round((cfg.x_wells[j] + Lw) / cfg.dx).astype(int))
            ibl = (np.round(cfg.x_bars[j] / cfg.dx).astype(int), np.round((cfg.x_bars[j] + cfg.L_barriers[j]) / cfg.dx).astype(int))
            ibr = (np.round(cfg.x_bars[j+1] / cfg.dx).astype(int), np.round((cfg.x_bars[j+1] + cfg.L_barriers[j+1]) / cfg.dx).astype(int))
            
            U_w   = float(np.mean(cfg.U_R[iw[0]:iw[1]]))
            V_eff = min(np.max(cfg.U_R[ibl[0]:ibl[1]]), np.max(cfg.U_R[ibr[0]:ibr[1]])) - U_w
            if iw[1] <= iw[0] + 2 or V_eff <= 0: continue

            z0 = 0.5 * Lw * np.sqrt(2.0 * cfg.m_star * V_eff) / cfg.hbar
            sq = lambda z: np.sqrt(max(z0**2 - z**2, 0.0))
            funcs = [lambda z: z * np.tan(z) - sq(z),      # even parity
                     lambda z: -z / np.tan(z) - sq(z)]     # odd parity

            for n in range(int(np.floor(2.0 * z0 / np.pi)) + 2):
                a, b = n * np.pi / 2.0 + eps, min((n + 1) * np.pi / 2.0 - eps, z0 - eps)
                if a >= z0 or b <= a: continue
                f = funcs[n % 2]
                try:
                    fa, fb = f(a), f(b)
                    if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0): continue
                    z = brentq(f, a, b, maxiter=200)
                except Exception:
                    continue
                E_abs_eV = (U_w + 2.0 * cfg.hbar**2 * z**2 / (cfg.m_star * Lw**2)) / cfg.e
                if E_low_eV <= E_abs_eV <= E_high_eV:
                    levels_eV.append(E_abs_eV)

        if not levels_eV:
            return np.array([], dtype=float)
        levels = np.sort(np.array(levels_eV))
        return levels[np.concatenate(([True], np.diff(levels) > 5e-4))]  # deduplicate

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
        E_eV_plot, T, T_analy, cfg, sigma_E_eV, E_center_eV = \
            TransmissionAnalyzer.compute_T(results_barrier, results_free)

        E_min = E_center_eV - 3 * sigma_E_eV
        E_max = E_center_eV + 3 * sigma_E_eV
        finite_well_res = TransmissionAnalyzer._finite_well_levels(cfg, 0, E_eV_plot.max())

        plt.figure(figsize=(8, 4))
        plt.plot(E_eV_plot, T, 'm-', lw=2, label="FDTD Simulation")

        # Dense analytical curve so narrow resonance peaks are smooth
        E_dense = np.linspace(E_min, E_max, 2000)
        T_dense = TransmissionAnalyzer.get_analytical_T(E_dense, cfg)
        plt.plot(E_dense, T_dense, 'k--', lw=1.5, label="Analytical (V_DC=0)")

        plt.axvline(E_min, color='r', linestyle='--', alpha=0.6, label=r'$\pm 3\sigma_E$ width')
        plt.axvline(E_max, color='r', linestyle='--', alpha=0.6)
        for V_barrier in cfg.V0_barriers / cfg.e:
            plt.axvline(V_barrier, color='g', linestyle=':', alpha=0.7)
        plt.plot([], [], color='g', linestyle=':', alpha=0.7, label='Barrier Heights (eV)')
        for res in finite_well_res:
            plt.axvline(res, color='b', linestyle='-.', alpha=0.7)
        plt.plot([], [], color='b', linestyle='-.', alpha=0.7, label='Finite Well Levels (eV)')

        plt.xlim(E_min, E_max)
        plt.ylim(0, 1.1)
        plt.title("Transmission Spectrum $T(E)$")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission Coefficient T")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def compute_T(results_barrier, results_free):
        """
        T(E) = |J_bar(E)| / |J_free(E)|, where
        J(ω) = (ℏ/m) Im[Ψ*(ω) · DΨ(ω)], with Ψ = FFT(ψ) and DΨ = FFT(∂ψ/∂x).
        ψ and ∂ψ/∂x are recorded separately so they can be FFT'd independently,
        avoiding the bilinear cross-terms that corrupt FFT(j(t)).
        """
        cfg      = results_barrier["config"]
        cfg_free = results_free["config"]

        def spectral_current(res, c, N_pad):
            Psi  = fft(res["psi_R"]  + 1j * res["psi_I"],  n=N_pad)
            DPsi = fft(res["dpsi_R"] + 1j * res["dpsi_I"], n=N_pad)
            return (c.hbar / c.m_star) * np.imag(np.conj(Psi) * DPsi)

        N_pad = cfg.nt * 1
        freqs = fftfreq(N_pad, cfg.dt)
        E_all = -(2 * np.pi * cfg.hbar * freqs) / cfg.e

        J_bar  = spectral_current(results_barrier, cfg,      N_pad)
        J_free = spectral_current(results_free,    cfg_free, N_pad)

        pos_mask = E_all > 0
        E_eV     = E_all[pos_mask]
        E_J      = E_eV * cfg.e
        J_bar    = J_bar[pos_mask]
        J_free   = J_free[pos_mask]

        U_obs_bar  = cfg.U_R[results_barrier["record_ix"]]
        U_obs_free = cfg_free.U_R[results_free["record_ix"]]
        valid_E    = (E_J > U_obs_bar) & (E_J > U_obs_free)
        E_eV_plot  = E_eV[valid_E]

        T = np.abs(J_bar[valid_E]) / np.abs(J_free[valid_E])
        T_analy = TransmissionAnalyzer.get_analytical_T(E_eV_plot, cfg)

        sigma_E_eV  = ((cfg.hbar**2 * cfg.k_x / cfg.m_star) / (2.0 * cfg.sigma_x)) / cfg.e
        E_center_eV = cfg.total_E / cfg.e

        return E_eV_plot, T, T_analy, cfg, sigma_E_eV, E_center_eV

    @staticmethod
    def plot_transmission_sweep(sweep, title="Transmission Sweep", analytical=True):
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(10, 5))

        E_mins, E_maxs = [], []
        entries = []  # (color, label, cfg) stored for post-loop plotting

        for idx, (label, res_bar, res_free) in enumerate(sweep):
            color = cmap(idx % 10)
            E_eV_plot, T, _, cfg, sigma_E_eV, E_center_eV = \
                TransmissionAnalyzer.compute_T(res_bar, res_free)

            ax.plot(E_eV_plot, T, color=color, lw=2, label=f"{label} (FDTD)")
            E_mins.append(E_center_eV - 3 * sigma_E_eV)
            E_maxs.append(E_center_eV + 3 * sigma_E_eV)
            entries.append((color, label, cfg))

        # Full x-range known after loop — dense analytical spans beyond the window
        x_lo, x_hi = min(E_mins), max(E_maxs)
        margin = (x_hi - x_lo) * 0.2
        E_dense = np.linspace(x_lo - margin, x_hi + margin, 3000)

        if analytical:
            for color, label, cfg in entries:
                T_dense = TransmissionAnalyzer.get_analytical_T(E_dense, cfg)
                ax.plot(E_dense, T_dense, color=color, lw=1.2, ls='--', alpha=0.7,
                        label=f"{label} (Analytical)")

        # Finite well levels — gray so they don't clash with any sweep color
        well_legend_added = False
        for _, _, cfg in entries:
            levels = TransmissionAnalyzer._finite_well_levels(cfg, x_lo, x_hi)
            for E_level in levels:
                ax.axvline(E_level, color='gray', linestyle='-.', lw=1.0, alpha=0.8,
                           label='Finite Well Levels' if not well_legend_added else '_nolegend_')
                well_legend_added = True

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Transmission Coefficient T")
        ax.legend(fontsize=8)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

class SimulationRunner:
    @staticmethod
    def execute(frame_skip=100, record_ix=None, disable_tqdm=False, record_history=True, **kwargs):
        cfg = SimulationConfig(**kwargs)
        solver = SchrodingerSolver(cfg)

        n_frames = int(np.ceil(cfg.nt / frame_skip)) if record_history else 0
        history  = np.zeros((n_frames, cfg.nx), dtype=np.float32) if record_history else None
        record_ix = record_ix or int(cfg.x_buf2 / cfg.dx) + int(20e-9 / cfg.dx)
        # ix-1 and ix+1 are in the same free-space buffer region (only ±0.5 nm away)
        ix = record_ix

        # Record ψ and ∂ψ/∂x separately — combined in freq. domain to avoid bilinear cross-terms
        rec_psi_R  = np.zeros(cfg.nt)
        rec_psi_I  = np.zeros(cfg.nt)
        rec_dpsi_R = np.zeros(cfg.nt)
        rec_dpsi_I = np.zeros(cfg.nt)
        frame_idx  = 0

        for it in tqdm(range(cfg.nt), desc=f"Simulating (nt={cfg.nt})", disable=disable_tqdm):
            solver.step()
            rec_psi_R[it]  = solver.psi_R[ix]
            rec_psi_I[it]  = solver.psi_I[ix]
            rec_dpsi_R[it] = (solver.psi_R[ix+1] - solver.psi_R[ix-1]) / (2 * cfg.dx)
            rec_dpsi_I[it] = (solver.psi_I[ix+1] - solver.psi_I[ix-1]) / (2 * cfg.dx)

            if record_history and not it % frame_skip and frame_idx < n_frames:
                history[frame_idx] = solver.density
                frame_idx += 1
        # Check if wavepacket has cleared the device region (trapped state warning)
        device_prob = np.sum(solver.density[cfg.i_bars[0]:cfg.i_buf2]) * cfg.dx
        if device_prob > 0.01:
            print(f"Warning: wavepacket still in device region (P={device_prob:.3f}). Increase T_total.")

        return {
            "config":    cfg,
            "history":   history,
            "frame_skip":frame_skip,
            "record_ix": record_ix,
            "psi_R":  rec_psi_R,  "psi_I":  rec_psi_I,
            "dpsi_R": rec_dpsi_R, "dpsi_I": rec_dpsi_I,
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
    def _run_bias(V, base_kwargs):
        e, h, k_B = 1.602176634e-19, 6.62607015e-34, 1.380649e-23
        Temp = 4.0  # 4 K for sharp Fermi edge
        mu_L = 22.436e-3 * e
        mu_R = mu_L - e * V

        res_f = SimulationRunner.execute(**{**base_kwargs, 'V0': 0.0, 'n_y': 0, 'n_z': 0, 'V_DC': V},
                                         disable_tqdm=True, record_history=False)
        res_b = SimulationRunner.execute(**{**base_kwargs, 'n_y': 0, 'n_z': 0, 'V_DC': V},
                                         disable_tqdm=True, record_history=False)

        def spec_J(res, c, N):
            Psi  = fft(res["psi_R"]  + 1j * res["psi_I"],  n=N)
            DPsi = fft(res["dpsi_R"] + 1j * res["dpsi_I"], n=N)
            return (c.hbar / c.m_star) * np.imag(np.conj(Psi) * DPsi)

        cfg = res_b["config"]
        N_pad = cfg.nt * 6
        freqs = fftfreq(N_pad, cfg.dt)
        E_J   = -(2 * np.pi * cfg.hbar * freqs)

        J_bar  = spec_J(res_b, cfg,             N_pad)
        J_free = spec_J(res_f, res_f["config"], N_pad)

        pos_mask   = E_J > 0
        E_J        = E_J[pos_mask]
        j_xb       = J_bar[pos_mask]
        j_xf       = J_free[pos_mask]

        U_obs_bar  = cfg.U_R[res_b["record_ix"]]
        U_obs_free = res_f["config"].U_R[res_f["record_ix"]]

        # Only use bins where the free-space signal is above the noise floor.
        # In the wavepacket wings J_free → 0; dividing by near-zero gives T >> 1
        # and contaminates the Landauer integral — this gets WORSE with longer T_total
        # because the FFT has more bins in the noisy wings.
        spectral_mask = np.abs(j_xf) > 1e-3 * np.max(np.abs(j_xf))
        valid = (E_J > U_obs_bar) & (E_J > U_obs_free) & spectral_mask
        E_J = E_J[valid]

        T_E = np.abs(j_xb[valid]) / np.abs(j_xf[valid])

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
    def compute_IV_curve(V_dc_arr, base_kwargs):
        print(f"Executing {len(V_dc_arr)} barrier biases with optimized Landauer factorization...")

        func = partial(IVCharacteristic._run_bias, base_kwargs=base_kwargs)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            currents = list(tqdm(executor.map(func, V_dc_arr), total=len(V_dc_arr), desc="Extracting IV Curve"))
        
        return np.array(currents)

    @staticmethod
    def plot_IV(V_dc_arr, base_kwargs):
        
        currents = IVCharacteristic.compute_IV_curve(V_dc_arr, base_kwargs)
            
        plt.figure(figsize=(8, 5))
        plt.semilogy(V_dc_arr * 1000, currents * 1e6, 'r-o', lw=2)
        plt.title("Resonant Tunneling Diode I-V Characteristic")
        plt.xlabel("$V_{DC}$ (mV)")
        plt.ylabel("Current (µA)")
        plt.grid(True, which="both", ls="--")
        plt.show()

    @staticmethod
    def plot_IV_sweep(voltages, sweep, title="IV Sweep"):
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, (label, currents) in enumerate(sweep):
            color = cmap(idx % 10)
            ax.semilogy(voltages * 1000, np.array(currents) * 1e6,
                        color=color, lw=2, marker='o', label=label)

        ax.set_title(title)
        ax.set_xlabel("$V_{DC}$ (mV)")
        ax.set_ylabel("Current (µA)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    # --- Single T(E) spectrum ---
    do_single = True
    if do_single:
        kw = dict(n_y=1, n_z=1, V0=0.6, V_DC=0.0, T_total=3000e-15, E_target=0.35, frame_skip=500)
        res_bar  = SimulationRunner.execute(**kw)
        res_free = SimulationRunner.execute(**{**kw, 'V0': 0.0}, dt=res_bar['config'].dt)
        TransmissionAnalyzer.plot_transmission(res_bar, res_free)
        SimulationRunner.plot_animation(res_bar)

    # --- I-V curve (NDR) ---
    do_IV_curve = True
    if do_IV_curve:
        voltages = np.linspace(0.1, 0.12, 50)
        IVCharacteristic.plot_IV(voltages, {"V0": 0.6, "T_total": 5000e-15, "E_target": 0.02234})

    # =========================================================================
    # === PARAMETER SWEEPS ===
    # =========================================================================

    def run_pair(label, shared_kwargs, frame_skip=500, animate=False):
        """Run barrier + free-space pair and return (label, res_bar, res_free)."""
        res_bar  = SimulationRunner.execute(**shared_kwargs, frame_skip=frame_skip)
        res_free = SimulationRunner.execute(
            **{**shared_kwargs, "V0": 0.0, "V_DC": 0.0},
            frame_skip=frame_skip,
            dt=res_bar["config"].dt)
        if animate:
            SimulationRunner.plot_animation(res_bar)
        return (label, res_bar, res_free)

    # --- Sweep V0 (barrier height) ---
    sweep_V0 = False
    if sweep_V0:
        base = dict(n_y=1, n_z=1, V_DC=0.0,
                    L_barriers=[10e-9, 10e-9], L_wells=[30e-9],
                    T_total=1000e-15, E_target=0.35)
        sweep_transmission = [run_pair(f"V0 = {v} eV", {**base, "V0": v})
                              for v in [0.2, 0.4, 0.6]]
        TransmissionAnalyzer.plot_transmission_sweep(sweep_transmission, title="Sweep: Barrier Height V0")
        
        iv_voltages = np.linspace(0.1, 0.12, 50)
        sweep_IV = [
            (f"V0 = {v} eV", IVCharacteristic.compute_IV_curve(iv_voltages, {**base, "V0": v, "E_target": 0.022}))
            for v in [0.2, 0.4, 0.6]
        ]
        IVCharacteristic.plot_IV_sweep(iv_voltages, sweep_IV, title="IV Sweep: Barrier Height V0")

    sweep_T_total = True
    if sweep_T_total:
        base = dict(n_y=1, n_z=1, V_DC=0.0,
                    L_barriers=[10e-9, 10e-9], L_wells=[30e-9],
                    V0=0.6, E_target=0.35)
        sweep_transmission = [run_pair(f"T_total = {v}", {**base, "T_total": v})
                              for v in [500e-15,1000e-15,2000e-15,5000e-15]]
        TransmissionAnalyzer.plot_transmission_sweep(sweep_transmission, title="Sweep: Total Simulation Time T_total")
        
        iv_voltages = np.linspace(0.1, 0.12, 50)
        sweep_IV = [
            (f"T_total = {v}", IVCharacteristic.compute_IV_curve(iv_voltages, {**base, "T_total": v, "E_target": 0.022}))
            for v in [500e-15,1000e-15,2000e-15,5000e-15]
        ]
        IVCharacteristic.plot_IV_sweep(iv_voltages, sweep_IV, title="IV Sweep: Total Simulation Time T_total")

    # --- Sweep L_wells (well length) ---
    sweep_well = False
    if sweep_well:
        base = dict(n_y=1, n_z=1, V0=0.6, V_DC=0.0,
                    L_barriers=[10e-9, 10e-9],
                    T_total=2000e-15, E_target=0.35)
        sweep = [run_pair(f"Lw = {int(Lw*1e9)} nm", {**base, "L_wells": [Lw]})
                 for Lw in [15e-9, 30e-9, 50e-9]]
        TransmissionAnalyzer.plot_transmission_sweep(sweep, title="Sweep: Well Length")

    # --- Sweep L_barriers (barrier length, same for both barriers) ---
    sweep_barrier = False
    if sweep_barrier:
        base = dict(n_y=1, n_z=1, V0=0.6, V_DC=0.0,
                    L_wells=[30e-9],
                    T_total=2000e-15, E_target=0.35)
        sweep = [run_pair(f"Lb = {int(Lb*1e9)} nm", {**base, "L_barriers": [Lb, Lb]})
                 for Lb in [5e-9, 10e-9, 20e-9]]
        TransmissionAnalyzer.plot_transmission_sweep(sweep, title="Sweep: Barrier Length")

    # --- Sweep E_target (wavepacket centre energy) ---
    sweep_Etarget = False
    if sweep_Etarget:
        base = dict(n_y=1, n_z=1, V0=0.6, V_DC=0.0,
                    L_barriers=[10e-9, 10e-9], L_wells=[30e-9],
                    T_total=2000e-15)
        sweep = [run_pair(f"E_target = {E} eV", {**base, "E_target": E})
                 for E in [0.15, 0.25, 0.35, 0.45]]
        TransmissionAnalyzer.plot_transmission_sweep(sweep, title="Sweep: Probe Energy E_target")

