import time
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import scipy.special as sp_special
from scipy.fft import fft, fftfreq

class FCI_TM_Solver:
    def __init__(self, Nx, Ny, Nt, lambda0, CFL, bc='PEC', solver="Schur", finesse=20):
        # Parameters
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.lambda0, self.CFL = lambda0, CFL
        self.finesse = finesse
        self.dx, self.dy = lambda0 / self.finesse, lambda0 / self.finesse
        self.bc = bc.upper()
        self.solver = solver

        # PBC drops the redundant N+1 boundary node to form a perfect ring
        self.nx_n = self.Nx if self.bc == 'PBC' else self.Nx + 1
        self.ny_n = self.Ny if self.bc == 'PBC' else self.Ny + 1

        # Material shizzle
        self.eps, self.mu, self.c = 8.854e-12, 1.256e-6, 3e8
        self.Z0 = np.sqrt(self.mu / self.eps)

        self.sigma = np.zeros((self.nx_n, self.ny_n)) #Define Drude media (so sigma_DC)
        self.gamma = 0.0
        self.epsilon_r  = np.ones((self.nx_n, self.ny_n))

        self.v_local = self.c / np.sqrt(self.epsilon_r)
        self.Z_local = self.Z0 / np.sqrt(self.epsilon_r)
        
        # Source Parameters
        self.f_c    = self.c / self.lambda0
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t

        # Component lengths
        self.dt = CFL / (self.c * np.sqrt(1/self.dx**2 + 1/self.dy**2))
        self.len_hx = self.nx_n * self.Ny
        self.len_hy = self.Nx * self.ny_n
        self.len_ez = self.nx_n * self.ny_n
        
        # Hx & Hx_dot + Hy & Hy_dot + Ez & Ez_dot & Ez_ddot & Jz:
        self.total_len = 2 * self.len_hx + 2 * self.len_hy + 4 * self.len_ez
        ez_start = 2*self.len_hx + 2*self.len_hy
        self.idx_ez = slice(ez_start, ez_start + self.len_ez)
        
        self._build_system()

    def my_src(self, t):
        # Gaussian Pulse: A * cos(2*pi*fc*(t-t0)) * exp(-0.5 * ((t-t0)/sig_t)**2)
        return self.A * np.cos(2 * np.pi * self.f_c * (t - self.t0)) * \
               np.exp(-0.5 * ((t - self.t0) / self.sig_t)**2)

    def _get_pml_profiles(self):
        self.kx = np.ones((self.nx_n,self.ny_n))
        self.ky = np.ones((self.nx_n,self.ny_n))
        self.sx = np.zeros((self.nx_n,self.ny_n))
        self.sy = np.zeros((self.nx_n,self.ny_n))

        p, m = int(1 * self.finesse), 4
        k_max = 2
        s_max = (m + 1) / (150 * np.pi * min([self.dx, self.dy]))

        for i in range(p):
            d_pml = (p - i) / p
            val_k = 1.0 + (k_max - 1.0) * (d_pml**m)
            val_s = s_max * (d_pml**m)

            self.kx[i, :], self.kx[-1-i, :] = val_k, val_k
            self.ky[:, i], self.ky[:, -1-i] = val_k, val_k
            self.sx[i, :], self.sx[-1-i, :] = val_s, val_s
            self.sy[:, i], self.sy[:, -1-i] = val_s, val_s
        
        return self.sx, self.sy, self.kx, self.ky
    
    def _get_operators(self, n, d):
        if self.bc == 'PBC':
            ones = np.ones(n)
            Ix = sp.eye(n, format='csr')
            
            # Eq 2.133: derivative is (f_{j+1} - f_j) / Δy  → forward difference, divisor = d (not 2d)
            Dx  = sp.diags_array([-ones, ones, ones], offsets=[0, 1, 1-n], shape=(n, n), format='csr') / d
            Dtx = sp.diags_array([ones, -ones, -ones], offsets=[0, -1, n-1], shape=(n, n), format='csr') / d
            
            A1 = sp.diags_array([ones, ones, ones], offsets=[0, -1, n-1], shape=(n, n), format='csr')
            A2 = sp.diags_array([ones, ones, ones], offsets=[0, 1, 1-n], shape=(n, n), format='csr')
            
            return Ix, Dx, Dtx, A1, A2
        
        else:
            Ix = sp.eye(n + 1, format='csr')

            # PEC: forward difference Ez→H over one cell spacing, divisor = d
            Dx  = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
            Dtx = ((sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)) / d).tolil()
            Dtx[n, n-1] = -1 / d  # one-sided at PEC wall

            A1 = sp.diags_array([1, 1], offsets=[0, -1], shape=(n, n-1), format='csr')
            A2 = sp.diags_array([1, 1], offsets=[0, 1], shape=(n, n+1), format='csr')

            return Ix, Dx.tocsr(), Dtx.tocsr(), A1.tocsr(), A2.tocsr()

    def _build_system(self):
        v, c, Z, gamma, dt = self.v_local, self.c, self.Z_local, self.gamma, self.dt
        sx, sy, kx, ky = self._get_pml_profiles()

        Ix, Dx, Dtx, Ax1, Ax2 = self._get_operators(self.Nx, self.dx)
        Iy, Dy, Dty, Ay1, Ay2 = self._get_operators(self.Ny, self.dy)
        
        0.5 * DY_ez_to_hx = sp.kron(Ix, Dy)  # Size: len_hx x len_ez
        0.5 * DX_ez_to_hy = sp.kron(Dx, Iy)  # Size: len_hy x len_ez
        0.5 * DY_hx_to_ez = sp.kron(Ix, Dty) # Size: len_ez x len_hx
        0.5 * DX_hy_to_ez = sp.kron(Dtx, Iy) # Size: len_ez x len_hy

        I_hx = sp.eye(self.len_hx, format='csr')
        I_hy = sp.eye(self.len_hy, format='csr')
        I_ez = sp.eye(self.len_ez, format='csr')

        # Eq 2.140: β_d^± = κd/Δτ ± Z0·σd/2  — always free-space c and Z0, never local v/Z
        bxp = (kx / (c * dt) + self.Z0 * sx / 2.0).flatten()
        byp = (ky / (c * dt) + self.Z0 * sy / 2.0).flatten()
        bxm = (kx / (c * dt) - self.Z0 * sx / 2.0).flatten()
        bym = (ky / (c * dt) - self.Z0 * sy / 2.0).flatten()

        # PBC: collocated grid → len_hx = len_hy = len_ez = nx_n*ny_n, no interpolation needed.
        # PEC: Interp_Y/X would be needed to map (Nx+1)*(Ny+1) Ez grid to (Nx+1)*Ny Hx grid.
        if self.bc == 'PBC':
            bxp_hx = bxp;  byp_hx = byp;  bxm_hx = bxm;  bym_hx = bym
            bxp_hy = bxp;  byp_hy = byp;  bxm_hy = bxm;  bym_hy = bym
        else:
            Interp_Y = sp.kron(Ix, Ay2) * 0.5
            Interp_X = sp.kron(Ax2, Iy) * 0.5
            bxp_hx = Interp_Y.dot(bxp);  byp_hx = Interp_Y.dot(byp)
            bxm_hx = Interp_Y.dot(bxm);  bym_hx = Interp_Y.dot(bym)
            bxp_hy = Interp_X.dot(bxp);  byp_hy = Interp_X.dot(byp)
            bxm_hy = Interp_X.dot(bxm);  bym_hy = Interp_X.dot(bym)

        self.bz_h = 1.0 / (c * dt)
        self.ap = 2.0 * gamma / dt + 1.0
        self.am = 2.0 * gamma / dt - 1.0

        def to_diag_vec(vec):
            return sp.diags_array(vec, format='csr')
            
        def to_diag_scal(val, length):
            val_flat = np.array(val).flatten()
            return sp.diags_array(np.full(length, val_flat), format='csr')

        # Eq 2.128 z-comp: κy·∂ẽz/∂τ + Z0σy·ẽz = κz·∂ėz/∂τ  (κz=1 in 2D TM)
        # → L56 = -1/(c·dt)  [NOT -1/(v·dt)]
        L11 = to_diag_scal(1.0/(c*dt), self.len_hx);    L12 = to_diag_vec(-bxp_hx)
        L22 = to_diag_vec(byp_hx);                      L25 = 0.5 * DY_ez_to_hx
        L33 = to_diag_vec(bxp_hy);                      L34 = to_diag_vec(-byp_hy)
        L44 = to_diag_scal(1.0/(c*dt), self.len_hy);    L45 = -0.5 * DX_ez_to_hy
        L55 = to_diag_vec(byp);                         L56 = -I_ez / (c * dt)
        L66 = to_diag_vec(bxp);                         L67 = -I_ez / (c * dt)
        L71 = 0.5 * DY_hx_to_ez;      L73 = -0.5 * DX_hy_to_ez;     L77 = I_ez / (c * dt);      L78 = I_ez / 2.0
        L87 = to_diag_vec(-self.sigma.flatten());       L88 = to_diag_scal(2.0*gamma/dt + 1.0, self.len_ez)

        # Assemble LHS Block Matrix
        self.LHS = sp.bmat([
            [L11,  L12,  None, None, None, None, None, None], # hx
            [None, L22,  None, None, L25,  None, None, None], # hx_dot
            [None, None, L33,  L34,  None, None, None, None], # hy
            [None, None, None, L44,  L45,  None, None, None], # hy_dot
            [None, None, None, None, L55,  L56,  None, None], # ez
            [None, None, None, None, None, L66,  L67,  None], # ez_dot
            [L71,  None, L73,  None, None, None, L77,  L78 ], # ez_ddot
            [None, None, None, None, None, None, L87,  L88 ]  # jz
        ], format='csc')

        # RHS: β_m coefficients, flipped derivative signs
        R11 = to_diag_scal(1.0/(c*dt), self.len_hx);    R12 = to_diag_vec(-bxm_hx)
        R22 = to_diag_vec(bym_hx);                      R25 = -0.5 * DY_ez_to_hx
        R33 = to_diag_vec(bxm_hy);                      R34 = to_diag_vec(-bym_hy)
        R44 = to_diag_scal(1.0/(c*dt), self.len_hy);    R45 = 0.5 * DX_ez_to_hy
        R55 = to_diag_vec(bym);                         R56 = -I_ez / (c * dt)
        R66 = to_diag_vec(bxm);                         R67 = -I_ez / (c * dt)
        R71 = -0.5 * DY_hx_to_ez;     R73 = 0.5 * DX_hy_to_ez;      R77 = I_ez / (c * dt);      R78 = -I_ez / 2.0
        R87 = to_diag_vec(self.sigma.flatten());        R88 = to_diag_scal(2.0*gamma/dt - 1.0, self.len_ez)

        # Assemble RHS Block Matrix
        self.RHS = sp.bmat([
            [R11,  R12,  None, None, None, None, None, None], # hx
            [None, R22,  None, None, R25,  None, None, None], # hx_dot
            [None, None, R33,  R34,  None, None, None, None], # hy
            [None, None, None, R44,  R45,  None, None, None], # hy_dot
            [None, None, None, None, R55,  R56,  None, None], # ez
            [None, None, None, None, None, R66,  R67,  None], # ez_dot
            [R71,  None, R73,  None, None, None, R77,  R78 ], # ez_ddot
            [None, None, None, None, None, None, R87,  R88 ]  # jz
        ], format='csc')

        if self.solver == "default":
            print("Pre-factoring the 8x8 system...")
            self.solve_func = spla.factorized(self.LHS.tocsc())

        elif self.solver == "Schur":
            print("Pre-factoring the 8x8 system using Schur complement...")
            
            # Extract blocks
            M11 = self.LHS[:2*self.len_hx + 2*self.len_hy, :2*self.len_hx + 2*self.len_hy].tocsc()
            M12 = self.LHS[:2*self.len_hx + 2*self.len_hy, 2*self.len_hx + 2*self.len_hy:].tocsc()
            M21 = self.LHS[2*self.len_hx + 2*self.len_hy:, :2*self.len_hx + 2*self.len_hy].tocsc()
            M22 = self.LHS[2*self.len_hx + 2*self.len_hy:, 2*self.len_hx + 2*self.len_hy:].tocsc()

            L11_inv = sp.diags_array(1.0/L11.diagonal(), format='csc')
            L22_inv = sp.diags_array(1.0/L22.diagonal(), format='csc')
            L33_inv = sp.diags_array(1.0/L33.diagonal(), format='csc')
            L44_inv = sp.diags_array(1.0/L44.diagonal(), format='csc')

            # Invert M11
            M11_inv = sp.bmat([
                [L11_inv,  -L11_inv @ L12 @ L22_inv,  None,       None],
                [None,      L22_inv,                  None,       None],
                [None,      None,                     L33_inv,  -L33_inv @ L34 @ L44_inv],
                [None,      None,                     None,       L44_inv]
            ], format='csc')

            # Precompute S = M22 - M21 * M11_inv * M12
            S = M22 - M21 @ M11_inv @ M12
            S_fact = spla.factorized(S.tocsc())

            # Define the solver function
            def solve_schur(b):
                # Split b into b1 and b2
                b1 = b[:2*self.len_hx + 2*self.len_hy]
                b2 = b[2*self.len_hx + 2*self.len_hy:]

                # Solve for u2 = S_inv* (b2 - M21 * M11_inv * b1)
                u2 = S_fact(b2 - M21 @ M11_inv @ b1)

                # Solve for u1 = M11_inv * (b1 - M12 * u2)
                u1 = M11_inv @ (b1 - M12 @ u2)

                # Combine to get full solution
                return np.concatenate([u1, u2])

            self.solve_func = solve_schur
            

    def run_simulation(self, src_pos, obs_pos, frame_skip=2):
        u = np.zeros(self.total_len)
        ez_history = []
        movie_frames = []
        
        # Global index for Ez component at src_pos
        offset = 2 * self.len_hx + 2 * self.len_hy
        x0, y0 = src_pos
        x_obs, y_obs = obs_pos

        src_idx = offset + x0 * self.ny_n + y0

        # shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        # neighbors = [offset + ((x0 + dx) % self.nx_n) * self.ny_n + ((y0 + dy) % self.ny_n) for dx, dy in shifts]
        # weights = np.array([0.5, 0.125, 0.125, 0.125, 0.125])

        fig, ax = plt.subplots()
        
        # Static markers for source and observation points
        ax.scatter(x0 * self.dx, y0 * self.dy, color='red', s=40, zorder=2, label='Source')
        ax.scatter(x_obs * self.dx, y_obs * self.dy, color='green', s=40, zorder=2, label='Observer')
        ax.legend(loc='upper right', fontsize=8)

        for i in tqdm(range(self.Nt), desc=f"Simulating ({self.bc})"):
            t = i * self.dt
            b = self.RHS.dot(u)

            #Smooth source over a couple of grid points to prevent checkerboarding
            src_val = self.my_src(t)
            b[src_idx] -= src_val
            # for idx, w in zip(neighbors, weights):
            #     b[idx] += src_val * w

            u = self.solve_func(b)
            ez_2d = u[self.idx_ez].reshape((self.nx_n, self.ny_n))
            ez_history.append(ez_2d[x_obs, y_obs])
            
            if i % frame_skip == 0:
                txt = ax.text(0.5, 1.05, f'Step: {i}/{self.Nt} | BC: {self.bc}', ha="center", transform=ax.transAxes)
                img = ax.imshow((ez_2d * self.Z_local).T, cmap='RdBu', origin='lower', animated=True,
                                extent=[0, self.Nx*self.dx, 0, self.Ny*self.dy], vmin=-0.1, vmax=0.1, zorder=1)
                movie_frames.append([txt, img])
        
        ani = ArtistAnimation(fig, movie_frames, interval=100, blit=True)
        return ani, np.array(ez_history)

    @staticmethod
    def plot_1d_intensity(dt, ez_rec):
        plt.figure()
        plt.plot(np.arange(len(ez_rec)) * dt, ez_rec)
        plt.title("Ez Field at Source Location")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def verify_with_hankel(self, ez_obs_data, src_pos, obs_pos, hankel_f_min=None, hankel_f_max=None):
        print("\n--- Plotting Hankel verification ---")
        
        dx_m = (obs_pos[0] - src_pos[0]) * self.dx
        dy_m = (obs_pos[1] - src_pos[1]) * self.dy
        r = np.sqrt(dx_m**2 + dy_m**2)
        
        if r == 0:
            raise ValueError("r must be greater than zero")

        # Setup time and frequency 
        t = np.arange(self.Nt) * self.dt
        n_pad = 2**int(np.ceil(np.log2(self.Nt * 8)))  # High resolution zero padding
        freqs = fftfreq(n_pad, self.dt)
        f_min = hankel_f_min if hankel_f_min is not None else self.f_c * 0.2
        f_max = hankel_f_max if hankel_f_max is not None else self.f_c * 1.8
        band_idx = np.where((freqs > f_min) & (freqs < f_max))[0]
        f_valid = freqs[band_idx]
        omega = 2 * np.pi * f_valid
        k0 = omega / self.c
        
        # FFT of simulation data and the source function (with zero padding)
        src_time = self.my_src(t)
        J_src_f = fft(src_time, n=n_pad) * self.dt
        J_src_valid = J_src_f[band_idx]
        
        Ez_sim_f = fft(ez_obs_data, n=n_pad) * self.dt
        Ez_sim_valid = Ez_sim_f[band_idx]
        
        H_sim = Ez_sim_valid / J_src_valid
        H_sim_corrected = H_sim * np.exp(1j * omega * self.dt)

        # Analytical Solution
        H_analytical = -(omega * self.mu / 4) * sp_special.hankel2(0, k0 * r)
        
        # Save Hankel data for error analysis
        self._hankel_data = {
            "f_valid": f_valid, 
            "obs_points": {
                "Observation Point": {
                    "H_sim": H_sim_corrected,
                    "H_analytical": H_analytical
                }
            }
        }

        # Plotting — Magnitude and Phase only
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        idx_fc = np.argmin(np.abs(f_valid - self.f_c))
        
        ax = axes[0]
        ax.plot(f_valid, np.abs(J_src_valid) / np.abs(J_src_valid[idx_fc]), label='Source', lw=2, color='lightgray', alpha=1.0)
        
        norm_sim = np.abs(H_sim_corrected[idx_fc])
        norm_anal = np.abs(H_analytical[idx_fc])
        
        ax.plot(f_valid, np.abs(H_sim_corrected) / norm_sim, label='Simulation', lw=2)
        ax.plot(f_valid, np.abs(H_analytical) / norm_anal, '--', label='Analytical', lw=2)
        
        f_break = 2 * self.f_c
        max_val = np.max(np.abs(H_sim_corrected[f_valid < f_break]) / norm_sim) if np.any(f_valid < f_break) else 1.0
        
        axes[1].plot(f_valid, np.unwrap(np.angle(H_sim_corrected)), label='Simulation', lw=2)
        axes[1].plot(f_valid, np.unwrap(np.angle(H_analytical)), '--', label='Analytical', lw=2)

        ax.set_ylim(0, 1.3 * max_val)
        ax.axvline(self.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({self.f_c:.2e} Hz)')
        ax.axvline(f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({f_break:.2e} Hz)')
        ax.axvline(f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({f_min:.2e} Hz)')
        ax.set_xscale('log')
        ax.set_title('Magnitude Response')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        axes[1].set_xscale('log')
        axes[1].set_title('Phase Response')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (rad)')
        axes[1].legend()
        axes[1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_error_analysis(self):
        if not hasattr(self, "_hankel_data"):
            print("Run verify_with_hankel first!")
            return
            
        hd = self._hankel_data
        f_valid = hd["f_valid"]
        
        # Print key frequency limits
        f_src_max = self.f_c * (1 + 1)  # Source bandwidth edge
        f_nyquist = self.c / (2 * min(self.dx, self.dy))
        f_cutoff = (1 / (np.pi * self.dt)) * np.arcsin(self.c * self.dt / min(self.dx, self.dy))
        print(f"  Source bandwidth:    ~[0, {f_src_max:.3e}] Hz  (2 * f_c)")
        print(f"  Spatial Nyquist:     {f_nyquist:.3e} Hz  ({f_nyquist/self.f_c:.1f} * f_c)")
        print(f"  Yee num. cutoff:     {f_cutoff:.3e} Hz  ({f_cutoff/self.f_c:.1f} * f_c)")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        idx_fc = np.argmin(np.abs(f_valid - self.f_c))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (name, data) in enumerate(hd["obs_points"].items()):
            H_sim, H_anal = data["H_sim"], data["H_analytical"]
            
            # Normalize both at f_c to remove constant amplitude offset from source injection
            sim_mag = np.abs(H_sim) / np.abs(H_sim[idx_fc])
            anal_mag = np.abs(H_anal) / np.abs(H_anal[idx_fc])
            rel_error = np.abs(sim_mag - anal_mag) / anal_mag
            
            # Phase Error Dispersion
            sim_phase = np.unwrap(np.angle(H_sim))
            anal_phase = np.unwrap(np.angle(H_anal))
            # Compute difference and pin it to 0 at the central frequency
            phase_diff = sim_phase - anal_phase
            phase_diff -= phase_diff[idx_fc]
            phase_error_deg = np.rad2deg(phase_diff)
            
            axes[0].plot(f_valid, rel_error * 100, label=name, lw=1.5, color=colors[i])
            axes[1].plot(f_valid, phase_error_deg, label=name, lw=1.5, color=colors[i])
            
        # Magnitude Error Formatting
        ax = axes[0]
        ax.axvline(self.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({self.f_c:.2e} Hz)')
        ax.axvline(2 * self.f_c, color='red', ls=':', alpha=0.5, label=f'f_break ({2 * self.f_c:.2e} Hz)')
        ax.axvline(self.f_c * 0.2, color='green', ls=':', alpha=0.5, label=f'f_min ({self.f_c * 0.2:.2e} Hz)')
        ax.set_xscale('log')
        ax.set_title('Relative Magnitude Error')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Relative Error (%)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        # Phase Error Formatting
        ax = axes[1]
        ax.axvline(self.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({self.f_c:.2e} Hz)')
        ax.axvline(2 * self.f_c, color='red', ls=':', alpha=0.5, label=f'f_break ({2 * self.f_c:.2e} Hz)')
        ax.axvline(self.f_c * 0.2, color='green', ls=':', alpha=0.5, label=f'f_min ({self.f_c * 0.2:.2e} Hz)')
        ax.set_xscale('log')
        ax.set_title('Phase Error Dispersion')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase Error (Degrees)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @classmethod
    def run_full_analysis(cls, params):
        src_pos = params.pop('Source_loc', (50, 100))
        obs_pos = params.pop('Obs_loc', (70, 100))
        frame_skip = params.pop('frame_skip', 2)
        hankel_f_min = params.pop('hankel_f_min', None)
        hankel_f_max = params.pop('hankel_f_max', None)
        
        t0 = time.time()
        solver = cls(**params)
        
        # Run simulation
        ani, ez_data = solver.run_simulation(src_pos, obs_pos, frame_skip=frame_skip)
        t1 = time.time()
        exec_time = t1 - t0
        print(f"\n>>> [{solver.solver} solver] Execution time (Setup + Sim): {exec_time:.4f} seconds <<<\n")
        
        # Plot 1D intensity at observer
        solver.plot_1d_intensity(solver.dt, ez_data)
        # Verify against analytical Hankel
        solver.verify_with_hankel(ez_data, src_pos, obs_pos, hankel_f_min, hankel_f_max)
        solver.plot_error_analysis()
        
        return ez_data, exec_time


sim_params = {
    'Nx': 600, 
    'Ny': 600, 
    'Nt': 300, 
    'lambda0': 1, 
    'CFL': 2,
    'Source_loc' : (50,100),
    'Obs_loc' : (80,100),
    'bc': 'PBC',
    'solver': 'Schur',
    'finesse': 15,
    'frame_skip': 3,
    'hankel_f_min': 0.0,
    'hankel_f_max': 3 * 299792458
}

results, time_schur = FCI_TM_Solver.run_full_analysis(sim_params)

# sim_params_2 = {
#     'Nx': 200, 
#     'Ny': 200, 
#     'Nt': 300, 
#     'lambda0': 1, 
#     'CFL': 2,
#     'Source_loc' : (50,100),
#     'Obs_loc' : (80,100),
#     'bc': 'PBC',
#     'solver': 'default'
# }

# results_2, time_default = FCI_TM_Solver.run_full_analysis(sim_params_2)

# print("\n--- Final Performance Comparison ---")
print(f"Schur complement solver time: {time_schur:.4f} seconds")
# print(f"Default solver time:          {time_default:.4f} seconds")