import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import scipy.special as sp_special
from scipy.fft import fft, fftfreq

class FCI_TM_Solver:
    def __init__(self, Nx, Ny, Nt, lambda0, CFL, bc='PEC'):
        # Parameters
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.lambda0, self.CFL = lambda0, CFL
        self.dx, self.dy = lambda0 / 30, lambda0 / 30
        self.bc = bc.upper()

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
        self.a      = 4 # Amount of sigmas between fc and 0 in frequency domain
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

        p, m = 20, 4
        k_max = 1
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
        if self.bc == 'PBC': # i swear there is something off here cus the wave is "soft" when using these:
            ones = np.ones(n)
            Ix = sp.eye(n, format='csr')
            
            Dx = sp.diags([-ones, ones, ones], [0, 1, 1-n], shape=(n, n), format='csr') / d
            Dtx = sp.diags([ones, -ones, -ones], [0, -1, n-1], shape=(n, n), format='csr') / d
            
            A1 = sp.diags([ones, ones, ones], [0, -1, n-1], shape=(n, n), format='csr')
            A2 = sp.diags([ones, ones, ones], [0, 1, 1-n], shape=(n, n), format='csr')
            
            return Ix, Dx, Dtx, A1, A2
        
        else:
            Ix = sp.eye(n + 1, format='csr')
            Dx = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
            Dtx = (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)).tolil()
            Dtx[n, n-1] = -2

            A1 = sp.diags([1, 1], [0, -1], shape=(n, n-1), format='csr')
            A2 = sp.diags([1, 1], [0, 1], shape=(n, n+1), format='csr')

            return Ix, Dx.tocsr(), Dtx.tocsr(), A1.tocsr(), A2.tocsr()

    def _build_system(self):
        v, c, Z, gamma, dt = self.v_local, self.c, self.Z_local, self.gamma, self.dt
        sx, sy, kx, ky = self._get_pml_profiles()

        Ix, Dx, Dtx, Ax1, Ax2 = self._get_operators(self.Nx, self.dx)
        Iy, Dy, Dty, Ay1, Ay2 = self._get_operators(self.Ny, self.dy)
        
        DY_ez_to_hx = sp.kron(Ix, Dy)  # Size: len_hx x len_ez
        DX_ez_to_hy = sp.kron(Dx, Iy)  # Size: len_hy x len_ez
        DY_hx_to_ez = sp.kron(Ix, Dty) # Size: len_ez x len_hx
        DX_hy_to_ez = sp.kron(Dtx, Iy) # Size: len_ez x len_hy

        I_hx = sp.eye(self.len_hx, format='csr')
        I_hy = sp.eye(self.len_hy, format='csr')
        I_ez = sp.eye(self.len_ez, format='csr')

        Interp_Y = sp.kron(Ix, Ay2) * 0.5  # Maps Ez grid -> Hx grid
        Interp_X = sp.kron(Ax2, Iy) * 0.5  # Maps Ez grid -> Hy grid

        N = self.nx_n * self.ny_n
        I = sp.eye(N, format='csr')

        # Update coefficients
        bxp = (kx / (v * dt) + Z * sx / 2.0).flatten()
        byp = (ky / (v * dt) + Z * sy / 2.0).flatten()
        bxm = (kx / (v * dt) - Z * sx / 2.0).flatten()
        bym = (ky / (v * dt) - Z * sy / 2.0).flatten()

        bxp_hx = Interp_Y.dot(bxp)
        byp_hx = Interp_Y.dot(byp)
        bxm_hx = Interp_Y.dot(bxm)
        bym_hx = Interp_Y.dot(bym)

        bxp_hy = Interp_X.dot(bxp)
        byp_hy = Interp_X.dot(byp)
        bxm_hy = Interp_X.dot(bxm)
        bym_hy = Interp_X.dot(bym)

        self.bz_h = 1.0 / (c * dt)
        self.bz_e = 1.0 / (v * dt)
        self.ap = 2.0 * gamma / dt + 1.0
        self.am = 2.0 * gamma / dt - 1.0

        def to_diag_vec(vec):
            return sp.diags(vec, format='csr')
            
        def to_diag_scal(val, length):
            val_flat = np.array(val).flatten()
            return sp.diags(np.full(length, val_flat), format='csr')
            
        L11 = to_diag_scal(1.0/(c*dt), self.len_hx);    L12 = to_diag_vec(-bxp_hx)
        L22 = to_diag_vec(byp_hx);                      L25 = DY_ez_to_hx
        L33 = to_diag_vec(bxp_hy);                      L34 = to_diag_vec(-byp_hy)
        L44 = to_diag_scal(1.0/(c*dt), self.len_hy);    L45 = -DX_ez_to_hy
        L55 = to_diag_vec(byp);                         L56 = to_diag_scal(-1.0/(v*dt), self.len_ez)
        L66 = to_diag_vec(bxp);                         L67 = -I_ez / (c * dt)
        L71 = DY_hx_to_ez; L73 = -DX_hy_to_ez; L77 = I_ez / (c * dt); L78 = I_ez / 2.0
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

        # RHS uses beta_m and flips signs on derivative terms
        R11 = to_diag_scal(1.0/(c*dt), self.len_hx);    R12 = to_diag_vec(-bxm_hx)
        R22 = to_diag_vec(bym_hx);                      R25 = -DY_ez_to_hx
        R33 = to_diag_vec(bxm_hy);                      R34 = to_diag_vec(-bym_hy)
        R44 = to_diag_scal(1.0/(c*dt), self.len_hy);    R45 = DX_ez_to_hy
        R55 = to_diag_vec(bym);                         R56 = to_diag_scal(-1.0/(v*dt), self.len_ez)
        R66 = to_diag_vec(bxm);                         R67 = -I_ez / (c * dt)
        R71 = -DY_hx_to_ez;     R73 = DX_hy_to_ez;      R77 = I_ez / (c * dt);      R78 = -I_ez / 2.0
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

        print("Pre-factoring the 8x8 system...")
        self.solve_func = spla.factorized(self.LHS.tocsc())

    def run_simulation(self, src_pos, obs_pos):
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

        for i in tqdm(range(self.Nt), desc=f"Simulating ({self.bc})"):
            t = i * self.dt
            b = self.RHS.dot(u)

            #Smooth source over a couple of grid points to prevent checkerboarding
            src_val = self.my_src(t)
            b[src_idx] += src_val
            # for idx, w in zip(neighbors, weights):
            #     b[idx] += src_val * w

            u = self.solve_func(b)
            ez_2d = u[self.idx_ez].reshape((self.nx_n, self.ny_n))
            ez_history.append(ez_2d[x_obs, y_obs])
            
            if i % 2 == 0:
                txt = ax.text(0.5, 1.05, f'Step: {i}/{self.Nt} | BC: {self.bc}', ha="center", transform=ax.transAxes)
                img = ax.imshow(ez_2d.T * self.Z_local, cmap='RdBu', origin='lower', animated=True,
                                extent=[0, self.Nx*self.dx, 0, self.Ny*self.dy], vmin=-0.1, vmax=0.1)
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

    def verify_with_hankel(self, ez_obs_data, src_pos, obs_pos):
        # 1. Calculate physical distance (r) between source and observer
        dx_m = (obs_pos[0] - src_pos[0]) * self.dx
        dy_m = (obs_pos[1] - src_pos[1]) * self.dy
        r = np.sqrt(dx_m**2 + dy_m**2)
        
        if r == 0:
            raise ValueError("Observation position must differ from source position (r cannot be 0).")

        # 2. Setup Time and Frequency arrays
        t = np.arange(self.Nt) * self.dt
        freqs = fftfreq(self.Nt, self.dt)
        
        # 3. FFT of simulation data and the source function
        src_time = self.my_src(t)
        Ez_sim_f = fft(ez_obs_data) * self.dt
        J_src_f = fft(src_time) * self.dt
        
        # Avoid division by zero by isolating the active frequency band
        band_idx = np.where((freqs > self.f_c * 0.2) & (freqs < self.f_c * 1.8))[0]
        f_valid = freqs[band_idx]
        
        Ez_sim_valid = Ez_sim_f[band_idx]
        J_src_valid = J_src_f[band_idx]
        
        # Normalize simulated field by the source spectrum (Cancels the t0 delay)
        H_sim = Ez_sim_valid / J_src_valid
        
        # Analytical Solution
        omega = 2 * np.pi * f_valid
        k0 = omega / self.c
        H_analytical = -(omega * self.mu / 4) * sp_special.hankel2(0, k0 * r)
        
        # --- THE FIXES ---
        # A) Correct for the 1-step FDTD recording delay
        H_sim_corrected = H_sim * np.exp(1j * omega * self.dt)
        
        # B) Unwrap the radians first, then properly convert to degrees
        sim_phase_deg = np.rad2deg(np.unwrap(np.angle(H_sim_corrected)))
        ana_phase_deg = np.rad2deg(np.unwrap(np.angle(H_analytical)))
        # -----------------
        
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
        plt.plot(f_valid, sim_phase_deg, label='Simulation Phase', lw=2)
        plt.plot(f_valid, ana_phase_deg, '--', label='Analytical Phase', lw=2)
        plt.title('Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (°)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    @classmethod
    def run_full_analysis(cls, params):
        src_pos = params.pop('Source_loc', (50, 100))
        obs_pos = params.pop('Obs_loc', (70, 100)) # e.g. 20 cells to the right
        solver = cls(**params)
        
        # Run simulation
        ani, ez_data = solver.run_simulation(src_pos, obs_pos)
        
        # Plot 1D intensity at observer
        solver.plot_1d_intensity(solver.dt, ez_data)
        
        # Verify against analytical Hankel
        solver.verify_with_hankel(ez_data, src_pos, obs_pos)
        
        return ez_data


sim_params = {
    'Nx': 200, 
    'Ny': 200, 
    'Nt': 300, 
    'lambda0': 1, 
    'CFL': 2,
    'Source_loc' : (50,100),
    'Obs_loc' : (80,100),
    'bc': 'PEC'
}

results = FCI_TM_Solver.run_full_analysis(sim_params)