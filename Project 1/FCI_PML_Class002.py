import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

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
        self.Z0      = np.sqrt(self.mu / self.eps)

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
        self.total_len = self.len_hx + self.len_hy + self.len_ez
        self.idx_ez = slice(self.len_hx + self.len_hy, self.total_len)
        
        self._build_system()

    def my_src(self, t):
        # Gaussian Pulse: A * cos(2*pi*fc*(t-t0)) * exp(-0.5 * ((t-t0)/sig_t)**2)
        return self.A * np.cos(2 * np.pi * self.f_c * (t - self.t0)) * \
               np.exp(-0.5 * ((t - self.t0) / self.sig_t)**2)

    def _get_pml_profiles(self):
        if self.bc == 'PBC': #to properly see PBC
            return np.zeros(self.len_ez), np.zeros(self.len_ez)

        self.kx = np.ones(self.nx_n)
        self.ky = np.ones(self.ny_n)
        self.sx = np.zeros(self.nx_n)
        self.sy = np.zeros(self.ny_n)

        p, m = 25, 4
        k_max = 4
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
            # PBC matrices:
            ones = np.ones(n)
            Ix = sp.eye(n, format='csr')
            Ahx = sp.diags([ones, ones, ones], [0, 1, 1-n], shape=(n, n)).tocsr()
            Ax  = sp.diags([ones, ones, ones], [0, 1, 1-n], shape=(n, n)).tocsr()
            Atx = sp.diags([ones, ones, ones], [0, -1, n-1], shape=(n, n)).tocsr()
            Dx  = sp.diags([-ones, ones, ones], [0, 1, 1-n], shape=(n, n)).tocsr() / d
            Dtx = sp.diags([ones, -ones, -ones], [0, -1, n-1], shape=(n, n)).tocsr() / d
            return Ix, Ahx, Ax, Atx, Dx, Dtx
        
        else:
            inv_d = sp.diags(1.0 / d)

            D1_core = sp.diags([1, -1], [0, -1], shape=(n, n-1))
            D2_core = sp.diags([-1, 1], [0, 1], shape=(n, n+1))
            D1 = inv_d @ D1_core
            D2 = inv_d @ D2_core

            return D1, D2

    def _build_system(self):
        v, c, Z, gamma, dt = self.v_local, self.c, self.Z_local, self.gamma, self.dt
        sx, sy, kx, ky = self._get_pml_profiles()
        Dx1, Dx2 = self._get_operators(self.Nx, self.dx)
        Dy1, Dy2 = self._get_operators(self.Ny, self.dy)
        I = np.eye(self.nx_n,self.ny_n)
        
        # 2D Operators: Match the field dimensions!
        DX1_2D = sp.kron(Dx1, sp.eye(self.ny_n)) 
        DY1_2D = sp.kron(sp.eye(self.nx_n), Dy1)
        

        # Update coefficients
        self.bxp = kx / (v * dt) + Z * sx / 2.0
        self.byp = ky / (v * dt) + Z * sy / 2.0
        self.bxm = kx / (v * dt) - Z * sx / 2.0
        self.bym = ky / (v * dt) - Z * sy / 2.0
        self.bz_h = 1.0 / (c * dt)
        self.bz_e = 1.0 / (v * dt)
        self.ap = 2.0 * gamma / dt + 1.0
        self.am = 2.0 * gamma / dt - 1.0
        I_ez = sp.eye(self.nx_n * self.ny_n)
        BZ_H = sp.diags([self.bz_h] * (self.nx_n * self.ny_n))

        L11, L12 = self.bz_h, -self.bxp
        L22 = self.byp
        L33, L34 = self.bxp, -self.byp
        L44 = self.bz_h
        L55, L56 = self.byp, -self.bz_e
        L66, L67 = self.bxp, -I / (c  * dt)
        L77, L78 = I / (c  * dt), I / 2
        L87, L88 = - self.sigma, self.ap

        # Assemble LHS Block Matrix
        self.LHS = sp.bmat([
            [L11,  L12,  None, None, None, None, None, None], # hx
            [None, L22,  None, None, None, None, None, None], # hx_dot
            [None, None, L33,  L34,  None, None, None, None], # hy
            [None, None, None, L44,  None, None, None, None], # hy_dot
            [None, None, None, None, L55,  L56,  None, None], # ez
            [None, None, None, None, None, L66,  L67,  None], # ez_dot
            [None, None, None, None, None, None, L77,  L78 ], # ez_ddot
            [None, None, None, None, None, None, L87,  L88 ]  # jz
        ], format='csc')

        # RHS uses beta_m and flips signs on derivative terms
        L11, L12 = self.bz_h, -self.bxm
        L22, L25 = self.bym, -Dy1
        L33, L34 = self.bxm, -self.bym
        L44, L45 = self.bz_h, Dx1
        L55, L56 = self.bym, -self.bz_e
        L66, L67 = self.bxm, -I / (c  * dt)
        L71, L73, L77, L78 = -Dy2, Dx2, I / (c  * dt), -I / 2
        L87, L88 = self.sigma, self.am
        
        # Assemble RHS Block Matrix
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

        print("Pre-factoring the 8x8 system...")
        self.solver = spla.factorized(self.LHS)

    def run_simulation(self, src_pos):
        u = np.zeros(self.total_len)
        ez_history = []
        movie_frames = []
        
        # Global index for Ez component at src_pos
        offset = self.len_hx + self.len_hy
        x0, y0 = src_pos
        
        shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors = [offset + ((x0 + dx) % self.nx_n) * self.ny_n + ((y0 + dy) % self.ny_n) for dx, dy in shifts]
        weights = np.array([0.5, 0.125, 0.125, 0.125, 0.125])
        fig, ax = plt.subplots()

        for i in tqdm(range(self.Nt), desc=f"Simulating ({self.bc})"):
            t = i * self.dt
            b = self.RHS.dot(u)

            #Smooth source over a couple of grid points to prevent checkerboarding
            src_val = self.my_src(t)
            for idx, w in zip(neighbors, weights):
                b[idx] += src_val * w

            u = self.solve_func(b)
            ez_2d = u[self.idx_ez].reshape((self.nx_n, self.ny_n))
            ez_history.append(ez_2d[x0, y0])
            
            if i % 2 == 0:
                txt = ax.text(0.5, 1.05, f'Step: {i}/{self.Nt} | BC: {self.bc}', ha="center", transform=ax.transAxes)
                img = ax.imshow(ez_2d.T, cmap='RdBu', origin='lower', animated=True,
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

    @classmethod
    def run_full_analysis(cls, params):
        src_pos = params.pop('Source_loc', (50, 100))
        solver = cls(**params)
        
        # Run simulation
        ani, ez_data = solver.run_simulation(src_pos)
        # Plot 1D intensity
        solver.plot_1d_intensity(solver.dt, ez_data)
        plt.show()
        return ez_data


sim_params = {
    'Nx': 200, 
    'Ny': 200, 
    'Nt': 200, 
    'lambda0': 1.0, 
    'CFL': 2,
    'Source_loc' : (50,100),
    'bc': 'PBC'
}

results = FCI_TM_Solver.run_full_analysis(sim_params)