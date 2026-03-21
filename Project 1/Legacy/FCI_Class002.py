import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class FCI_TM_Solver:
    def __init__(self, Nx, Ny, Nt, lambda0, CFL, bc='PEC'):
        # Constants & Parameters
        self.eps, self.mu, self.sigma, self.c = 8.854e-12, 1.256e-6, 0.0, 3e8 # Set sigma=0 for no dampening
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.lambda0, self.CFL = lambda0, CFL
        self.dx, self.dy = lambda0 / 30, lambda0 / 30
        self.dt = CFL / (self.c * np.sqrt(1/self.dx**2 + 1/self.dy**2))
        self.bc = bc.upper()

        # PBC drops the redundant N+1 boundary node to form a perfect ring
        self.nx_n = self.Nx if self.bc == 'PBC' else self.Nx + 1
        self.ny_n = self.Ny if self.bc == 'PBC' else self.Ny + 1
        
        # Source Parameters
        self.f_c    = self.c / self.lambda0
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t

        # Component lengths
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

        sx = np.zeros((self.nx_n, self.ny_n))
        sy = np.zeros((self.nx_n, self.ny_n))
        p, m = 25, 4
        eta_max = (m + 1) / (150 * np.pi * min([self.dx, self.dy]))

        for i in range(p):
            d = (p - i) / p
            val = eta_max * d ** m
            sx[i, :] = val # Left
            sx[-1-i, :] = val # Right
            sy[:, i] = val # Bottom
            sy[:, -1-i] = val # Top
        return sx.flatten(), sy.flatten()
        
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
            # PEC matrices:
            Ix = sp.eye(n + 1, format='csr')
            Ahx = (sp.eye(n, n + 1, k=0) + sp.eye(n, n + 1, k=1)).tocsr()
            Ax = (sp.eye(n, n) + sp.eye(n, n, k=1)).tocsr()
            Atx = (sp.eye(n + 1, n + 1) + sp.eye(n + 1, n + 1, k=-1)).tolil()
            Dx = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
            Dtx = (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)).tolil()
            Dtx[n, n-1] = -2
            Atx[n, n] = 2
            return Ix, Ahx, Ax, Atx.tocsr(), Dx.tocsr(), Dtx.tocsr() / d

    def _build_system(self):
        s_x, s_y = self._get_pml_profiles()
        Ix, Ahx, Ax, Atx, Dx, Dtx = self._get_operators(self.Nx, self.dx)
        Iy, Ahy, Ay, Aty, Dy, Dty = self._get_operators(self.Ny, self.dy)


        Ezz_p = (self.eps / self.dt + self.sigma / 2) * sp.eye(self.len_ez) + sp.diags((s_x + s_y) / 2)
        Ezz_m = (self.eps / self.dt - self.sigma / 2) * sp.eye(self.len_ez) - sp.diags((s_x + s_y) / 2)

        mux_diag = (self.mu / self.dt) + s_x / 2
        muy_diag = (self.mu / self.dt) + s_y / 2
        Mxx = sp.diags(mux_diag.repeat(self.Ny) if len(mux_diag) < self.len_hx else mux_diag[:self.len_hx])
        Mxx = (self.mu / self.dt) * sp.eye(self.len_hx) + sp.diags(np.zeros(self.len_hx)) # Placeholder for complex H-stretch
        Myy = (self.mu / self.dt) * sp.eye(self.len_hy)

        # LHS
        L11, L13 = sp.kron(Ix, Ay) @ Mxx, sp.kron(Ix, Dy)
        L22, L23 = sp.kron(Ax, Iy) @ Myy, -sp.kron(Dx, Iy)
        L31, L32, L33 = sp.kron(Atx, Dty), -sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_p
        LHS = sp.bmat([[L11, None, L13], 
                       [None, L22, L23], 
                       [L31, L32, L33]], 
                       format='csr')
        
        # RHS
        R11, R13 = sp.kron(Ix, Ay) @ Mxx, -sp.kron(Ix, Dy)
        R22, R23 = sp.kron(Ax, Iy) @ Myy, sp.kron(Dx, Iy)
        R31, R32, R33 = -sp.kron(Atx, Dty), sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_m
        self.RHS = sp.bmat([[R11, None, R13],
                            [None, R22, R23], 
                            [R31, R32, R33]], 
                            format='csr')

        self.solve_func = spla.factorized(LHS.tocsc())

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
