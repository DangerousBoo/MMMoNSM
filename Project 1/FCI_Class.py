import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class FCI_TM_Solver:
    def __init__(self, Nx, Ny, Nt, lambda0, CFL):
        # Constants & Parameters
        self.eps, self.mu, self.sigma, c = 8.854e-12, 1.256e-6, 0.0, 3e8 # Set sigma=0 for no dampening
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.lambda0, self.CFL = lambda0, CFL
        self.dx, self.dy = lambda0 / 30, lambda0 / 30
        self.dt = CFL / (c * np.sqrt(1/self.dx**2 + 1/self.dy**2))

        # Source Parameters
        self.f_c    = self.c / self.lam_c
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t
        
        # Component lengths
        self.len_hx = (Nx + 1) * Ny
        self.len_hy = Nx * (Ny + 1)
        self.len_ez = (Nx + 1) * (Ny + 1)
        self.total_len = self.len_hx + self.len_hy + self.len_ez
        self.idx_ez = slice(self.len_hx + self.len_hy, self.total_len)
        
        self._build_system()

    def my_src(self):
        t = np.linspace(0,self.Nt * self.dt)
        f_c = 3e8 / 1.0
        t0 = 4 * (3 / (2 * np.pi * f_c))
        # Gaussian Pulse
        return 5 * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/(3/(2*np.pi*f_c)))**2)

    def _get_pml_profiles(self):
        sx = np.zeros((self.Nx + 1, self.Ny + 1))
        sy = np.zeros((self.Nx + 1, self.Ny + 1))
        p, m = 20, 4
        eta_max = (m + 1) / (150 * np.pi * min([self.cfg.dx_f, self.cfg.dy_f]))

        for i in range(self.pml_cells):
            d = (p - i) / p
            val = eta_max * d ** self.m
            sx[i, :] = val # Left
            sx[-1-i, :] = val # Right
            sy[:, i] = val # Bottom
            sy[:, -1-i] = val # Top
        return sx.flatten(), sy.flatten()
        
    def _get_operators(self, n, d):
        Ix = sp.eye(n + 1, format='csr')
        Ahx = (sp.eye(n, n + 1, k=0) + sp.eye(n, n + 1, k=1)).tocsr()
        Ax = (sp.eye(n, n) + sp.eye(n, n, k=1)).tocsr()
        Atx = (sp.eye(n + 1, n + 1) + sp.eye(n + 1, n + 1, k=-1)).tolil()
        Dx = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
        Dtx = (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)).tolil()
        
        # Boundary conditions
        Dtx[n, n-1] = -2
        Atx[n, n] = 2
        
        return Ix, Ahx, Ax, Atx.tocsr(), Dx.tocsr(), Dtx.tocsr() / d

    def _build_system(self):
        Ix, Ahx, Ax, Atx, Dx, Dtx = self._get_operators(self.Nx, self.dx)
        Iy, Ahy, Ay, Aty, Dy, Dty = self._get_operators(self.Ny, self.dy)

        Mxx = (self.mu / self.dt) * sp.eye(self.len_hx)
        Myy = (self.mu / self.dt) * sp.eye(self.len_hy)
        Ezz_p = (self.eps / self.dt + self.sigma / 2) * sp.eye(self.len_ez)
        Ezz_m = (self.eps / self.dt - self.sigma / 2) * sp.eye(self.len_ez)

        # LHS
        L11, L13 = sp.kron(Ix, Ay) @ Mxx, sp.kron(Ix, Dy)
        L22, L23 = sp.kron(Ax, Iy) @ Myy, -sp.kron(Dx, Iy)
        L31, L32, L33 = sp.kron(Atx, Dty), -sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_p
        LHS = sp.bmat([[L11, None, L13], [None, L22, L23], [L31, L32, L33]], format='csr')
        
        # RHS
        R11, R13 = sp.kron(Ix, Ay) @ Mxx, -sp.kron(Ix, Dy)
        R22, R23 = sp.kron(Ax, Iy) @ Myy, sp.kron(Dx, Iy)
        R31, R32, R33 = -sp.kron(Atx, Dty), sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_m
        self.RHS = sp.bmat([[R11, None, R13], [None, R22, R23], [R31, R32, R33]], format='csr')

        self.solve_func = spla.factorized(LHS)

    def run_simulation(self, src_func, src_pos):
        u = np.zeros(self.total_len)
        ez_history = []
        movie_frames = []
        
        src_global_idx = (self.len_hx + self.len_hy) + (src_pos[0] * (self.Ny + 1) + src_pos[1])
        
        fig, ax = plt.subplots()
        for i in tqdm(range(self.Nt), desc="Simulating"):
            t = i * self.dt
            b = self.RHS.dot(u)
            b[src_global_idx] += src_func(t)
            u = self.solve_func(b)
            
            ez_2d = u[self.idx_ez].reshape((self.Nx + 1, self.Ny + 1))
            ez_history.append(ez_2d[src_pos[0], src_pos[1]])
            
            if i % 2 == 0:
                txt = ax.text(0.5, 1.05, f'Step: {i}/{self.Nt}', ha="center", transform=ax.transAxes)
                img = ax.imshow(ez_2d.T, cmap='RdBu', origin='lower', animated=True,
                                extent=[0, self.Nx*self.dx, 0, self.Ny*self.dy], vmin=-0.1, vmax=0.1)
                movie_frames.append([txt, img])
        
        ani = ArtistAnimation(fig, movie_frames, interval=50, blit=True)
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
    def run_full_analysis(cls, params, src_func, src_pos):
        # 1. Initialize
        solver = cls(**params)
        
        # 2. Run
        ani, ez_data = solver.run_simulation(src_func, src_pos)
        
        # 3. Plot 1D
        solver.plot_1d_intensity(solver.dt, ez_data)
        
        # 4. Show Animation
        plt.show()
        return ez_data

# --- Execution ---



sim_params = {'Nx': 100, 'Ny': 100, 'Nt': 200, 'lambda0': 1, 'CFL': 4}
source_position = (30, 30)

results = FCI_TM_Solver.run_full_analysis(sim_params, my_src, source_position)