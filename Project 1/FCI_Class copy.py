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
        self.sigma = 1e-10 if self.bc == 'PBC' else 0.0

        # Source Parameters
        self.f_c    = self.c / self.lambda0
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

    def my_src(self, t):
        # Gaussian Pulse: A * cos(2*pi*fc*(t-t0)) * exp(-0.5 * ((t-t0)/sig_t)**2)
        return self.A * np.cos(2 * np.pi * self.f_c * (t - self.t0)) * \
               np.exp(-0.5 * ((t - self.t0) / self.sig_t)**2)

    def _get_pml_profiles(self):
        sx = np.zeros((self.Nx + 1, self.Ny + 1))
        sy = np.zeros((self.Nx + 1, self.Ny + 1))
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
        Ix = sp.eye(n + 1, format='csr')
        Ahx = (sp.eye(n, n + 1, k=0) + sp.eye(n, n + 1, k=1)).tocsr()
        Ax = (sp.eye(n, n) + sp.eye(n, n, k=1)).tolil()
        Atx = (sp.eye(n + 1, n + 1) + sp.eye(n + 1, n + 1, k=-1)).tolil()
        Dx = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
        Dtx = (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)).tolil()
        
        if self.bc == 'PBC':
            Ax[n-1, 0] = 1 
            Dtx[0, n-1] = -1
            Atx[0, n-1] = 1
        else:
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
                       format='csr').tolil()
        
        # RHS
        R11, R13 = sp.kron(Ix, Ay) @ Mxx, -sp.kron(Ix, Dy)
        R22, R23 = sp.kron(Ax, Iy) @ Myy, sp.kron(Dx, Iy)
        R31, R32, R33 = -sp.kron(Atx, Dty), sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_m
        self.RHS = sp.bmat([[R11, None, R13],
                            [None, R22, R23], 
                            [R31, R32, R33]], 
                            format='csr').tolil()
        
        if self.bc == 'PBC':
            # --- Apply Exact Algebraic Boundary Constraints (Var[N] - Var[0] = 0) ---
            def apply_pbc(idx_dup, idx_src):
                LHS[idx_dup, :] = 0
                self.RHS[idx_dup, :] = 0
                LHS[idx_dup, idx_dup] = 1
                LHS[idx_dup, idx_src] = -1
            
            # 1. Hx wrapping in X
            for j in range(self.Ny):
                apply_pbc(self.Nx * self.Ny + j, 0 * self.Ny + j)
                
            # 2. Hy wrapping in Y
            for i in range(self.Nx):
                apply_pbc(self.len_hx + i * (self.Ny + 1) + self.Ny, self.len_hx + i * (self.Ny + 1) + 0)
                
            # 3. Ez wrapping in X and Y
            ez_off = self.len_hx + self.len_hy
            for j in range(self.Ny + 1): # Map right edge to left edge
                apply_pbc(ez_off + self.Nx * (self.Ny + 1) + j, ez_off + 0 * (self.Ny + 1) + j)
            for i in range(self.Nx):     # Map top edge to bottom edge
                apply_pbc(ez_off + i * (self.Ny + 1) + self.Ny, ez_off + i * (self.Ny + 1) + 0)

        self.solve_func = spla.factorized(LHS.tocsc())
        self.RHS = self.RHS.tocsr()

    def run_simulation(self, src_pos):
        u = np.zeros(self.total_len)
        ez_history = []
        movie_frames = []
        
        # Global index for Ez component at src_pos
        src_global_idx = (self.len_hx + self.len_hy) + (src_pos[0] * (self.Ny + 1) + src_pos[1])
        
        fig, ax = plt.subplots()
        for i in tqdm(range(self.Nt), desc=f"Simulating ({self.bc})"):
            t = i * self.dt
            b = self.RHS.dot(u)
            b[src_global_idx] += self.my_src(t)
            u = self.solve_func(b)
            ez_2d = u[self.idx_ez].reshape((self.Nx + 1, self.Ny + 1))
            ez_history.append(ez_2d[src_pos[0], src_pos[1]])
            
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
        src_pos = params.pop('src_pos', (params['Nx']//2, params['Ny']//2))
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
    'bc': 'PBC'
}

results = FCI_TM_Solver.run_full_analysis(sim_params)
