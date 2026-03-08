import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class FCI_TM_Solver:
    def __init__(self, Nx, Ny, Nt, lambda0, CFL):
        self.eps, self.mu, self.sigma, c = 8.854e-12, 1.256e-6, 0.001, 3e8
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.lambda0, self.CFL = lambda0, CFL
        self.dx, self.dy = lambda0 / 30, lambda0 / 30
        self.dt = CFL / (c*np.sqrt(1/self.dx**2 + 1/self.dy**2))
        
        # 1. Component lengths
        self.len_hx = (Nx + 1) * Ny
        self.len_hy = Nx * (Ny + 1)
        self.len_ez = (Nx + 1) * (Ny + 1)
        self.total_len = self.len_hx + self.len_hy + self.len_ez
        
        # 2. Slice for Ez extraction
        self.idx_ez = slice(self.len_hx + self.len_hy, self.total_len)
        
        # 3. Initialize operators and matrices
        self._build_system()
        
    def _get_operators(self, n, d):
        Ix      = sp.eye(n + 1, format='csr')
        A_hx    = (sp.eye(n, n + 1, k=0) + sp.eye(n, n + 1, k=1)).tocsr()
        Ax      = (sp.eye(n, n) + sp.eye(n, n, k=1)).tocsr()
        Atx     = (sp.eye(n + 1, n + 1) + sp.eye(n + 1, n + 1, k=-1)).tolil()
        Dx      = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
        Dtx     = (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)).tolil()
        Dtx[n, n-1] = -2
        Atx[n, n] = 2
        
        return Ix, A_hx, Ax, Atx.tocsr(), Dx.tocsr(), Dtx.tocsr() / d

    def _build_system(self):
        Ix, Ahx, Ax, Atx, Dx, Dtx = self._get_operators(self.Nx, self.dx)
        Iy, Ahy, Ay, Aty, Dy, Dty = self._get_operators(self.Ny, self.dy)

        # Material Diagonals
        Mxx = (self.mu / self.dt) * sp.eye(self.len_hx)
        Myy = (self.mu / self.dt) * sp.eye(self.len_hy)
        Ezz_p = (self.eps / self.dt + self.sigma / 2) * sp.eye(self.len_ez)
        Ezz_m = (self.eps / self.dt - self.sigma / 2) * sp.eye(self.len_ez)

        # LHS Construction
        L11, L13 = sp.kron(Ix, Ay) @ Mxx, sp.kron(Ix, Dy)
        L22, L23 = sp.kron(Ax, Iy) @ Myy, -sp.kron(Dx, Iy)
        L31, L32, L33 = sp.kron(Atx, Dty), -sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_p

        LHS = sp.bmat([[L11, None, L13], [None, L22, L23], [L31, L32, L33]], format='csr')
        
        # RHS Construction
        R11, R13 = sp.kron(Ix, Ay) @ Mxx, -sp.kron(Ix, Dy)
        R22, R23 = sp.kron(Ax, Iy) @ Myy, sp.kron(Dx, Iy)
        R31, R32, R33 = -sp.kron(Atx, Dty), sp.kron(Dtx, Aty), sp.kron(Atx, Aty) @ Ezz_m

        self.RHS = sp.bmat([[R11, None, R13], [None, R22, R23], [R31, R32, R33]], format='csr')
        
        print("Factorizing LHS...")
        self.solve_func = spla.factorized(LHS)

    def run_simulation(self, Nt, src_func, src_pos):
        u = np.zeros(self.total_len)
        ez_history = []
        movie_frames = []
        
        # Calculate global index for source in Ez portion
        src_global_idx = (self.len_hx + self.len_hy) + (src_pos[0] * (self.Ny + 1) + src_pos[1])
        
        fig, ax = plt.subplots()
        for i in tqdm(range(Nt), desc=f"Simulating (Nt={Nt})"):
            t = i * self.dt
            b = self.RHS.dot(u)
            b[src_global_idx] += src_func(t)
            u = self.solve_func(b)
            
            ez_2d = u[self.idx_ez].reshape((self.Nx + 1, self.Ny + 1))
            ez_history.append(ez_2d[src_pos[0], src_pos[1]])
            
            if i % 2 == 0:
                txt = ax.text(0.5, 1.05, f'Step: {i}/{Nt}', ha="center", transform=ax.transAxes)
                img = ax.imshow(ez_2d.T, cmap='RdBu', origin='lower', animated=True,
                                extent=[0, self.Nx*self.dx, 0, self.Ny*self.dy], vmin=-0.1, vmax=0.1)
                movie_frames.append([txt, img])
                
        return ez_history, fig, movie_frames

#TODO: ADD 1D PLOT FROM FCI 003 & FIX DAMPENING OF WAVE???

params = {'Nx': 100, 'Ny': 100, 'Nt':100,'lambda0': 1, 'CFL': 2}
solver = FCI_TM_Solver(**params)


def my_src(t):
    f_c = 3e8 / 1.0
    t0 = 4 * (3 / (2 * np.pi * f_c))
    return 5 * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/(3/(2*np.pi*f_c)))**2)

history, fig, movie = solver.run_simulation(Nt=150, src_func=my_src, src_pos=(50, 50))
ani = ArtistAnimation(fig, movie, interval=50, blit=True)
plt.show()