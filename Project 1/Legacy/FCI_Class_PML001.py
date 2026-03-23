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
        self.f_c    = c / self.lambda0
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t
        
        # Component lengths
        self.len_hx = Nx * Ny
        self.len_hy = Nx * Ny
        self.len_ez = Nx * Ny
        self.total_len = self.len_hx + self.len_hy + self.len_ez
        self.idx_ez = slice(0, self.len_ez)
        self.idx_hx = slice(self.len_ez, self.len_ez + self.len_hx)
        self.idx_hy = slice(self.len_ez + self.len_hx, self.total_len)
        self._build_system()
    
    @staticmethod
    def my_src():
        # t = np.linspace(0,self.Nt * self.dt)
        f_c = 3e8 / 1.0
        t0 = 4 * (3 / (2 * np.pi * f_c))
        # Gaussian Pulse
        return lambda t:5 * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/(3/(2*np.pi*f_c)))**2)    

    def _get_pml_profiles(self):
        sx = np.zeros((self.Nx, self.Ny))
        sy = np.zeros((self.Nx, self.Ny))
        p, m = 20, 4
        eta_max = (m + 1) / (150 * np.pi * min([self.dx, self.dy]))

        for i in range(self.pml_cells):
            d = (p - i) / p
            val = eta_max * d ** self.m
            sx[i, :] = val # Left
            sx[-1-i, :] = val # Right
            sy[:, i] = val # Bottom
            sy[:, -1-i] = val # Top
        return sx.flatten(), sy.flatten()
        
    def _get_operators(self, n, d):
        Ix = sp.eye(n, format='csc')
        Ax = (sp.eye_array(n) + sp.eye_array(n, k=1) + sp.eye_array(n, k=-n+1)).tocsc()
        Dx = (sp.eye_array(n, k=1) - sp.eye_array(n) + sp.eye_array(n, k=-n+1)) / d

        return Ix, Ax, Dx.tocsc()

    def _build_system(self):
        Ix, Ax, Dx = self._get_operators(self.Nx, self.dx)
        Iy, Ay, Dy = self._get_operators(self.Ny, self.dy)

        Mxx = (self.mu / self.dt) * sp.eye(self.len_hx)
        Myy = (self.mu / self.dt) * sp.eye(self.len_hy)
        Ezz_p = (self.eps / self.dt + self.sigma / 2) * sp.eye(self.len_ez)
        Ezz_m = (self.eps / self.dt - self.sigma / 2) * sp.eye(self.len_ez)

        # LHS
        L11, L12, L13 = sp.kron(Ax, Ay) @ Ezz_p, sp.kron(Ax,Dy),-sp.kron(Dx, Ay)
        L21, L22 = sp.kron(Ix,Dy), sp.kron(Ix,Ay) @ Mxx
        L31, L33 = -sp.kron(Dx,Iy), sp.kron(Ax, Iy) @ Myy
        LHS = sp.bmat([[L11, L12, L13], 
                       [L21, L22, None], 
                       [L31, None, L33]], format='csc')
        
        # RHS
        R11 = sp.kron(Ax, Ay) @ Ezz_m 
        
        self.RHS = sp.bmat([[R11, -L12, -L13], 
                            [-L21, L22, None], 
                            [-L31, None, L33]], format='csc')

        self.solve_func = spla.factorized(LHS)

    def run_simulation(self, src_func, src_pos):
        u = np.zeros(self.total_len)
        ez_history = []
        movie_frames = []
        
        src_global_idx = src_pos[0] * self.Ny + src_pos[1]
        
        fig, ax = plt.subplots()
        for i in tqdm(range(self.Nt), desc="Simulating"):
            t = i * self.dt
            b = self.RHS.dot(u)
            b[src_global_idx] += src_func(t)
            b[src_global_idx + 2] += src_func(t)
            u = self.solve_func(b)
            
            ez_2d = u[self.idx_ez].reshape((self.Nx, self.Ny))
            ez_history.append(ez_2d[src_pos[0]+10, src_pos[1]+10])
            
            # if i % 2 == 0:
            txt = ax.text(0.5, 1.05, f'Step: {i}/{self.Nt}', ha="center", transform=ax.transAxes)
            img = ax.imshow(ez_2d.T, cmap='RdBu', origin='lower', animated=True,
                            extent=[0, self.Nx*self.dx, 0, self.Ny*self.dy], vmin=-0.3, vmax=0.3)
            movie_frames.append([txt, img])
        
        ani = ArtistAnimation(fig, movie_frames, interval=200, blit=True)
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
    def run_full_analysis(cls, params, src_func, src_pos=(30, 30)):
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



sim_params = {'Nx': 100, 'Ny': 100, 'Nt': 200, 'lambda0': 1, 'CFL': 3}
source_position = (40, 40)

        
results = FCI_TM_Solver.run_full_analysis(sim_params, FCI_TM_Solver.my_src(), source_position)

