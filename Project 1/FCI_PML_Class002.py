import time
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.special as sp_special
from scipy.fft import fft, fftfreq

from Yee_Class002 import SimulationConfig

class FCI_TM_Solver:
    def __init__(self, config): #Nx, Ny, Nt, lambda0, CFL, bc='PEC', solver="Schur", finesse=20):
        self.cfg = config


        # PBC drops the redundant N+1 boundary node to form a perfect ring
        if self.cfg.bc == 'PBC':
            self.nx_n = self.cfg.nx
            self.ny_n = self.cfg.ny
            self.cfg.dx = np.append(self.cfg.dx, self.cfg.dx_0)
            self.cfg.dy = np.append(self.cfg.dy, self.cfg.dy_0)
        else:
            self.nx_n = self.cfg.nx + 1
            self.ny_n = self.cfg.ny + 1
            self.cfg.dx = np.append(self.cfg.dx, self.cfg.dx_0)
            self.cfg.dy = np.append(self.cfg.dy, self.cfg.dy_0)
        

        self.cfg.sigma = np.zeros((self.nx_n, self.ny_n)) #Define Drude media (so sigma_DC)
        self.cfg.gamma = 0.0
        self.cfg.epsilon_r  = np.ones((self.nx_n, self.ny_n))

        self.len_hx = self.nx_n * self.cfg.ny
        self.len_hy = self.cfg.nx * self.ny_n
        self.len_ez = self.nx_n * self.ny_n
        
        # Hx & Hx_dot + Hy & Hy_dot + Ez & Ez_dot & Ez_ddot & Jz:
        self.total_len = 2 * self.len_hx + 2 * self.len_hy + 4 * self.len_ez
        ez_start = 2*self.len_hx + 2*self.len_hy
        self.idx_ez = slice(ez_start, ez_start + self.len_ez)
        
        self._build_system()

    def my_src(self, t):
        # Gaussian Pulse: A * cos(2*pi*fc*(t-t0)) * exp(-0.5 * ((t-t0)/sig_t)**2)
        return self.cfg.A * np.cos(2 * np.pi * self.cfg.f_c * (t - self.cfg.t0)) * \
               np.exp(-0.5 * ((t - self.cfg.t0) / self.cfg.sig_t)**2)

    def _get_pml_profiles(self):
        self.kx = np.ones((self.nx_n,self.ny_n))
        self.ky = np.ones((self.nx_n,self.ny_n))
        self.sx = np.zeros((self.nx_n,self.ny_n))
        self.sy = np.zeros((self.nx_n,self.ny_n))

        p, m = int(1 * self.cfg.finesse), 4
        k_max = 2
        s_max = (m + 1) / (150 * np.pi * min([self.cfg.dx_0, self.cfg.dy_0]))

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
        if self.cfg.bc == 'PBC':
            ones = np.ones(n)
            Ix = sp.eye(n, format='csr')
            
            Dx = sp.diags(1/(2*d), offsets=0, shape=(n, n), format='csr') @ sp.diags_array([-ones, ones, ones], offsets=[0, 1, 1-n], shape=(n, n), format='csr')
            Dtx = sp.diags(1/(2*d), offsets=0, shape=(n, n), format='csr') @ sp.diags_array([ones, -ones, -ones], offsets=[0, -1, n-1], shape=(n, n), format='csr')
            
            A1 = sp.diags_array([ones, ones, ones], offsets=[0, -1, n-1], shape=(n, n), format='csr')
            A2 = sp.diags_array([ones, ones, ones], offsets=[0, 1, 1-n], shape=(n, n), format='csr')
            
            return Ix, Dx, Dtx, A1, A2
        
        else:
            Ix = sp.eye(n + 1, format='csr')

            Dx = sp.diags(1/(2*d), offsets=0, shape=(n, n), format='csr') @ (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0))
            Dtx = ((sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)) / (2*d)).tolil()
            Dtx[n, n-1] = - 1 / d

            A1 = sp.diags_array([1, 1], offsets=[0, -1], shape=(n, n-1), format='csr')
            A2 = sp.diags_array([1, 1], offsets=[0, 1], shape=(n, n+1), format='csr')

            return Ix, Dx.tocsr(), Dtx.tocsr(), A1.tocsr(), A2.tocsr()

    def _build_system(self):
        v, c, Z, gamma, dt = self.cfg.v_local, self.cfg.c, self.cfg.Z_local, self.cfg.gamma, self.cfg.dt
        sx, sy, kx, ky = self._get_pml_profiles()

        Ix, Dx, Dtx, Ax1, Ax2 = self._get_operators(self.nx_n, self.cfg.dx)
        Iy, Dy, Dty, Ay1, Ay2 = self._get_operators(self.ny_n, self.cfg.dy)
        
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
            return sp.diags_array(vec, format='csr')
            
        def to_diag_scal(val, length):
            val_flat = np.array(val).flatten()
            return sp.diags_array(np.full(length, val_flat), format='csr')
            
        L11 = to_diag_scal(1.0/(c*dt), self.len_hx);    L12 = to_diag_vec(-bxp_hx)
        L22 = to_diag_vec(byp_hx);                      L25 = DY_ez_to_hx
        L33 = to_diag_vec(bxp_hy);                      L34 = to_diag_vec(-byp_hy)
        L44 = to_diag_scal(1.0/(c*dt), self.len_hy);    L45 = -DX_ez_to_hy
        L55 = to_diag_vec(byp);                         L56 = to_diag_scal(-1.0/(v*dt), self.len_ez)
        L66 = to_diag_vec(bxp);                         L67 = -I_ez / (c * dt)
        L71 = DY_hx_to_ez;      L73 = -DX_hy_to_ez;     L77 = I_ez / (c * dt);      L78 = I_ez / 2.0
        L87 = to_diag_vec(-self.cfg.sigma.flatten());       L88 = to_diag_scal(2.0*gamma/dt + 1.0, self.len_ez)

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
        R87 = to_diag_vec(self.cfg.sigma.flatten());        R88 = to_diag_scal(2.0*gamma/dt - 1.0, self.len_ez)

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

        if self.cfg.solver == "default":
            print("Pre-factoring the 8x8 system...")
            self.solve_func = spla.factorized(self.LHS.tocsc())

        elif self.cfg.solver == "Schur":
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
            

    def run_simulation(self):
        u = np.zeros(self.total_len)
        ez_history = []
        
        # Global index for Ez component at src_pos
        offset = 2 * self.len_hx + 2 * self.len_hy
        x0, y0 = self.cfg.x0, self.cfg.y0

        src_idx = offset + x0 * self.ny_n + y0

        # shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        # neighbors = [offset + ((x0 + dx) % self.nx_n) * self.ny_n + ((y0 + dy) % self.ny_n) for dx, dy in shifts]
        # weights = np.array([0.5, 0.125, 0.125, 0.125, 0.125])


        for i in tqdm(range(self.cfg.nt), desc=f"Simulating ({self.cfg.bc})"):
            t = i * self.cfg.dt
            b = self.RHS.dot(u)

            #Smooth source over a couple of grid points to prevent checkerboarding
            src_val = self.my_src(t)
            b[src_idx] -= src_val
            # for idx, w in zip(neighbors, weights):
            #     b[idx] += src_val * w

            u = self.solve_func(b)
            ez_2d = u[self.idx_ez].reshape((self.nx_n, self.ny_n))
            ez_history.append(ez_2d)
            
        return np.array(ez_history)

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
    def run_plot(cls, params):

        config = SimulationConfig(**params)
        t0 = time.time()
        solver = cls(config)
        
        # Run simulation
        ez_data = solver.run_simulation()
        t1 = time.time()
        exec_time = t1 - t0
        print(f"\n>>> [{solver.cfg.solver} solver] Execution time (Setup + Sim): {exec_time:.4f} seconds <<<\n")
        
        nodes_x = np.concatenate(([0], np.cumsum(solver.cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(solver.cfg.dy)))
        X, Y = np.meshgrid(nodes_x, nodes_y)
        fig, ax = plt.subplots(figsize=(10, 5))

        quad = ax.pcolormesh(X, Y, (ez_data[0]*config.Z_local).T, shading='auto')

        def update(i):
            frame_data = (ez_data[i]*config.Z_local).T
            quad.set_array(frame_data.ravel())
            return quad,
        
        anim = FuncAnimation(fig, update, frames=len(ez_data), interval=1000*config.dt, blit=True)
        plt.show()
        

        return ez_data, exec_time


sim_params = {
    'CFL': 2,
    'Obs_loc' : (80,100),
    'bc': 'PBC',
    'solver': 'Schur',
    'finesse': 10,
    'frame_skip': 3,
    'hankel_f_min': 0.0,
    'hankel_f_max': 3 * 299792458,
    'grid_refinement': False
}

results, time_schur = FCI_TM_Solver.run_plot(sim_params)



print(f"Schur complement solver time: {time_schur:.4f} seconds")