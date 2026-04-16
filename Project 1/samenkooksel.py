import time
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import scipy.special as sp_special
from scipy.fft import fft, fftfreq

# ==============================================================================
# 1. Configuration
# ==============================================================================
class SimulationConfig:
    """Handles all physical and numerical parameters."""
        
    def __init__(self, **kwargs):
        self.solver_type = kwargs.get("solver_type", "yee").lower()

        # FCI settings specifically
        self.fci_bc = kwargs.get("bc", "PBC").upper()
        self.fci_solver = kwargs.get("fci_solver", "Schur")
        
        # Physical Constants
        self.c          = 299792458
        self.epsilon0   = 8.854e-12
        self.mu0        = 4 * np.pi * 1e-7
        self.Z0         = np.sqrt(self.mu0 / self.epsilon0)
        
        # Material Properties
        self.eps_clad   = 2.218**2
        self.eps_core   = 2.22**2
        self.gamma_f    = 2.0

        # Source Parameters
        self.lam_c  = 1.0
        self.f_c    = self.c / self.lam_c
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t
        self.f_break = 2 * self.f_c
        
        # Dimensions expressed in amount of wavelengths
        f = 1.5
        self.L_wg   = f * 10 * self.lam_c
        self.w_core = f * 3 * self.lam_c
        self.w_clad = f * 3 * self.lam_c
        self.w_air  = f * 2.5 * self.lam_c
        self.d      = f * 4 * self.lam_c
        self.t_m    = f * 0.03 * self.lam_c
        self.Ll     = f * 5 * self.lam_c
        self.Lr     = f * 2.5 * self.lam_c
        self.L      = self.Ll + self.d + self.t_m + self.L_wg + self.Lr
        self.W      = f * (2 * self.w_air + 2 * self.w_clad + self.w_core)
        self.T      = self.t0 + 3*self.sig_t + (self.d + self.t_m + np.sqrt(self.eps_core) * self.L_wg + self.Lr) / self.c
        self.finesse = kwargs.get("finesse", 30)
        self.f_min  = self.c / self.W

        # Update params with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.wg_type = getattr(self, "wg_type", "step")
        self.grid_refinement = getattr(self, "grid_refinement", "gradual")
        self.alpha = getattr(self, "alpha", 1.05)
        
        # Build Grids
        self._build_dx()
        self._build_dy()
        
        if self.fci_bc == "PBC" and self.solver_type == "fci":
            self.nx = len(self.dx)
            self.ny = len(self.dy)
        else:
            self.nx = len(self.dx) + 1
            self.ny = len(self.dy) + 1
        
        self.epsilon_r  = np.ones((self.nx, self.ny))
        self.sigma      = np.zeros((self.nx-2, self.ny-2))
        self.gamma      = np.zeros((self.nx-2, self.ny-2))

        self.dx_f = self.dx.min()
        self.dy_f = self.dy.min()
        
        self.x0 = getattr(self, "x0", self.n_Ll)
        self.y0 = getattr(self, "y0", self.ny // 2)
        
        self.x1 = self.x0 + kwargs["x1"] if "x1" in kwargs else self.nx - 50
        self.y1 = self.y0 + kwargs["y1"] if "y1" in kwargs else self.y0

        dist_req = self.d / 2.0
        offset_x = np.searchsorted(np.cumsum(self.dx[self.x0:]), dist_req)
        offset_y = np.searchsorted(np.cumsum(self.dy[self.y0:]), dist_req)
        self.x_diag = getattr(self, "x_diag", self.x0 + offset_x)
        self.y_diag = getattr(self, "y_diag", self.y0 + offset_y)

        self.x_after = getattr(self, "x_after", self.nx - 50)
        self.y_after = getattr(self, "y_after", self.y0)
        self.x_before = getattr(self, "x_before", self.x0 + int(self.n_d // 2))
        self.y_before = getattr(self, "y_before", self.y0)

        self.dx_d = np.concatenate(([self.dx[0]/2], (self.dx[:-1] + self.dx[1:])/2, [self.dx[-1]/2]))
        self.dy_d = np.concatenate(([self.dy[0]/2], (self.dy[:-1] + self.dy[1:])/2, [self.dy[-1]/2]))
        
        if self.solver_type == "fci":
            CFL_default = 2.0
        else:
            CFL_default = 0.95

        self.CFL = getattr(self, "CFL", CFL_default)
        self.dt  = self.CFL / (self.c * np.sqrt((1/self.dx.min()**2) + (1/self.dy.min()**2)))
        self.nt = getattr(self, "nt", int(np.ceil(self.T / self.dt)))
        
        self.free_space_sim = getattr(self, "free_space_sim", False)
        self._setup_waveguide()
        
        self.v_local = self.c / np.sqrt(self.epsilon_r)
        self.Z_local = self.Z0 / np.sqrt(self.epsilon_r)


    def _build_dx(self):
        self.dx_0 = self.lam_c / self.finesse
        if not getattr(self, "grid_refinement", True):
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.n_d = int(np.ceil(self.d / self.dx_0))
            self.n_wg = int(np.ceil(self.L_wg / self.dx_0))
            self.n_Lr = int(np.ceil(self.Lr / self.dx_0))
            self.dx = np.concatenate([
                np.full(self.n_Ll, self.Ll / self.n_Ll),
                np.full(self.n_d, self.d / self.n_d),
                np.full(self.n_wg, self.L_wg / self.n_wg),
                np.full(self.n_Lr, self.Lr / self.n_Lr)
            ])
            return

        if self.grid_refinement == "gradual":    
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.L_f_dt, self.n_f_dt = self._L_and_n_fine(self.dx_0, self.t_m, self.alpha)
            self.n_d = int(np.ceil(self.d / self.dx_0 - self.L_f_dt / self.dx_0)) + self.n_f_dt
            self.L_f_twg, self.n_f_twg = self._L_and_n_fine(self.dx_0 / np.sqrt(self.eps_core), self.t_m, self.alpha)
            self.n_wg = int(np.ceil((self.L_wg - self.L_f_twg) * np.sqrt(self.eps_core)/ self.dx_0) + self.n_f_twg)
            self.L_f_wg_Lr, self.n_f_wg_Lr = self._L_and_n_fine(self.dx_0, self.dx_0 / np.sqrt(self.eps_core), self.alpha)
            self.n_Lr = int(np.ceil(self.Lr / self.dx_0 - self.L_f_wg_Lr / self.dx_0)) + self.n_f_wg_Lr
            
            self.dx = np.concatenate([
                np.full(self.n_Ll, self.Ll / self.n_Ll),
                np.full(int(self.n_d - self.n_f_dt), (self.d - self.L_f_dt) / (self.n_d - self.n_f_dt)),
                self.alpha ** np.arange(self.n_f_dt, -1, -1) * self.t_m,
                self.alpha ** np.arange(1, self.n_f_twg + 1) * self.t_m,
                np.full(int(self.n_wg - self.n_f_twg), (self.L_wg - self.L_f_twg) / (self.n_wg - self.n_f_twg)),
                self.alpha ** np.arange(1, self.n_f_wg_Lr + 1) * self.dx_0 / np.sqrt(self.eps_core),
                np.full(int(self.n_Lr - self.n_f_wg_Lr), (self.Lr - self.L_f_wg_Lr) / (self.n_Lr - self.n_f_wg_Lr))
            ])

        if self.grid_refinement == "step":
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.n_d = int(np.ceil(self.d / self.dx_0))
            self.n_wg = int(np.ceil(self.L_wg / (self.dx_0 / np.sqrt(self.eps_core))))
            self.n_Lr = int(np.ceil(self.Lr / self.dx_0))
            self.dx = np.concatenate([
                np.full(self.n_Ll, self.Ll / self.n_Ll),
                np.full(self.n_d, self.d / self.n_d),
                np.array([self.t_m]),
                np.full(self.n_wg, (self.L_wg / self.n_wg)),
                np.full(self.n_Lr, self.Lr / self.n_Lr)
            ])

    def _build_dy(self):
        self.dy_0 = self.lam_c / self.finesse
        if not getattr(self, "grid_refinement", True):
            self.n_air = int(np.ceil(self.w_air / self.dy_0))
            self.n_clad = int(np.ceil(self.w_clad / self.dy_0))
            self.n_core = int(np.ceil(self.w_core / self.dy_0))
            self.dy = np.concatenate([
                np.full(self.n_air, self.w_air / self.n_air),
                np.full(self.n_clad, self.w_clad / self.n_clad),
                np.full(self.n_core, self.w_core / self.n_core),
                np.full(self.n_clad, self.w_clad / self.n_clad),
                np.full(self.n_air, self.w_air / self.n_air)
            ])
            return
        
        if self.grid_refinement == "gradual":    
            self.L_f_ac, self.n_f_ac = self._L_and_n_fine(self.dy_0, self.dy_0 / np.sqrt(self.eps_clad), self.alpha)
            self.n_air = int(np.ceil(self.w_air / self.dy_0 - self.L_f_ac / self.dy_0)) + self.n_f_ac
            
            if self.wg_type == "step":
                self.n_clad = int(np.ceil(self.w_clad / self.dy_0 * np.sqrt(self.eps_clad)))
                self.n_core = int(np.ceil(self.w_core / self.dy_0 * np.sqrt(self.eps_core)))
                dy_mid = [
                    np.full(self.n_clad, self.w_clad / self.n_clad),
                    np.full(self.n_core, self.w_core / self.n_core),
                    np.full(self.n_clad, self.w_clad / self.n_clad)
                ]
            else:
                self.deps_max = 0.01 # percentage of (self.eps_core - self.eps_clad)
                if self.eps_core != self.eps_clad:
                    self.a_eps = 2 * np.sqrt(self.eps_core) * (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) / (self.w_core / 2) ** 2
                    self.b_eps = (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) ** 2 / (self.w_core / 2) ** 4
                    self.dy_core = min(np.abs(self.deps_max * (self.eps_core-self.eps_clad) / (self.a_eps * 2 + 4 * self.b_eps ** 3)), self.dy_0 / np.sqrt(self.eps_core))
                else:
                    self.dy_core = self.dy_0 / np.sqrt(self.eps_core)
                
                self.L_f_cc, self.n_f_cc = self.L_and_n_fine(self.dy_0 / np.sqrt(self.eps_clad), self.dy_core, self.alpha)
                self.n_clad = int(np.ceil((self.w_clad - self.L_f_cc) / self.dy_0 * np.sqrt(self.eps_clad))) + self.n_f_cc
                self.n_core = int(np.ceil(self.w_core / self.dy_core))
                
                dy_mid = [
                    np.full(self.n_clad - self.n_f_cc, (self.w_clad - self.L_f_cc) / (self.n_clad - self.n_f_cc)),
                    self.alpha ** np.arange(self.n_f_cc, 0, -1) * self.dy_core,
                    np.full(self.n_core, self.w_core / self.n_core),
                    self.alpha ** np.arange(1, self.n_f_cc + 1) * self.dy_core,
                    np.full(self.n_clad - self.n_f_cc, (self.w_clad - self.L_f_cc) / (self.n_clad - self.n_f_cc))
                ]
                
            self.dy = np.concatenate([
                np.full(int(self.n_air - self.n_f_ac), (self.w_air - self.L_f_ac) / (self.n_air - self.n_f_ac)),
                self.alpha ** np.arange(self.n_f_ac, 0, -1) * self.dy_0 / np.sqrt(self.eps_clad),
                *dy_mid,
                self.alpha ** np.arange(1, self.n_f_ac + 1) * self.dy_0 / np.sqrt(self.eps_clad),
                np.full(int(self.n_air - self.n_f_ac), (self.w_air - self.L_f_ac) / (self.n_air - self.n_f_ac))
            ])

        if self.grid_refinement == "step":
            self.n_air = int(np.ceil(self.w_air / self.dy_0))
            self.n_clad = int(np.ceil(self.w_clad / self.dy_0 * np.sqrt(self.eps_clad)))

            if self.wg_type == "step":
                self.n_core = int(np.ceil(self.w_core / self.dy_0 * np.sqrt(self.eps_core)))
            else:
                self.deps_max = 0.01 # percentage of (self.eps_core - self.eps_clad)
                if self.eps_core != self.eps_clad:
                    self.a_eps = 2 * np.sqrt(self.eps_core) * (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) / (self.w_core / 2) ** 2
                    self.b_eps = (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) ** 2 / (self.w_core / 2) ** 4
                    self.dy_core = min(np.abs(self.deps_max * (self.eps_core-self.eps_clad) / (self.a_eps * 2 + 4 * self.b_eps ** 3)), self.dy_0 / np.sqrt(self.eps_core))
                else:
                    self.dy_core = self.dy_0 / np.sqrt(self.eps_core)
                self.n_core = int(np.ceil(self.w_core / self.dy_core))
                
            self.dy = np.concatenate([
                np.full(self.n_air, self.w_air / self.n_air),
                np.full(self.n_clad, self.w_clad / self.n_clad),
                np.full(self.n_core, self.w_core / self.n_core),
                np.full(self.n_clad, self.w_clad / self.n_clad),
                np.full(self.n_air, self.w_air / self.n_air)
            ])
        
    def _L_and_n_fine(self, d_coarse, d_fine, alpha = np.sqrt(2)):
        if self.eps_clad == self.eps_core:
            return 0, 0
        n_f = int(np.ceil(np.log(d_coarse / d_fine) / np.log(alpha))) - 1
        L_f = d_fine * (alpha**(n_f-1) - alpha) / (alpha - 1)
        return L_f, n_f
    
    def _setup_waveguide(self):
        if self.free_space_sim:
            return
        self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air):int(-self.n_air)] = self.eps_clad
        if self.wg_type == "step":
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = self.eps_core
        else:
            eps_val = lambda y: self.eps_core + self.a_eps * (y-self.W/2)**2 + self.b_eps * (y-self.W/2)**4 
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = eps_val(self.dy_core * np.arange(self.n_air + self.n_clad, (self.n_air + self.n_clad + self.n_core) + 1))
            
        self.sigma[int(self.n_Ll + self.n_d - 1),   :int(self.n_air + self.n_clad)] = 3.5e7
        self.sigma[int(self.n_Ll + self.n_d - 1), - int(self.n_air + self.n_clad):] = 3.5e7

# ==============================================================================
# 2. Solvers
# ==============================================================================
class YeeSolver:
    def __init__(self, config):
        self.cfg = config
        self.ix, self.iy = slice(1,-1), slice(1,-1)
        self.Ez      = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Ez_dot  = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Ez_ddot = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Jc      = np.zeros((self.cfg.nx, self.cfg.ny))
        self.Hx      = np.zeros((self.cfg.nx, self.cfg.ny-1))
        self.Hx_dot  = np.zeros((self.cfg.nx, self.cfg.ny-1))
        self.Hy      = np.zeros((self.cfg.nx-1, self.cfg.ny))
        self.Hy_dot  = np.zeros((self.cfg.nx-1, self.cfg.ny))
        self.Hx_dot_old = np.zeros_like(self.Hx_dot)
        self.Hy_dot_old = np.zeros_like(self.Hy_dot)
        self.Ez_ddot_old = np.zeros_like(self.Ez_ddot)
        self.Ez_dot_old = np.zeros_like(self.Ez_dot)

        self._init_pml()
        self._init_coefficients()

    def _init_pml(self):
        p, m = int(2.0 * self.cfg.finesse), 4
        eta_max = (m + 1) / (150 * np.pi * min([self.cfg.dx_0, self.cfg.dy_0]))
        self.kx, self.ky = np.ones((self.cfg.nx, self.cfg.ny)), np.ones((self.cfg.nx, self.cfg.ny))
        self.etax, self.etay = np.zeros((self.cfg.nx, self.cfg.ny)), np.zeros((self.cfg.nx, self.cfg.ny))

        for i in range(p):
            d_pml = (p - i) / p
            val_k = 1.0
            val_eta = eta_max * (d_pml**m)
            self.kx[i, :], self.kx[-1-i, :] = val_k, val_k
            self.ky[:, i], self.ky[:, -1-i] = val_k, val_k
            self.etax[i, :], self.etax[-1-i, :] = val_eta, val_eta
            self.etay[:, i], self.etay[:, -1-i] = val_eta, val_eta

    def _init_coefficients(self):
        cfg = self.cfg
        self.bxp = self.kx / (cfg.v_local * cfg.dt) + cfg.Z_local * self.etax / 2.0
        self.byp = self.ky / (cfg.v_local * cfg.dt) + cfg.Z_local * self.etay / 2.0
        self.bxm = self.kx / (cfg.v_local * cfg.dt) - cfg.Z_local * self.etax / 2.0
        self.bym = self.ky / (cfg.v_local * cfg.dt) - cfg.Z_local * self.etay / 2.0

        self.byp_hx = (self.byp[:, :-1] + self.byp[:, 1:]) / 2.0
        self.bym_hx = (self.bym[:, :-1] + self.bym[:, 1:]) / 2.0
        self.byp_hy = (self.byp[:-1, :] + self.byp[1:, :]) / 2.0
        self.bym_hy = (self.bym[:-1, :] + self.bym[1:, :]) / 2.0
        self.bxp_hx = (self.bxp[:, :-1] + self.bxp[:, 1:]) / 2.0
        self.bxm_hx = (self.bxm[:, :-1] + self.bxm[:, 1:]) / 2.0
        self.bxp_hy = (self.bxp[:-1, :] + self.bxp[1:, :]) / 2.0
        self.bxm_hy = (self.bxm[:-1, :] + self.bxm[1:, :]) / 2.0

        self.bz_h = 1.0 / (cfg.c * cfg.dt)
        self.bz_e = 1.0 / (cfg.v_local * cfg.dt)
        self.ap = 2.0 * cfg.gamma / cfg.dt + 1.0
        self.am = 2.0 * cfg.gamma / cfg.dt - 1.0

        sub_v = cfg.v_local[1:-1, 1:-1]
        sub_z = cfg.Z_local[1:-1, 1:-1]
        self.coef_n = (1.0 / (sub_v * cfg.dt) - sub_z * cfg.sigma / (2.0 * self.ap))
        self.coef_p = (1.0 / (sub_v * cfg.dt) + sub_z * cfg.sigma / (2.0 * self.ap))
        self.coef_j = 0.5 * (1.0 + self.am / self.ap)

        self.inv_dx, self.inv_dy = 1.0 / cfg.dx, 1.0 / cfg.dy
        self.inv_dx_d, self.inv_dy_d = 1.0 / cfg.dx_d, 1.0 / cfg.dy_d

    def step(self, t):
        cfg = self.cfg
        ix, iy = self.ix, self.iy

        self.Hx_dot_old[:] = self.Hx_dot
        diff_Ez_y = self.Ez[:, 1:] - self.Ez[:, :-1]
        self.Hx_dot = (self.bym_hx * self.Hx_dot - diff_Ez_y * self.inv_dy[None, :]) / self.byp_hx
        self.Hx += (self.bxp_hx * self.Hx_dot - self.bxm_hx * self.Hx_dot_old) / self.bz_h

        self.Hy_dot_old[:] = self.Hy_dot
        diff_Ez_x = self.Ez[1:, :] - self.Ez[:-1, :]
        self.Hy_dot += diff_Ez_x * self.inv_dx[:, None] / self.bz_h
        self.Hy = (self.bxm_hy * self.Hy + (self.byp_hy * self.Hy_dot - self.bym_hy * self.Hy_dot_old)) / self.bxp_hy

        diff_Hy_x = self.Hy[1:, iy] - self.Hy[:-1, iy]
        diff_Hx_y = self.Hx[ix, 1:] - self.Hx[ix, :-1]
        curl_h = diff_Hy_x * self.inv_dx_d[ix, None] - diff_Hx_y * self.inv_dy_d[None, iy]
        
        self.Ez_ddot_old[:] = self.Ez_ddot
        self.Ez_ddot[ix, iy] = (self.coef_n * self.Ez_ddot[ix, iy] - self.coef_j * self.Jc[ix, iy] + curl_h) / self.coef_p
        
        avg_Ez_ddot = self.Ez_ddot[ix, iy] + self.Ez_ddot_old[ix, iy]
        self.Jc[ix, iy] = (self.am * self.Jc[ix, iy] + cfg.sigma * cfg.Z_local[ix, iy] * avg_Ez_ddot) / self.ap
        
        self.Ez_dot_old[:] = self.Ez_dot
        diff_Ez_ddot = self.Ez_ddot[ix, iy] - self.Ez_ddot_old[ix, iy]
        self.Ez_dot[ix, iy] = (self.bxm[ix, iy] * self.Ez_dot[ix, iy] + diff_Ez_ddot / (cfg.v_local[ix, iy] * cfg.dt)) / self.bxp[ix, iy]

        diff_Ez_dot = self.Ez_dot[ix, iy] - self.Ez_dot_old[ix, iy]
        self.Ez[ix, iy] = (self.bym[ix, iy] * self.Ez[ix, iy] + self.bz_e[ix, iy] * diff_Ez_dot) / self.byp[ix, iy]
        
        src = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        self.Ez[cfg.x0, cfg.y0] -= cfg.dx[cfg.x0] * cfg.dy[cfg.y0] * src / self.coef_p[cfg.x0-1, cfg.y0-1]


class FCISolver:
    def __init__(self, config): #Nx, Ny, Nt, lambda0, CFL, bc='PEC', solver="Schur", finesse=20):
        self.cfg = config


        # PBC drops the redundant N+1 boundary node to form a perfect ring
        if self.cfg.fci_bc == 'PBC':
            self.nx_n = self.cfg.nx
            self.ny_n = self.cfg.ny
            self.cfg.dx = np.concatenate(([self.cfg.dx_0], self.cfg.dx, [self.cfg.dx_0]))
            self.cfg.dy = np.concatenate(([self.cfg.dy_0], self.cfg.dy, [self.cfg.dy_0]))
            self.cfg.dx = np.concatenate(([self.cfg.dx_0], self.cfg.dx, [self.cfg.dx_0]))
            self.cfg.dy = np.concatenate(([self.cfg.dy_0], self.cfg.dy, [self.cfg.dy_0]))
        else:
            self.nx_n = self.cfg.nx + 1
            self.ny_n = self.cfg.ny + 1
            self.cfg.dx = np.concatenate(([self.cfg.dx_0], self.cfg.dx, [self.cfg.dx_0]))
            self.cfg.dy = np.concatenate(([self.cfg.dy_0], self.cfg.dy, [self.cfg.dy_0]))
  
        

        self.cfg.sigma = np.zeros((self.nx_n, self.ny_n)) #Define Drude media (so sigma_DC)
        self.cfg.gamma = 0.0
        self.cfg.epsilon_r  = np.ones((self.nx_n, self.ny_n))

        self.cfg.v_local = self.cfg.c / np.sqrt(self.cfg.epsilon_r)
        self.cfg.Z_local = self.cfg.Z0 / np.sqrt(self.cfg.epsilon_r)

        self.len_hx = self.nx_n * self.cfg.ny
        self.len_hy = self.cfg.nx * self.ny_n
        self.len_ez = self.nx_n * self.ny_n
        
        # Hx & Hx_dot + Hy & Hy_dot + Ez & Ez_dot & Ez_ddot & Jz:
        self.total_len = 2 * self.len_hx + 2 * self.len_hy + 4 * self.len_ez
        ez_start = 2 * self.len_hx + 2 * self.len_hy
        self.idx_ez = slice(ez_start, ez_start + self.len_ez)
        
        self.u = np.zeros(self.total_len)
        self._build_system()

    def _get_pml_profiles(self):
        self.kx = np.ones((self.nx_n,self.ny_n))
        self.ky = np.ones((self.nx_n,self.ny_n))
        self.sx = np.zeros((self.nx_n,self.ny_n))
        self.sy = np.zeros((self.nx_n,self.ny_n))

        p, m = int(2.0 * self.cfg.finesse), 4
        k_max = 1.0
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
        if self.cfg.fci_bc == 'PBC':
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
            Dtx = sp.diags(1/(2*d), offsets=0, shape=(n, n), format='csr') @ (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1))
            Dtx[n, n-1] = - 1 / d[-1]

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
        L87 = to_diag_vec(-self.cfg.sigma.flatten());   L88 = to_diag_scal(2.0*gamma/dt + 1.0, self.len_ez)

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
        R87 = to_diag_vec(self.cfg.sigma.flatten());    R88 = to_diag_scal(2.0*gamma/dt - 1.0, self.len_ez)

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

        if self.cfg.fci_solver == "default":
            print("Pre-factoring the 8x8 system...")
            self.solve_func = spla.factorized(self.LHS.tocsc())

        elif self.cfg.fci_solver == "Schur":
            print("Pre-factoring the 8x8 system using Schur complement...")

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

            def solve_schur(b):
                b1 = b[:2*self.len_hx + 2*self.len_hy]
                b2 = b[2*self.len_hx + 2*self.len_hy:]

                u2 = S_fact(b2 - M21 @ M11_inv @ b1)
                u1 = M11_inv @ (b1 - M12 @ u2)

                return np.concatenate([u1, u2])

            self.solve_func = solve_schur

    def step(self, t):
        # 1. 
        b = self.RHS.dot(self.u)
        self.u = self.solve_func(b)
        self.Ez = self.u[self.idx_ez].reshape((self.nx_n, self.ny_n))
        
        # 2. 
        src_val = self.cfg.A * np.cos(2 * np.pi * self.cfg.f_c * (t - self.cfg.t0)) * \
                  np.exp(-0.5 * ((t - self.cfg.t0) / self.cfg.sig_t)**2)
                  
        # 3. Match YEE's exact amplitude scaling
        v = self.cfg.v_local[self.cfg.x0, self.cfg.y0]
        dx = self.cfg.dx[self.cfg.x0]
        dy = self.cfg.dy[self.cfg.y0]
        coef_p = 1.0 / (v * self.cfg.dt) + self.cfg.Z_local[self.cfg.x0, self.cfg.y0] * self.cfg.sigma[self.cfg.x0, self.cfg.y0] / (2.0 * self.ap)
        
        scaled_src = src_val * (dx * dy) / coef_p
        
        # 4. Spatially distribute
        weights = [
            (-1, -1, 1/16), (0, -1, 1/8), (1, -1, 1/16),
            (-1,  0, 1/8),  (0,  0, 1/4), (1,  0, 1/8),
            (-1,  1, 1/16), (0,  1, 1/8), (1,  1, 1/16)
        ]
        
        for i, j, w in weights:
            self.Ez[self.cfg.x0 + i, self.cfg.y0 + j] -= scaled_src * w
            
        self.u[self.idx_ez] = self.Ez.flatten()

# ==============================================================================
# 3. Simulation Runner
# ==============================================================================
class SimulationRunner:
    @staticmethod
    def execute(frame_skip=3, **kwargs):
        config = SimulationConfig(**kwargs)
        if config.solver_type == "fci":
            print("Initializing FCI Solver (Schur/Sparse)...")
            solver = FCISolver(config)
        else:
            print("Initializing standard Yee Solver...")
            solver = YeeSolver(config)
            
        n_frames = int(np.ceil(config.nt / frame_skip))
        field_history = np.zeros((n_frames, config.nx, config.ny), dtype=np.float32)
        recorder_plane = np.zeros((n_frames, config.ny), dtype=np.float32)
        recorder_full = np.zeros((config.nt, config.ny), dtype=np.float32)
        
        rec_diag = np.zeros(config.nt, dtype=np.float32)
        rec_after = np.zeros(config.nt, dtype=np.float32)
        rec_before = np.zeros(config.nt, dtype=np.float32)
        
        frame_idx = 0
        for it in tqdm(range(config.nt), desc=f"Simulating (nt={config.nt}, solver={config.solver_type})"):
            t = it * config.dt
            solver.step(t)
            
            recorder_full[it, :] = solver.Ez[config.x1, :]
            rec_diag[it] = solver.Ez[config.x_diag, config.y_diag]
            rec_after[it] = solver.Ez[config.x_after, config.y_after]
            rec_before[it] = solver.Ez[config.x_before, config.y_before]
            
            if it % frame_skip == 0 and frame_idx < n_frames:
                field_history[frame_idx] = solver.Ez
                recorder_plane[frame_idx, :] = solver.Ez[config.x1, :]
                frame_idx += 1
            
        return {
            "config": config,
            "history": field_history,
            "recorder": recorder_plane,
            "recorder_full": recorder_full,
            "rec_diag": rec_diag,
            "rec_after": rec_after,
            "rec_before": rec_before,
            "frame_skip": frame_skip,
            "times": np.arange(config.nt) * config.dt,
            "z_norm": getattr(solver, 'Z_local', config.Z_local) # Fetch Z_local safely depending on solver setup
        }

# ==============================================================================
# 4. Simulation Analyzer
# ==============================================================================
class SimulationAnalyzer:
    @staticmethod
    def verify_with_hankel(results):
        print("\n--- Plotting Hankel verification ---")
        cfg = results["config"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))[:cfg.nx]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny]
        t = np.arange(cfg.nt) * cfg.dt
        n_pad = 2**int(np.ceil(np.log2(cfg.nt * 8)))
        freqs = fftfreq(n_pad, cfg.dt)
        f_min = getattr(cfg, "hankel_f_min", cfg.f_c * 0.2)
        f_max = getattr(cfg, "hankel_f_max", cfg.f_c * 3.0)
        band_idx = np.where((freqs > f_min) & (freqs < f_max))[0]
        f_valid = freqs[band_idx]
        omega = 2 * np.pi * f_valid
        k0 = omega / cfg.c
        
        src_time = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        J_src_f = fft(src_time, n=n_pad) * cfg.dt
        J_src_valid = J_src_f[band_idx]

        results["_hankel_data"] = {
            "f_valid": f_valid, "omega": omega, "k0": k0, "J_src_valid": J_src_valid,
            "obs_points": {}
        }
        
        def process_point(name, ez_data, x_idx, y_idx):
            dx_m = nodes_x[x_idx] - nodes_x[cfg.x0]
            dy_m = nodes_y[y_idx] - nodes_y[cfg.y0]
            r = np.sqrt(dx_m**2 + dy_m**2)
            if r == 0: return
            
            Ez_sim_f = fft(ez_data, n=n_pad) * cfg.dt
            Ez_sim_valid = Ez_sim_f[band_idx]
            H_sim = Ez_sim_valid / J_src_valid
            H_sim_corrected = H_sim * np.exp(1j * omega * cfg.dt)
            H_analytical = -(omega * cfg.mu0 / 4) * sp_special.hankel2(0, k0 * r)
            results["_hankel_data"]["obs_points"][name] = {
                "r": r, "H_sim": H_sim_corrected, "H_analytical": H_analytical
            }

        process_point("Recorder Full", results["recorder_full"][:, cfg.y1], cfg.x1, cfg.y1)
        process_point("Obs Diag", results["rec_diag"], cfg.x_diag, cfg.y_diag)
        process_point("Obs After", results["rec_after"], cfg.x_after, cfg.y_after)
        process_point("Obs Before", results["rec_before"], cfg.x_before, cfg.y_before)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
        
        ax = axes[0]
        ax.plot(f_valid, np.abs(J_src_valid) / np.abs(J_src_valid[idx_fc]), label='Source', lw=2, color='lightgray')
        
        if "Recorder Full" in results["_hankel_data"]["obs_points"]:
            data_main = results["_hankel_data"]["obs_points"]["Recorder Full"]
            H_sim_c = data_main["H_sim"]
            H_anal = data_main["H_analytical"]
            norm_sim = np.abs(H_sim_c[idx_fc])
            norm_anal = np.abs(H_anal[idx_fc])
            
            ax.plot(f_valid, np.abs(H_sim_c) / norm_sim, label='Simulation', lw=2)
            ax.plot(f_valid, np.abs(H_anal) / norm_anal, '--', label='Analytical', lw=2)
            max_val = np.max(np.abs(H_sim_c[f_valid < cfg.f_break]) / norm_sim)
            
            axes[1].plot(f_valid, np.unwrap(np.angle(H_sim_c)), label='Simulation', lw=2)
            axes[1].plot(f_valid, np.unwrap(np.angle(H_anal)), '--', label='Analytical', lw=2)
            ax.set_ylim(0, 1.3 * max_val)

        ax.axvline(cfg.f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({cfg.f_break:.2e} Hz)') 
        ax.axvline(cfg.f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({cfg.f_min:.2e} Hz)') 
        ax.set_xscale('log')
        ax.set_title(f'Magnitude Response [{cfg.solver_type.upper()}]')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        axes[1].set_xscale('log')
        axes[1].set_title(f'Phase Response [{cfg.solver_type.upper()}]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (rad)')
        axes[1].legend()
        axes[1].grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_error_analysis(results):
        if "_hankel_data" not in results: return
        hd = results["_hankel_data"]
        cfg = results["config"]
        f_valid = hd["f_valid"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (name, data) in enumerate(hd["obs_points"].items()):
            H_sim, H_anal = data["H_sim"], data["H_analytical"]
            sim_mag = np.abs(H_sim) / np.abs(H_sim[idx_fc])
            anal_mag = np.abs(H_anal) / np.abs(H_anal[idx_fc])
            rel_error = np.abs(sim_mag - anal_mag) / anal_mag
            phase_diff = np.unwrap(np.angle(H_sim)) - np.unwrap(np.angle(H_anal))
            phase_diff -= phase_diff[idx_fc]
            phase_error_deg = np.rad2deg(phase_diff)
            
            axes[0].plot(f_valid, rel_error * 100, label=name, lw=1.5, color=colors[i])
            axes[1].plot(f_valid, phase_error_deg, label=name, lw=1.5, color=colors[i])
            
        for ax in axes:
            ax.axvline(cfg.f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({cfg.f_break:.2e} Hz)')
            ax.axvline(cfg.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({cfg.f_c:.2e} Hz)')
            ax.axvline(cfg.f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({cfg.f_min:.2e} Hz)') 
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        axes[0].set_title(f'Relative Magnitude Error [{cfg.solver_type.upper()}]')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Relative Error (%)')
        
        axes[1].set_title(f'Phase Error Dispersion [{cfg.solver_type.upper()}]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase Error (Degrees)')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_2d_animation(results, fps=40):
        cfg = results["config"]
        hist = results["history"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))[:cfg.nx]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny]
        X, Y = np.meshgrid(nodes_x, nodes_y)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        
        z_norm = np.array(results["z_norm"]) # Using fetched z_norm
        vmax = np.max(np.abs(hist * z_norm)) * 0.8
        if vmax == 0:
            vmax = 1e-4
        quad = ax.pcolormesh(X, Y, (hist[0] * z_norm).T, shading='nearest', cmap='RdBu_r', vmin=-vmax, vmax=vmax, zorder=1)
                             
        scat1 = ax.scatter(nodes_x[cfg.x0], nodes_y[cfg.y0], color='red', s=20, zorder=2, label='Source')
        scat2 = ax.scatter(nodes_x[cfg.x1], nodes_y[cfg.y1], color='green', s=20, zorder=2, label='Recorder')
        scat3 = ax.scatter(nodes_x[cfg.x_diag], nodes_y[cfg.y_diag], color='cyan', s=30, zorder=2, marker='x', label='Obs Diag')
        scat4 = ax.scatter(nodes_x[cfg.x_after], nodes_y[cfg.y_after], color='magenta', s=30, zorder=2, marker='x', label='Obs After')
        scat5 = ax.scatter(nodes_x[cfg.x_before], nodes_y[cfg.y_before], color='yellow', s=30, zorder=2, marker='x', label='Obs Before')
        
        ax.legend(loc='upper right', fontsize=8)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontweight='bold')
        fs = results["frame_skip"]
        
        def update(i):
            frame_data = (hist[i] * z_norm).T
            quad.set_array(frame_data.ravel())
            time_text.set_text(f'Frame {i*fs}/{cfg.nt} | t = {i*fs*cfg.dt*1e9:.3f} ns')
            return quad, time_text, scat1, scat2, scat3, scat4, scat5

        interval_ms = max(1, int(1000 / fps))
        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval_ms, blit=True)
        results["ani_2d"] = ani
        ax.set_title(f"Field Animation ({cfg.solver_type.upper()})")
        plt.show()

    @staticmethod
    def compare_recorders(*results_list):
        fig, ax = plt.subplots(figsize=(10, 4))
        for res in results_list:
            cfg = res["config"]
            t = res["times"]
            ez = res["recorder_full"][:, cfg.y1]
            ax.plot(t * 1e9, ez, label=cfg.solver_type.upper(), lw=1.5)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Ez Amplitude")
        ax.set_title("Recorder Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ==============================================================================
# Execution Example
# ==============================================================================
if __name__ == "__main__":
    t0 = time.time()
    res_fci = SimulationRunner.execute(
        solver_type = "fci",
        frame_skip = 10,
        finesse = 9,
        free_space_sim = True,
        grid_refinement = False,
        do_hankel = True,
    )
    t1 = time.time()
    print(f"FCI executed in {t1-t0:.2f} seconds.")

    # SimulationAnalyzer.verify_with_hankel(res_fci)
    # SimulationAnalyzer.plot_error_analysis(res_fci)
    SimulationAnalyzer.plot_2d_animation(res_fci)

    t0 = time.time()
    res_yee = SimulationRunner.execute(
        solver_type="yee",
        frame_skip=10,
        finesse=9,
        free_space_sim=True,
        do_hankel=True,
    )
    t1 = time.time()
    print(f"YEE executed in {t1-t0:.2f} seconds.")

    # SimulationAnalyzer.verify_with_hankel(res_yee)
    # SimulationAnalyzer.plot_error_analysis(res_yee)
    SimulationAnalyzer.plot_2d_animation(res_yee)

    SimulationAnalyzer.compare_recorders(res_fci, res_yee)

