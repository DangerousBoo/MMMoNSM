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
        self.L_wg   = f * 6 * self.lam_c
        self.w_core = f * 1 * self.lam_c
        self.w_clad = f * 1 * self.lam_c
        self.w_air  = f * 1 * self.lam_c
        self.d      = f * 4 * self.lam_c
        self.t_m    = f * 0.03 * self.lam_c
        self.Ll     = f * 2 * self.lam_c
        self.Lr     = f * 1 * self.lam_c
        self.L      = self.Ll + self.d + self.t_m + self.L_wg + self.Lr
        self.W      = 2 * self.w_air + 2 * self.w_clad + self.w_core
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
        
        print(self.ny)
        
        self.epsilon_r  = np.ones((self.nx, self.ny))
        self.sigma      = np.zeros((self.nx-2, self.ny-2))
        self.gamma      = np.zeros((self.nx-2, self.ny-2))

        self.dx_f = self.dx.min()
        self.dy_f = self.dy.min()
        
        self.x0 = getattr(self, "x0", self.n_Ll)
        self.y0 = getattr(self, "y0", self.ny // 2)
        
        self.A *= (self.dx[self.x0] * self.dy[self.y0])**(-2)
        
        sum_dx = np.cumsum(self.dx)
        
        self.x1 = self.x0 + kwargs["x1"] if "x1" in kwargs else np.argmin(np.abs(sum_dx - (self.Ll + self.d + self.t_m + self.L_wg + 0.5*self.Lr)))
        self.y1 = self.y0 + kwargs["y1"] if "y1" in kwargs else self.y0

        dist_req = self.d / 2.0
        offset_x = np.searchsorted(np.cumsum(self.dx[self.x0:]), dist_req)
        offset_y = np.searchsorted(np.cumsum(self.dy[self.y0:]), dist_req)
        
        
        
        self.x_diag = getattr(self, "x_diag", self.x0 + offset_x)
        self.y_diag = getattr(self, "y_diag", self.y0 + offset_y)

        self.x_after = getattr(self, "x_after", np.argmin(np.abs(sum_dx - (self.Ll + self.d + self.t_m + self.L_wg))))
        self.y_after = getattr(self, "y_after", self.y0)
        self.x_before = getattr(self, "x_before", np.argmin(np.abs(sum_dx - (self.Ll+ self.d))))
        self.y_before = getattr(self, "y_before", self.y0)

        self.dx_d = np.concatenate(([self.dx[0]/2], (self.dx[:-1] + self.dx[1:])/2, [self.dx[-1]/2]))
        self.dy_d = np.concatenate(([self.dy[0]/2], (self.dy[:-1] + self.dy[1:])/2, [self.dy[-1]/2]))
        
        if self.solver_type == "fci":
            CFL_default = 1
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
        self.deps_max = getattr(self, "deps_max", 0.1) # percentage of (self.eps_core - self.eps_clad)
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
                d_clad = self.dy_0 / np.sqrt(self.eps_clad)
                if self.eps_core != self.eps_clad:
                    self.a_eps = 2 * np.sqrt(self.eps_core) * (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) / (self.w_core / 2) ** 2
                    self.b_eps = (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) ** 2 / (self.w_core / 2) ** 4
                    base_dy_core = self.dy_0 / np.sqrt(self.eps_core)
                    delta_eps = np.abs(self.eps_core - self.eps_clad)
                    x_edge = self.w_core / 2
                    max_grad = np.abs(2 * self.a_eps * x_edge + 4 * self.b_eps * x_edge**3)
                    if max_grad > 0:
                        dy_est = self.deps_max * delta_eps / max_grad
                    else:
                        dy_est = base_dy_core
                    # Safety floor prevents pathological core cell counts for extreme geometries.
                    self.dy_core = min(dy_est, base_dy_core)
                else:
                    self.dy_core = d_clad

                self.L_f_ac, self.n_f_ac = self._L_and_n_fine(self.dy_0, d_clad, self.alpha, max_length=self.w_air)
                air_taper = self.alpha ** np.arange(self.n_f_ac, 0, -1) * d_clad
                rem_air = max(self.w_air - self.L_f_ac, 0.0)
                n_air_coarse = int(np.ceil(rem_air / self.dy_0)) if rem_air > 0 else 0
                self.n_air = n_air_coarse + self.n_f_ac
                air_coarse = np.full(n_air_coarse, rem_air / n_air_coarse) if n_air_coarse > 0 else np.array([], dtype=float)
                
                self.L_f_cc, self.n_f_cc = self._L_and_n_fine(self.dy_0 / np.sqrt(self.eps_clad), self.dy_core, self.alpha, max_length=self.w_clad)
                clad_taper = self.alpha ** np.arange(self.n_f_cc, 0, -1) * self.dy_core
                rem_clad = max(self.w_clad - self.L_f_cc, 0.0)
                n_clad_coarse = int(np.ceil(rem_clad / d_clad)) if rem_clad > 0 else 0
                self.n_clad = n_clad_coarse + self.n_f_cc
                self.n_core = max(1, int(np.ceil(self.w_core / self.dy_core)))

                clad_coarse = np.full(n_clad_coarse, rem_clad / n_clad_coarse) if n_clad_coarse > 0 else np.array([], dtype=float)
                core_cells = np.full(self.n_core, self.w_core / self.n_core)
                
                dy_mid = [
                    clad_coarse,
                    clad_taper,
                    core_cells,
                    clad_taper[::-1],
                    clad_coarse
                ]
                
            self.dy = np.concatenate([
                air_coarse if self.wg_type != "step" else np.full(int(self.n_air - self.n_f_ac), (self.w_air - self.L_f_ac) / (self.n_air - self.n_f_ac)),
                air_taper if self.wg_type != "step" else self.alpha ** np.arange(self.n_f_ac, 0, -1) * self.dy_0 / np.sqrt(self.eps_clad),
                *dy_mid,
                air_taper[::-1] if self.wg_type != "step" else self.alpha ** np.arange(1, self.n_f_ac + 1) * self.dy_0 / np.sqrt(self.eps_clad),
                air_coarse if self.wg_type != "step" else np.full(int(self.n_air - self.n_f_ac), (self.w_air - self.L_f_ac) / (self.n_air - self.n_f_ac))
            ])

        if self.grid_refinement == "step":
            self.n_air = int(np.ceil(self.w_air / self.dy_0))
            self.n_clad = int(np.ceil(self.w_clad / self.dy_0 * np.sqrt(self.eps_clad)))

            if self.wg_type == "step":
                self.n_core = int(np.ceil(self.w_core / self.dy_0 * np.sqrt(self.eps_core)))
            else:
                if self.eps_core != self.eps_clad:
                    self.a_eps = 2 * np.sqrt(self.eps_core) * (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) / (self.w_core / 2) ** 2
                    self.b_eps = (np.sqrt(self.eps_clad) - np.sqrt(self.eps_core)) ** 2 / (self.w_core / 2) ** 4
                    base_dy_core = self.dy_0 / np.sqrt(self.eps_core)
                    delta_eps = np.abs(self.eps_core - self.eps_clad)
                    x_edge = self.w_core / 2
                    max_grad = np.abs(2 * self.a_eps * x_edge + 4 * self.b_eps * x_edge**3)
                    if max_grad > 0:
                        dy_est = self.deps_max * delta_eps / max_grad
                    else:
                        dy_est = base_dy_core
                    self.dy_core = min(dy_est, base_dy_core)
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
        
    def _L_and_n_fine(self, d_coarse, d_fine, alpha=np.sqrt(2), max_length=None):
        if self.eps_clad == self.eps_core:
            return 0, 0
        n_f = int(np.ceil(np.log(d_coarse / d_fine) / np.log(alpha))) - 1
        n_f = max(0, n_f)

        if alpha == 1:
            L_f = d_fine * n_f
        else:
            L_f = d_fine * alpha * (alpha**n_f - 1) / (alpha - 1)

        if max_length is not None:
            while n_f > 0 and L_f > max_length:
                n_f -= 1
                if alpha == 1:
                    L_f = d_fine * n_f
                else:
                    L_f = d_fine * alpha * (alpha**n_f - 1) / (alpha - 1)

            if n_f == 0:
                L_f = 0.0

        # The tapered section uses cell lengths d_fine * alpha^k for k = 1..n_f.
        # Sum that geometric series so the coarse remainder fits the intended width.
        return L_f, n_f
    
    def _setup_waveguide(self):
        if self.free_space_sim:
            return
        self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air):int(-self.n_air)] = self.eps_clad
        if self.wg_type == "step":
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = self.eps_core
        else:
            core_slice = (slice(int(self.n_Ll + self.n_d + 1), int(self.n_Ll + self.n_d + 1 + self.n_wg)),
                          slice(int(self.n_air + self.n_clad), int(- (self.n_air + self.n_clad))))
            if hasattr(self, "a_eps") and hasattr(self, "b_eps"):
                eps_val = lambda y: self.eps_core + self.a_eps * (y - self.W / 2)**2 + self.b_eps * (y - self.W / 2)**4
                y_nodes = np.concatenate(([0], np.cumsum(self.dy)))
                core_start = int(self.n_air + self.n_clad)
                core_stop = int(self.n_air + self.n_clad + self.n_core) + 1
                self.epsilon_r[core_slice] = eps_val(y_nodes[core_start:core_stop])
            else:
                self.epsilon_r[core_slice] = self.eps_core
            
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


        # PBC: N nodes form a closed ring; no extra boundary node needed
        if self.cfg.fci_bc == 'PBC':
            self.nx_n = self.cfg.nx
            self.ny_n = self.cfg.ny
        else:
            self.nx_n = self.cfg.nx + 1
            self.ny_n = self.cfg.ny + 1
  
        

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
        k_max = 2.0
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
            # Forward-difference operator scaled by local spacing; Ax averages node i with i+1 (mod n)
            Dx = sp.diags_array(1/d) @ sp.diags_array([-ones, ones, ones], offsets=[0, 1, 1-n], shape=(n, n), format='csr')
            Ax = sp.diags_array([ones, ones, ones], offsets=[0, 1, 1-n], shape=(n, n), format='csr')
            return Ix, Dx, Ax
        else:
            Ix = sp.eye(n + 1, format='csr')
            Dx  = sp.diags(1/(2*d), offsets=0, shape=(n, n), format='csr') @ (sp.eye(n, n+1, k=1) - sp.eye(n, n+1, k=0))
            Dtx = sp.diags(1/(2*d), offsets=0, shape=(n, n), format='csr') @ (sp.eye(n+1, n, k=0) - sp.eye(n+1, n, k=-1))
            Dtx = Dtx.tolil(); Dtx[n, n-1] = -1 / d[-1]; Dtx = Dtx.tocsr()
            A1  = sp.diags_array([1, 1], offsets=[0, -1], shape=(n, n-1), format='csr')
            A2  = sp.diags_array([1, 1], offsets=[0,  1], shape=(n, n+1), format='csr')
            return Ix, Dx.tocsr(), Dtx, A1.tocsr(), A2.tocsr()

    def _build_system(self):
        v     = self.cfg.v_local.flatten()
        Z     = self.cfg.Z_local.flatten()
        gamma = self.cfg.gamma
        dt    = self.cfg.dt
        sx, sy, kx, ky = self._get_pml_profiles()
        sx = sx.flatten(); sy = sy.flatten()
        kx = kx.flatten(); ky = ky.flatten()

        Ix, Dx, Ax = self._get_operators(self.nx_n, self.cfg.dx)
        Iy, Dy, Ay = self._get_operators(self.ny_n, self.cfg.dy)

        bxp = kx / (v * dt) + Z * sx / 2.0
        byp = ky / (v * dt) + Z * sy / 2.0
        bxm = kx / (v * dt) - Z * sx / 2.0
        bym = ky / (v * dt) - Z * sy / 2.0
        bz  = 1.0 / (v * dt)
        ap  = np.full(self.len_ez, 2.0 * gamma / dt + 1.0)
        am  = np.full(self.len_ez, 2.0 * gamma / dt - 1.0)
        sigma = self.cfg.sigma.flatten()

        def diag(vec):
            return sp.diags_array(vec, format='csc')

        # L blocks: kron(Ax, Dy) computes the curl using local-spacing derivatives
        # and area-interpolation (Ax/Ay), replacing the old transpose-diff + interp split
        L11 = diag(bz);                                  L12 = diag(-bxp)
        L22 = sp.kron(Ix, Ay) @ diag(byp);               L25 = sp.kron(Ix, Dy) @ diag(1/Z)
        L33 = diag(bxp);                                 L34 = diag(-byp)
        L44 = sp.kron(Ax, Iy) @ diag(bz);                L45 = sp.kron(Dx, Iy) @ diag(-1/Z)
        L55 = diag(byp);                                 L56 = diag(-bz)
        L66 = diag(bxp);                                 L67 = diag(-bz)
        L71 = sp.kron(Ax, Dy); L73 = -sp.kron(Dx, Ay);  L77 = sp.kron(Ax, Ay) @ diag(1/(Z * v * dt)); L78 = sp.kron(Ax, Ay) / 2
        L87 = diag(-sigma);    L88 = diag(ap)

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

        R11 = diag(bz);                                  R12 = diag(-bxm)
        R22 = sp.kron(Ix, Ay) @ diag(bym);               R25 = -sp.kron(Ix, Dy) @ diag(1/Z)
        R33 = diag(bxm);                                 R34 = diag(-bym)
        R44 = sp.kron(Ax, Iy) @ diag(bz);                R45 = sp.kron(Dx, Iy) @ diag(1/Z)
        R55 = diag(bym);                                 R56 = diag(-bz)
        R66 = diag(bxm);                                 R67 = diag(-bz)
        R71 = -sp.kron(Ax, Dy); R73 = sp.kron(Dx, Ay);  R77 = sp.kron(Ax, Ay) @ diag(1/(Z * v * dt)); R78 = -sp.kron(Ax, Ay) / 2
        R87 = diag(sigma);      R88 = diag(am)

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

        if not self.cfg.schur:
            print("Pre-factoring system (full LU)...")
            self.solve_func = spla.factorized(self.LHS.tocsc())
        else:
            print("Pre-factoring system (Schur complement)...")
            # Get blocks
            
            M12 = self.LHS[2*self.len_hx + 2*self.len_hy:, :2*self.len_hx + 2*self.len_hy]
            M21 = self.LHS[:2*self.len_hx + 2*self.len_hy, 2*self.len_hx + 2*self.len_hy:]
            M22 = self.LHS[2*self.len_hx + 2*self.len_hy:, 2*self.len_hx + 2*self.len_hy:]
            
            # Compute inverse of M11
            L11_inv = diag(1.0/L11.diagonal())
            L22_inv = spla.spsolve(L22.tocsc(), sp.eye(L22.shape[0], format='csc'), permc_spec='COLAMD')
            L33_inv = diag(1.0/L33.diagonal())
            L44_inv = spla.spsolve(L44.tocsc(), sp.eye(L44.shape[0], format='csc'), permc_spec='COLAMD')

            M11_inv = sp.bmat([
                [L11_inv,  -L11_inv @ L12 @ L22_inv,  None,       None],
                [None,      L22_inv,                  None,       None],
                [None,      None,                     L33_inv,  -L33_inv @ L34 @ L44_inv],
                [None,      None,                     None,       L44_inv]
            ], format='csc')

            # Compute inverse of Schur complement using Schur 
            S_12 = M22[:2*self.len_ez, 2*self.len_ez:]
            S_21 = sp.bmat([
                [-L71 @ L11_inv @ L12 @ L22_inv @ L25 - L73 @ L33_inv @ L34 @ L44_inv @ L45, None],
                [sp.eye(self.len_ez, format='csc') * 0, None]
            ], format='csc')
            S_22 = M22[2*self.len_ez:, 2*self.len_ez:]

            # invert S_11 =  [L55, L56],
            #                [0,   L66]
            L55_inv = diag(1.0/L55.diagonal())
            L66_inv = diag(1.0/L66.diagonal())

            S_11_inv = sp.bmat([
                [L55_inv, L55_inv @ L56 @ L66_inv],
                [None, L66_inv]
            ], format='csc')

            # Do schur once more
            L88_inv = diag(1.0/L88.diagonal())

            S2 = L77 - S_12[:self.len_ez, :self.len_ez] @ L55_inv @ L56 @ L66_inv @ L67 - L78 @ L88_inv @ L87

            S2_inv = spla.spsolve(S2.tocsc(), sp.eye(S2.shape[0], format='csc'), permc_spec='COLAMD')

            S1_inv = sp.bmat([
                [S2_inv, -S2_inv @ L78 @ L88_inv],
                [-L88_inv @ L87 @ S2_inv, L88_inv + L88_inv @ L87 @ S2_inv @ L78 @ L88_inv]
            ], format='csc')
            
            S_inv = sp.bmat([
                [S_11_inv + S_11_inv @ S_12 @ S1_inv @ S_21 @ S_11_inv, -S_11_inv @ S_12 @ S1_inv],
                [-S1_inv @ S_21 @ S_11_inv, S1_inv]
            ], format='csc')


            def schur_solve(b):
                b1 = b[:2*self.len_hx + 2*self.len_hy]
                b2 = b[2*self.len_hx + 2*self.len_hy:]

                u2 = S_inv @ (b2 - M21 @ M11_inv @ b1)
                u1 = M11_inv @ (b1 - M12 @ u2)
                u = np.concatenate([u1, u2])

                return u
            
            self.solve_func = schur_solve
        

    def step(self, t):
        b = self.RHS.dot(self.u)

        src_val = self.cfg.A * np.cos(2 * np.pi * self.cfg.f_c * (t - self.cfg.t0)) * \
                  np.exp(-0.5 * ((t - self.cfg.t0) / self.cfg.sig_t)**2)

        dx = self.cfg.dx[self.cfg.x0]
        dy = self.cfg.dy[self.cfg.y0]
        src_idx = self.idx_ez.start + self.cfg.x0 * self.ny_n + self.cfg.y0
        b[src_idx] -= src_val * dx * dy

        self.u = self.solve_func(b)
        self.Ez = self.u[self.idx_ez].reshape((self.nx_n, self.ny_n))

# ==============================================================================
# 3. Simulation Runner
# ==============================================================================
class SimulationRunner:
    @staticmethod
    def execute(frame_skip=3, **kwargs):
        config = SimulationConfig(**kwargs)
        if config.solver_type == "fci":
            print("Initializing FCI Solver...")
            solver = FCISolver(config)
        else:
            print("Initializing standard Yee Solver...")
            solver = YeeSolver(config)
        
        recorders = getattr(config, "recorders", ["all"])
            
        n_frames = int(np.ceil(config.nt / frame_skip))
        field_history = np.zeros((n_frames, config.nx, config.ny), dtype=np.float32)
        recorder_plane = np.zeros((n_frames, config.ny), dtype=np.float32)
        recorder_full = np.zeros((config.nt, config.ny), dtype=np.float32)
        
        if "all" in recorders or "diag" in recorders:
            rec_diag = np.zeros(config.nt, dtype=np.float32)
        if "all" in recorders or "after" in recorders:
            rec_after = np.zeros(config.nt, dtype=np.float32)
        if "all" in recorders or "before" in recorders:
            rec_before = np.zeros(config.nt, dtype=np.float32)
        
        frame_idx = 0
        for it in tqdm(range(config.nt), desc=f"Simulating (nt={config.nt}, solver={config.solver_type})"):
            t = it * config.dt
            solver.step(t)
            
            recorder_full[it, :] = solver.Ez[config.x1, :]
            
            if "all" in recorders or "diag" in recorders:
                rec_diag[it] = solver.Ez[config.x_diag, config.y_diag]
            if "all" in recorders or "after" in recorders:
                rec_after[it] = solver.Ez[config.x_after, config.y_after]
            if "all" in recorders or "before" in recorders:
                rec_before[it] = solver.Ez[config.x_before, config.y_before]
            
            if it % frame_skip == 0 and frame_idx < n_frames:
                field_history[frame_idx] = solver.Ez
                recorder_plane[frame_idx, :] = solver.Ez[config.x1, :]
                frame_idx += 1
        result = {
                "config": config,
                "history": field_history,
                "recorder": recorder_plane,
                "recorder_full": recorder_full,
                "frame_skip": frame_skip,
                "times": np.arange(config.nt) * config.dt,
                "z_norm": getattr(solver, 'Z_local', config.Z_local)
            }

        if "all" in recorders or "diag" in recorders:
            result["rec_diag"] = rec_diag

        if "all" in recorders or "after" in recorders:
            result["rec_after"] = rec_after

        if "all" in recorders or "before" in recorders:
            result["rec_before"] = rec_before

        return result

# ==============================================================================
# 4. Simulation Analyzer
# ==============================================================================
class SimulationAnalyzer:
    @staticmethod
    def _compute_hankel_data(results):
        if "_hankel_data" in results:
            return results["_hankel_data"]

        cfg = results["config"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))[:cfg.nx]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny]
        t = np.arange(cfg.nt) * cfg.dt
        n_pad = 2**int(np.ceil(np.log2(cfg.nt * 8)))
        freqs = fftfreq(n_pad, cfg.dt)
        f_min = getattr(cfg, "hankel_f_min", cfg.f_c * 0)
        f_max = getattr(cfg, "hankel_f_max", cfg.f_c * 3.0)
        band_idx = np.where((freqs > f_min) & (freqs < f_max))[0]
        f_valid = freqs[band_idx]
        omega = 2 * np.pi * f_valid
        k0 = omega / cfg.c

        src_time = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        J_src_f = fft(src_time, n=n_pad) * cfg.dt
        J_src_valid = J_src_f[band_idx]

        results["_hankel_data"] = {
            "f_valid": f_valid,
            "omega": omega,
            "k0": k0,
            "J_src_valid": J_src_valid,
            "obs_points": {},
        }

        def process_point(name, ez_data, x_idx, y_idx):
            dx_m = nodes_x[x_idx] - nodes_x[cfg.x0]
            dy_m = nodes_y[y_idx] - nodes_y[cfg.y0]
            r = np.sqrt(dx_m**2 + dy_m**2)
            if r == 0:
                return

            Ez_sim_f = fft(ez_data, n=n_pad) * cfg.dt
            Ez_sim_valid = Ez_sim_f[band_idx]
            H_sim = Ez_sim_valid / J_src_valid
            H_sim_corrected = H_sim * np.exp(1j * omega * cfg.dt)
            H_analytical = -(omega * cfg.mu0 / 4) * sp_special.hankel2(0, k0 * r)
            results["_hankel_data"]["obs_points"][name] = {
                "r": r,
                "H_sim": H_sim_corrected,
                "H_analytical": H_analytical,
            }

        recorders = getattr(cfg, "recorders", ["all"])

        process_point("Recorder Full", results["recorder_full"][:, cfg.y1], cfg.x1, cfg.y1)
        if "all" in recorders or "diag" in recorders:
            process_point("Obs Diag", results["rec_diag"], cfg.x_diag, cfg.y_diag)
        if "all" in recorders or "after" in recorders:
            process_point("Obs After", results["rec_after"], cfg.x_after, cfg.y_after)
        if "all" in recorders or "before" in recorders:
            process_point("Obs Before", results["rec_before"], cfg.x_before, cfg.y_before)

        return results["_hankel_data"]

    @staticmethod
    def verify_with_hankel(results):
        print("\n--- Plotting Hankel verification ---")
        hd = SimulationAnalyzer._compute_hankel_data(results)
        cfg = results["config"]
        f_valid = hd["f_valid"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
        
        ax = axes[0]
        ax.plot(f_valid, np.abs(hd["J_src_valid"]) / np.abs(hd["J_src_valid"][idx_fc]), label='Source', lw=2, color='lightgray')
        
        if "Recorder Full" in hd["obs_points"]:
            data_main = hd["obs_points"]["Recorder Full"]
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
    def _compare_hankel_verification(results_list):
        base_cfg = results_list[0]["config"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        source_plotted = False
        y_max = 0

        for i, res in enumerate(results_list):
            hd = SimulationAnalyzer._compute_hankel_data(res)
            cfg = res["config"]
            f_valid = hd["f_valid"]
            idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
            label = getattr(cfg, "label", cfg.solver_type.upper())

            if not source_plotted:
                axes[0].plot(f_valid, np.abs(hd["J_src_valid"]) / np.abs(hd["J_src_valid"][idx_fc]), label='Source', lw=2, color='lightgray')
                source_plotted = True

            if "Recorder Full" not in hd["obs_points"]:
                continue

            label = cfg.solver_type.upper()
            data_main = hd["obs_points"]["Recorder Full"]
            H_sim_c = data_main["H_sim"]
            H_anal = data_main["H_analytical"]
            norm_sim = np.abs(H_sim_c[idx_fc])
            norm_anal = np.abs(H_anal[idx_fc])
            color = colors[i]

            axes[0].plot(f_valid, np.abs(H_sim_c) / norm_sim, label=label, lw=2, color=color)
            axes[1].plot(f_valid, np.unwrap(np.angle(H_sim_c)), label=label, lw=2, color=color)

            if i == 0:
                axes[0].plot(f_valid, np.abs(H_anal) / norm_anal, '--', label='Analytical', lw=2, color='black')
                axes[1].plot(f_valid, np.unwrap(np.angle(H_anal)), '--', label='Analytical', lw=2, color='black')

            y_max = max(y_max, np.max(np.abs(H_sim_c[f_valid < cfg.f_break]) / norm_sim))

        for ax in axes:
            ax.axvline(base_cfg.f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({base_cfg.f_break:.2e} Hz)')
            ax.axvline(base_cfg.f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({base_cfg.f_min:.2e} Hz)')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        axes[0].set_ylim(0, 1.3 * y_max if y_max > 0 else 1.0)
        axes[0].set_title(f'Magnitude Response [{base_cfg.solver_type.upper()} vs comparison]')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Normalized Magnitude')
        axes[1].set_title(f'Phase Response [{base_cfg.solver_type.upper()} vs comparison]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (rad)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _compare_error_analysis(results_list):
        base_cfg = results_list[0]["config"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        max_val = 0
        max_phase_error = 0
        min_phase_error = 0

        for i, res in enumerate(results_list):
            hd = SimulationAnalyzer._compute_hankel_data(res)
            cfg = res["config"]
            f_valid = hd["f_valid"]
            idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
            color = colors[i]
            label = getattr(cfg, "label", cfg.solver_type.upper())

            if "Recorder Full" not in hd["obs_points"]:
                continue

            data = hd["obs_points"]["Recorder Full"]
            H_sim, H_anal = data["H_sim"], data["H_analytical"]
            sim_mag = np.abs(H_sim) / np.abs(H_sim[idx_fc])
            anal_mag = np.abs(H_anal) / np.abs(H_anal[idx_fc])
            rel_error = np.abs(sim_mag - anal_mag) / anal_mag

            mask = (f_valid > cfg.f_min) & (f_valid < cfg.f_break)
            if np.any(mask):
                max_val = np.max([max_val, np.max(rel_error[mask])])

            rms_rel_error = np.sqrt(np.mean(rel_error[mask]**2)) if np.any(mask) else 0
            print(f"{label}: RMS Relative Error in valid band = {np.round(rms_rel_error*100, 2)}%")

            phase_diff = np.unwrap(np.angle(H_sim)) - np.unwrap(np.angle(H_anal))
            phase_diff -= phase_diff[idx_fc]
            phase_error_deg = np.rad2deg(phase_diff)
            
            rms_phase_error = np.sqrt(np.mean(phase_error_deg[mask]**2)) if np.any(mask) else 0
            print(f"{label}: RMS Phase Error in valid band = {np.round(rms_phase_error, 2)} degrees")

            if np.any(mask):
                min_phase_error = np.min([min_phase_error, np.min(phase_error_deg[mask])])
                max_phase_error = np.max([max_phase_error, np.max(phase_error_deg[mask])])

            axes[0].plot(f_valid, rel_error * 100, label=label, lw=1.5, color=color)
            axes[1].plot(f_valid, phase_error_deg, label=label, lw=1.5, color=color)

        for ax in axes:
            ax.axvline(base_cfg.f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({base_cfg.f_break:.2e} Hz)')
            ax.axvline(base_cfg.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({base_cfg.f_c:.2e} Hz)')
            ax.axvline(base_cfg.f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({base_cfg.f_min:.2e} Hz)')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        axes[0].set_title('Relative Magnitude Error [comparison]')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Relative Error (%)')
        axes[0].set_ylim(0, 300 * max_val if max_val > 0 else 1.0)
        axes[1].set_title('Phase Error Dispersion [comparison]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase Error (Degrees)')
        axes[1].set_ylim(1.5 * min_phase_error, 1.5 * max_phase_error if max_phase_error != 0 else 1.0)

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
        
        max_val = 0
        max_phase_error = 0
        min_phase_error = 0
        
        for i, (name, data) in enumerate(hd["obs_points"].items()):
            H_sim, H_anal = data["H_sim"], data["H_analytical"]
            sim_mag = np.abs(H_sim) / np.abs(H_sim[idx_fc])
            anal_mag = np.abs(H_anal) / np.abs(H_anal[idx_fc])
            rel_error = np.abs(sim_mag - anal_mag) / anal_mag
            
            mask = (f_valid > cfg.f_min) & (f_valid < cfg.f_break)
            max_val = np.max([max_val, np.max(rel_error[mask])])
            print(max_val)
            
            phase_diff = np.unwrap(np.angle(H_sim)) - np.unwrap(np.angle(H_anal))
            phase_diff -= phase_diff[idx_fc]
            phase_error_deg = np.rad2deg(phase_diff)
            
            min_phase_error = np.min([min_phase_error, np.min(phase_error_deg[mask])])
            max_phase_error = np.max([max_phase_error, np.max(phase_error_deg[mask])])
            
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
        axes[0].set_ylim(0, 300 * max_val)
        axes[1].set_title(f'Phase Error Dispersion [{cfg.solver_type.upper()}]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase Error (Degrees)')
        axes[1].set_ylim(1.5 * min_phase_error, 1.5 * max_phase_error)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_2d_animation(results, fps=40, color_sensitivity=0.001):
        cfg = results["config"]
        hist = results["history"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))[:cfg.nx]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny]
        X, Y = np.meshgrid(nodes_x, nodes_y)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        
        z_norm = np.array(results["z_norm"]) # Using fetched z_norm
        vmax = np.max(np.abs(hist * z_norm)) * 0.8 * color_sensitivity
        if vmax == 0:
            vmax = 1e-4
        quad = ax.pcolormesh(X, Y, (hist[0] * z_norm).T, shading='nearest', cmap='RdBu_r', vmin=-vmax, vmax=vmax, zorder=1)
                 
        recorders = getattr(cfg, "recorders", ["all"])         
                    
        scat1 = ax.scatter(nodes_x[cfg.x0], nodes_y[cfg.y0], color='red', s=20, zorder=2, label='Source')
        scat2 = ax.scatter(nodes_x[cfg.x1], nodes_y[cfg.y1], color='green', s=20, zorder=2, label='Recorder')
        
        if "all" in recorders or "diag" in recorders:
            scat3 = ax.scatter(nodes_x[cfg.x_diag], nodes_y[cfg.y_diag], color='cyan', s=30, zorder=2, marker='x', label='Obs Diag')
        if "all" in recorders or "after" in recorders:
            scat4 = ax.scatter(nodes_x[cfg.x_after], nodes_y[cfg.y_after], color='magenta', s=30, zorder=2, marker='x', label='Obs After')
        if "all" in recorders or "before" in recorders:
            scat5 = ax.scatter(nodes_x[cfg.x_before], nodes_y[cfg.y_before], color='yellow', s=30, zorder=2, marker='x', label='Obs Before')
        
        ax.legend(loc='upper right', fontsize=8)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontweight='bold')
        fs = results["frame_skip"]
        
        def update(i):
            frame_data = (hist[i] * z_norm).T
            quad.set_array(frame_data.ravel())
            time_text.set_text(f'Frame {i*fs}/{cfg.nt} | t = {i*fs*cfg.dt*1e9:.3f} ns')
            out = [quad, time_text, scat1, scat2]

            if "all" in recorders or "diag" in recorders:
                out.append(scat3)

            if "all" in recorders or "after" in recorders:
                out.append(scat4)

            if "all" in recorders or "before" in recorders:
                out.append(scat5)

            return tuple(out)

        interval_ms = max(1, int(1000 / fps))
        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval_ms, blit=True)
        results["ani_2d"] = ani
        ax.set_title(f"Field Animation ({cfg.solver_type.upper()})")
        plt.show()

    @staticmethod
    def plot_eps_r_colormap(*results_list):
        """Plot the relative-permittivity map for one or more simulations."""
        if len(results_list) == 1 and isinstance(results_list[0], (list, tuple)):
            results_list = tuple(results_list[0])
        if len(results_list) == 0:
            raise ValueError("plot_eps_r_colormap requires at least one results dict.")

        n_plots = len(results_list)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)

        for ax, res in zip(axes[0], results_list):
            cfg = res["config"]
            nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))[:cfg.nx]
            nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny]
            X, Y = np.meshgrid(nodes_x, nodes_y)

            eps_r = np.asarray(cfg.epsilon_r)
            mesh = ax.pcolormesh(X, Y, eps_r.T, shading='nearest', cmap='viridis')
            ax.set_aspect('equal')
            ax.set_xlabel("X-position (m)")
            ax.set_ylabel("Y-position (m)")
            label = getattr(cfg, "label", cfg.solver_type.upper())
            ax.set_title(fr"$\epsilon_r$ map: {label}")
            fig.colorbar(mesh, ax=ax, label=r"$\epsilon_r$")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_1d_intensity(results, fps=40):
        """1D intensity animation of Ez² at the recorder plane (x = x1) over time."""
        cfg = results["config"]
        rec = results["recorder"]          # shape: (n_frames, ny) — frame-skipped
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny + 1]

        fig, ax = plt.subplots(figsize=(8, 4))
        max_int = np.max(rec ** 2) or 1.0
        line, = ax.plot([], [], color='red', lw=2)

        ax.set_xlim(nodes_y.min(), nodes_y.max())
        ax.set_ylim(0, max_int * 1.2)
        ax.set_xlabel("Y-position (m)")
        ax.set_ylabel("Ez² Intensity")

        # Shade waveguide layers if geometry info is available
        if all(hasattr(cfg, a) for a in ("n_air", "n_clad", "n_core")):
            ny_a = int(cfg.n_air)
            ny_c = int(cfg.n_clad)
            ny_k = int(cfg.n_core)
            ax.axvspan(nodes_y[ny_a],           nodes_y[ny_a + ny_c],              color='yellow', alpha=0.12, label='Cladding')
            ax.axvspan(nodes_y[ny_a + ny_c],    nodes_y[ny_a + ny_c + ny_k],       color='blue',   alpha=0.12, label='Core')
            ax.axvspan(nodes_y[ny_a + ny_c + ny_k], nodes_y[ny_a + 2*ny_c + ny_k],color='yellow', alpha=0.12)
            ax.legend(fontsize=8)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontweight='bold')
        fs = results["frame_skip"]
        nt = cfg.nt

        def update(i):
            line.set_data(nodes_y[:rec.shape[1]], rec[i, :] ** 2)
            time_text.set_text(f'Intensity | Frame {i*fs}/{nt} | t = {i*fs*cfg.dt*1e9:.3f} ns')
            return line, time_text

        interval_ms = max(1, int(1000 / fps))
        ani = FuncAnimation(fig, update, frames=len(rec), interval=interval_ms, blit=True)
        results["ani_1d"] = ani
        ax.set_title(f"1D Ez² at Recorder Plane x=x1 ({cfg.solver_type.upper()})")
        plt.show()

    @staticmethod
    def plot_cumulative_energy_flux(*results_list, time_index=-1, use_squared=True):
        """Plot cumulative-in-time E(y) profiles at x=x1 for one or more simulations."""
        if len(results_list) == 1 and isinstance(results_list[0], (list, tuple)):
            results_list = tuple(results_list[0])
        if len(results_list) == 0:
            raise ValueError("plot_cumulative_energy_flux requires at least one results dict.")

        first_cfg = results_list[0]["config"]
        first_ez_line_t = results_list[0]["recorder_full"]
        first_y_nodes = np.concatenate(([0], np.cumsum(first_cfg.dy)))

        fig, ax = plt.subplots(figsize=(9, 4))

        # Shade waveguide layers if geometry info is available.
        if all(hasattr(first_cfg, a) for a in ("n_air", "n_clad", "n_core")):
            ny_a = int(first_cfg.n_air)
            ny_c = int(first_cfg.n_clad)
            ny_k = int(first_cfg.n_core)
            if ny_a + 2 * ny_c + ny_k < len(first_y_nodes):
                ax.axvspan(first_y_nodes[ny_a], first_y_nodes[ny_a + ny_c], color='yellow', alpha=0.12, label='Cladding')
                ax.axvspan(first_y_nodes[ny_a + ny_c], first_y_nodes[ny_a + ny_c + ny_k], color='blue', alpha=0.12, label='Core')
                ax.axvspan(first_y_nodes[ny_a + ny_c + ny_k], first_y_nodes[ny_a + 2 * ny_c + ny_k], color='yellow', alpha=0.12)

        plotted_times = []
        for res in results_list:
            cfg = res["config"]
            ez_line_t = res["recorder_full"]
            y_nodes = np.concatenate(([0], np.cumsum(cfg.dy)))
            y = y_nodes[:ez_line_t.shape[1]]

            signal_t_y = ez_line_t ** 2 if use_squared else ez_line_t
            cumulative_t_y = np.cumsum(signal_t_y, axis=0) * cfg.dt
            idx = int(np.clip(time_index, -len(cumulative_t_y), len(cumulative_t_y) - 1))
            profile_y = cumulative_t_y[idx, :]
            t_ns = res["times"][idx] * 1e9
            plotted_times.append(t_ns)

            label = getattr(cfg, "label", cfg.solver_type.upper())
            ax.plot(y, profile_y, lw=2, label=label)

        ax.set_xlabel("Y-position (m)")
        if use_squared:
            ax.set_ylabel(r"$\int E_z(y,t)^2\,dt$")
            ax.set_title("Cumulative E²(y) at x=x1")
        else:
            ax.set_ylabel(r"$\int E_z(y,t)\,dt$")
            ax.set_title("Cumulative E(y) at x=x1")

        if plotted_times:
            if len(plotted_times) == 1:
                ax.set_title(ax.get_title() + f", t={plotted_times[0]:.3f} ns")
            else:
                ax.set_title(ax.get_title() + f" (multi-input, index={time_index})")

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_recorders(*results_list):
        fig, ax = plt.subplots(figsize=(10, 4))
        for res in results_list:
            cfg = res["config"]
            t = res["times"]
            ez = res["recorder_full"][:, cfg.y1]
            label = getattr(cfg, "label", cfg.solver_type.upper())
            ax.plot(t * 1e9, ez, label=label, lw=1.5)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Ez Amplitude")
        ax.set_title("Recorder Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        if results_list:
            SimulationAnalyzer._compare_hankel_verification(results_list)
            SimulationAnalyzer._compare_error_analysis(results_list)

# ==============================================================================
# Execution Example
# ==============================================================================
if __name__ == "__main__":
    
    Compare_FCI_YEE = False
    if Compare_FCI_YEE:
        t0 = time.time()
        res_fci_schur = SimulationRunner.execute(
            solver_type = "fci",
            schur = False,
            frame_skip = 10,
            finesse = 10,
            free_space_sim = True,
            grid_refinement = False,
            do_hankel = True,
            recorders = ["after"],
            label = "FCI (Schur)"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")

        SimulationAnalyzer.plot_2d_animation(res_fci_schur)

        t0 = time.time()
        res_yee = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=10,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"Yee"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_yee)
        
        SimulationAnalyzer.compare_recorders(res_fci_schur, res_yee)
    
    Compare_Grid_Refinement_YEE = False
    
    if Compare_Grid_Refinement_YEE:
        t0 = time.time()
        res_yee_10 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=10,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"Yee $\lambda/10$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_yee_10)
        
        t0 = time.time()
        res_yee_20 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=20,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"Yee $\lambda/20$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_yee_20)
        
        t0 = time.time()
        res_yee_30 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"Yee $\lambda/30$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_yee_30)      
        SimulationAnalyzer.compare_recorders(res_yee_10, res_yee_20, res_yee_30)

    Compare_Grid_Refinement_FCI = False
    
    if Compare_Grid_Refinement_FCI:
        t0 = time.time()
        res_fci_10 = SimulationRunner.execute(
            solver_type="fci",
            schur = False,
            frame_skip=10,
            finesse=10,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"FCI $\lambda/10$"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_fci_10)
        
        t0 = time.time()
        res_fci_20 = SimulationRunner.execute(
            solver_type="fci",
            schure = False,
            frame_skip=10,
            finesse=20,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"FCI $\lambda/20$"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_fci_20)
        
        t0 = time.time()
        res_fci_30 = SimulationRunner.execute(
            solver_type="fci",
            schur = False,
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"FCI $\lambda/30$"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_fci_30)      
        SimulationAnalyzer.compare_recorders(res_fci_10, res_fci_20, res_fci_30)

    Grin_vs_step_Yee = True
    if Grin_vs_step_Yee:
        t0 = time.time()
        res_step = SimulationRunner.execute(
            solver_type = "yee",
            frame_skip = 10,
            eps_core = 2.5**2,
            finesse = 20,
            free_space_sim = False,
            grid_refinement = 'gradual',
            wg_type = 'step',
            do_hankel = False,
            recorders = ["after"],
            label = "Step"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_grin = SimulationRunner.execute(
            solver_type = "yee",
            deps_max = 0.4,
            eps_core = 2.5**2,
            frame_skip = 10,
            finesse = 20,
            free_space_sim = False,
            grid_refinement = 'gradual',
            wg_type = 'grin',
            do_hankel = False,
            recorders = ["after"],
            label = "Grin"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")

        SimulationAnalyzer.plot_2d_animation(res_step)
        SimulationAnalyzer.plot_2d_animation(res_grin)
        SimulationAnalyzer.plot_eps_r_colormap(res_step, res_grin)
        SimulationAnalyzer.plot_cumulative_energy_flux(res_step, res_grin)

    
