import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.special as sp_special
from scipy.fft import fft, fftfreq

################################################################################################################################################
#                                                             Parameters:                                                       
################################################################################################################################################
class SimulationConfig:
    """Handles all physical and numerical parameters."""
        
    def __init__(self, **kwargs):
        
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
        self.L_wg   = f * 4 * self.lam_c # Length of the waveguide in meters
        self.w_core = f * 2 * self.lam_c # Width of the core in meters
        self.w_clad = f * 2 * self.lam_c # Width of the cladding on each side in meters
        self.w_air  = f * 2.5 * self.lam_c # Width of the air region on each side next to the cladding in meters
        self.d      = f * 2 * self.lam_c # Distance between source and the barrier in meters
        self.t_m    = f * 0.03 * self.lam_c # Thickness of the barrier infront of the waveguide
        self.Ll     = f * 2 * self.lam_c # Length of the left region before the source in meters
        self.Lr     = f * 2 * self.lam_c # Length of the right region after the waveguide in meters
        self.L      = self.Ll + self.d + self.t_m + self.L_wg + self.Lr # Total length of the simulation domain in x direction in meters
        self.W      = f * (2 * self.w_air + 2 * self.w_clad + self.w_core) # Total width of the simulation domain in y direction in meters
        self.T      = self.t0 + 3*self.sig_t + (self.d + self.t_m + np.sqrt(self.eps_core) * self.L_wg + self.Lr) / self.c # Total of time to capture the full pulse propagation through the waveguide
        self.finesse = 30 # Dictates how many cells per central wavelength
        self.f_min  = self.c / self.W

        # Give possibility to change parameters via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.wg_type = getattr(self, "wg_type", "step") # Default waveguide type is "step"
        self.grid_refinement = getattr(self, "grid_refinement", "gradual") # Whether to use non-uniform grid refinement

        self.alpha = getattr(self, "alpha", 1.05)
        
        # Build Grids
        self._build_dx()
        self._build_dy()
        
        self.nx = len(self.dx) + 1
        self.ny = len(self.dy) + 1
        
        self.epsilon_r  = np.ones((self.nx, self.ny))
        self.sigma      = np.zeros((self.nx-2, self.ny-2))
        self.gamma      = np.zeros((self.nx-2, self.ny-2))

        # Grid Spacing (Non-uniform)
        self.dx_f = self.dx.min()
        self.dy_f = self.dy.min()
        
        self.x0 = getattr(self, "x0", self.n_Ll)
        self.y0 = getattr(self, "y0", self.ny // 2)
        
        if "x1" in kwargs:
            self.x1 = self.x0 + kwargs["x1"]
        else:
            self.x1 = self.nx - 50
            
        if "y1" in kwargs:
            self.y1 = self.y0 + kwargs["y1"]
        else:
            self.y1 = self.y0

        # Observation points with standard values
        # Diagonal from the source
        self.x_diag = getattr(self, "x_diag", self.x0 + 20)
        self.y_diag = getattr(self, "y_diag", self.y0 + 20)

        # Horizontal after the waveguide
        self.x_after = getattr(self, "x_after", self.nx - 50)
        self.y_after = getattr(self, "y_after", self.y0)

        # Horizontal before the waveguide (and wall)
        self.x_before = getattr(self, "x_before", self.x0 + int(self.n_d // 2))
        self.y_before = getattr(self, "y_before", self.y0)

        self.dx_d = np.concatenate(([self.dx[0]/2], (self.dx[:-1] + self.dx[1:])/2, [self.dx[-1]/2]))
        self.dy_d = np.concatenate(([self.dy[0]/2], (self.dy[:-1] + self.dy[1:])/2, [self.dy[-1]/2]))

        # Time Stepping
        # Strictly keep CFL below 1.0 (e.g. 0.95) for unconditionally stable propagation into non-uniform domains
        CFL = 0.95
        self.dt  = CFL / (self.c * np.sqrt((1/self.dx.min()**2) + (1/self.dy.min()**2)))
        self.nt = int(np.ceil(self.T / self.dt))
        
        
        self.free_space_sim = getattr(self, "free_space_sim", False)
        self.setup_waveguide()
        
        self.v_local = self.c / np.sqrt(self.epsilon_r)
        self.Z_local = self.Z0 / np.sqrt(self.epsilon_r)
        
    def _build_dx(self):
        """Constructs the non-uniform grid in the x-direction."""
        self.dx_0 = self.lam_c / self.finesse
        
        if not getattr(self, "grid_refinement", "gradual"):
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.n_d = int(np.ceil(self.d / self.dx_0))
            self.n_wg = int(np.ceil(self.L_wg / self.dx_0))
            self.n_Lr = int(np.ceil(self.Lr / self.dx_0))
            nx_total = self.n_Ll + self.n_d + self.n_wg + self.n_Lr
            # To avoid roundoff errors on total length L, we just build it segment by segment uniform
            self.dx = np.concatenate([
                np.full(self.n_Ll, self.Ll / self.n_Ll),
                np.full(self.n_d, self.d / self.n_d),
                np.full(self.n_wg, self.L_wg / self.n_wg),
                np.full(self.n_Lr, self.Lr / self.n_Lr)
            ])
            return
        if self.grid_refinement == "gradual":    
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.L_f_dt, self.n_f_dt = self.L_and_n_fine(self.dx_0, self.t_m, self.alpha)
            self.n_d = int(np.ceil(self.d / self.dx_0 - self.L_f_dt / self.dx_0)) + self.n_f_dt
            self.L_f_twg, self.n_f_twg = self.L_and_n_fine(self.dx_0 / np.sqrt(self.eps_core), self.t_m, self.alpha)
            self.n_wg = int(np.ceil((self.L_wg - self.L_f_twg) * np.sqrt(self.eps_core)/ self.dx_0) + self.n_f_twg)
            self.L_f_wg_Lr, self.n_f_wg_Lr = self.L_and_n_fine(self.dx_0, self.dx_0 / np.sqrt(self.eps_core), self.alpha)
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
        """Constructs the non-uniform grid in the y-direction."""
        self.dy_0 = self.lam_c / self.finesse
        
        if not getattr(self, "grid_refinement", "gradual"):
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
            self.L_f_ac, self.n_f_ac = self.L_and_n_fine(self.dy_0, self.dy_0 / np.sqrt(self.eps_clad), self.alpha)
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
        
    def L_and_n_fine(self, d_coarse, d_fine, alpha = np.sqrt(2)):
        if self.eps_clad == self.eps_core:
            return 0, 0
        else:
            n_f = int(np.ceil(np.log(d_coarse / d_fine) / np.log(alpha))) - 1
            L_f = d_fine * (alpha**(n_f-1) - alpha) / (alpha - 1)
            
            return L_f, n_f
    
    def setup_waveguide(self):
        if getattr(self, "free_space_sim", False):
            return
            
        self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air):int(-self.n_air)] = self.eps_clad
        if self.wg_type == "step":
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = self.eps_core
        else:
            eps_val = lambda y: self.eps_core + self.a_eps * (y-self.W/2)**2 + self.b_eps * (y-self.W/2)**4 
            self.epsilon_r[int(self.n_Ll + self.n_d + 1):int(self.n_Ll + self.n_d + 1 + self.n_wg), int(self.n_air + self.n_clad):int(- (self.n_air + self.n_clad))] = eps_val(self.dy_core * np.arange(self.n_air + self.n_clad, (self.n_air + self.n_clad + self.n_core) + 1))
            
        self.sigma[int(self.n_Ll + self.n_d - 1),   :int(self.n_air + self.n_clad)] = 3.5e7 #conductivity of aluminum
        self.sigma[int(self.n_Ll + self.n_d - 1), - int(self.n_air + self.n_clad):] = 3.5e7
        
################################################################################################################################################
#                                                          Yee Solver Class:
################################################################################################################################################
class YeeSolver:
    def __init__(self, config):
        self.cfg = config
        self.init_fields()
        self.init_pml()
        self.init_coefficients()

    def init_fields(self):
        self.ix = slice(1,-1)
        self.iy = slice(1,-1)
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

    def init_pml(self):
        # Bound the PML thickness to 2x finesse (2 wavelengths), which effectively 
        # utilizes the massive 2.5 wavelength air margin safely without collision.
        p, m = int(2.0 * self.cfg.finesse), 4
        eta_max = (m + 1) / (150 * np.pi * min([self.cfg.dx_0, self.cfg.dy_0]))
        ksi_k_max = 1.0 
        
        self.kx, self.ky = np.ones((self.cfg.nx, self.cfg.ny)), np.ones((self.cfg.nx, self.cfg.ny))
        self.etax, self.etay = np.zeros((self.cfg.nx, self.cfg.ny)), np.zeros((self.cfg.nx, self.cfg.ny))

        for i in range(p):
            d_pml = (p - i) / p
            val_k = 1.0 + (ksi_k_max - 1.0) * (d_pml**m)
            val_eta = eta_max * (d_pml**m)

            self.kx[i, :], self.kx[-1-i, :] = val_k, val_k
            self.ky[:, i], self.ky[:, -1-i] = val_k, val_k
            self.etax[i, :], self.etax[-1-i, :] = val_eta, val_eta
            self.etay[:, i], self.etay[:, -1-i] = val_eta, val_eta

    def init_coefficients(self):
        cfg = self.cfg
        # Update coefficients
        self.bxp = self.kx / (cfg.v_local * cfg.dt) + cfg.Z_local * self.etax / 2.0
        self.byp = self.ky / (cfg.v_local * cfg.dt) + cfg.Z_local * self.etay / 2.0
        self.bxm = self.kx / (cfg.v_local * cfg.dt) - cfg.Z_local * self.etax / 2.0
        self.bym = self.ky / (cfg.v_local * cfg.dt) - cfg.Z_local * self.etay / 2.0

        # Interpolations
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

        # Magnetic Field Update
        self.Hx_dot_old[:] = self.Hx_dot
        diff_Ez_y = self.Ez[:, 1:] - self.Ez[:, :-1]
        self.Hx_dot = (self.bym_hx * self.Hx_dot - diff_Ez_y * self.inv_dy[None, :]) / self.byp_hx
        self.Hx += (self.bxp_hx * self.Hx_dot - self.bxm_hx * self.Hx_dot_old) / self.bz_h

        self.Hy_dot_old[:] = self.Hy_dot
        diff_Ez_x = self.Ez[1:, :] - self.Ez[:-1, :]
        self.Hy_dot += diff_Ez_x * self.inv_dx[:, None] / self.bz_h
        self.Hy = (self.bxm_hy * self.Hy + (self.byp_hy * self.Hy_dot - self.bym_hy * self.Hy_dot_old)) / self.bxp_hy

        # Electric Field Update
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
        
        # Source
        src = cfg.A * np.cos(2*np.pi*cfg.f_c*(t-cfg.t0)) * np.exp(-0.5*((t-cfg.t0)/cfg.sig_t)**2)
        self.Ez[cfg.x0, cfg.y0] -= cfg.dx[cfg.x0] * cfg.dy[cfg.y0] * src / self.coef_p[cfg.x0-1, cfg.y0-1]

################################################################################################################################################
#                                                          Simulation Runner:
################################################################################################################################################
class SimulationRunner:
    @staticmethod
    def execute(frame_skip=3, **kwargs):
        """
        Runs the simulation with optional parameter overrides.
        """
        config = SimulationConfig(**kwargs)
        solver = YeeSolver(config)
        
        n_frames = int(np.ceil(config.nt / frame_skip))
        field_history = np.zeros((n_frames, config.nx, config.ny), dtype=np.float32)
        recorder_plane = np.zeros((n_frames, config.ny), dtype=np.float32)
        recorder_full = np.zeros((config.nt, config.ny), dtype=np.float32)
        
        # New point recorders
        rec_diag = np.zeros(config.nt, dtype=np.float32)
        rec_after = np.zeros(config.nt, dtype=np.float32)
        rec_before = np.zeros(config.nt, dtype=np.float32)
        
        frame_idx = 0
        for it in tqdm(range(config.nt), desc=f"Simulating (nt={config.nt})"):
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
            "times": np.arange(config.nt) * config.dt
        }

    @staticmethod
    def plot_2d_animation(results, fps=40):
        """Triggers the 2D Field Animation.
        """
        cfg = results["config"]
        hist = results["history"]
        
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        X, Y = np.meshgrid(nodes_x, nodes_y)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        
        # Initial plot:
        quad = ax.pcolormesh(X, Y, (hist[0] * cfg.Z_local).T, 
                             shading='nearest', cmap='RdBu_r', vmin=-1e-4, vmax=1e-4, zorder=1)
                             
        # Plot source and recorders on top
        scat1 = ax.scatter(nodes_x[cfg.x0], nodes_y[cfg.y0], color='red', s=20, zorder=2, label='Source')
        scat2 = ax.scatter(nodes_x[cfg.x1], nodes_y[cfg.y1], color='green', s=20, zorder=2, label='Recorder')
        
        # New observation points
        scat3 = ax.scatter(nodes_x[cfg.x_diag], nodes_y[cfg.y_diag], color='cyan', s=30, zorder=2, marker='x', label='Obs Diag')
        scat4 = ax.scatter(nodes_x[cfg.x_after], nodes_y[cfg.y_after], color='magenta', s=30, zorder=2, marker='x', label='Obs After')
        scat5 = ax.scatter(nodes_x[cfg.x_before], nodes_y[cfg.y_before], color='yellow', s=30, zorder=2, marker='x', label='Obs Before')
        
        ax.legend(loc='upper right', fontsize=8)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontweight='bold')

        fs = results["frame_skip"]
        def update(i):
            frame_data = (hist[i] * cfg.Z_local).T
            quad.set_array(frame_data.ravel())
            time_text.set_text(f'Frame {i*fs}/{cfg.nt} | t = {i*fs*cfg.dt*1e9:.3f} ns')
            return quad, time_text, scat1, scat2, scat3, scat4, scat5

        interval_ms = max(1, int(1000 / fps))
        print(f'[2D] fps={fps}, interval={interval_ms}ms, total_frames={len(hist)}')
        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval_ms, blit=True)
        results["ani_2d"] = ani
        plt.show()

    @staticmethod
    def plot_1d_intensity(results, fps=40):
        """Triggers the 1D Intensity Animation at the recorder plane.
        """
        cfg = results["config"]
        rec = results["recorder"]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        
        fig, ax = plt.subplots(figsize=(8, 4))
        max_int = np.max(rec**2)
        line, = ax.plot([], [], color='red', lw=2)
        
        ax.set_xlim(nodes_y.min(), nodes_y.max())
        ax.set_ylim(0, max_int * 1.2)
        
        # Waveguide:
        ax.axvspan(nodes_y[int(cfg.n_air)             ], nodes_y[int(cfg.n_air + cfg.n_clad)], color='yellow', alpha=0.1)
        ax.axvspan(nodes_y[int(cfg.n_air + cfg.n_clad)], nodes_y[int(cfg.n_air + cfg.n_clad + cfg.n_core)], color='blue'  , alpha=0.1)
        ax.axvspan(nodes_y[int(cfg.n_air + cfg.n_clad + cfg.n_core)], nodes_y[int(cfg.n_air + 2 * cfg.n_clad + cfg.n_core)], color='yellow', alpha=0.1)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontweight='bold')

        fs = results["frame_skip"]
        n_frames = len(rec)
        def update(i):
            line.set_data(nodes_y, rec[i, :]**2)
            time_text.set_text(f'Intensity | Frame {i*fs}/{cfg.nt} | t = {i*fs*cfg.dt*1e9:.3f} ns')
            return line, time_text

        interval_ms = max(1, int(1000 / fps))
        print(f'[1D] fps={fps}, interval={interval_ms}ms, total_frames={n_frames}')
        ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=True)
        results["ani_1d"] = ani
        plt.show()

    @staticmethod
    def verify_with_hankel(results):
        print("\n--- Plotting Hankel verification ---")
        
        cfg = results["config"]
        
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        
        # Setup time and frequency 
        t = np.arange(cfg.nt) * cfg.dt
        n_pad = 2**int(np.ceil(np.log2(cfg.nt * 8)))  # High resolution zero padding
        freqs = fftfreq(n_pad, cfg.dt)
        f_min = getattr(cfg, "hankel_f_min", cfg.f_c * 0.2)
        f_max = getattr(cfg, "hankel_f_max", cfg.f_c * 1.8)
        band_idx = np.where((freqs > f_min) & (freqs < f_max))[0]
        f_valid = freqs[band_idx]
        omega = 2 * np.pi * f_valid
        k0 = omega / cfg.c
        
        # FFT of source function (with zero padding)
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
            if r == 0:
                print(f"Skipping {name}: r=0")
                return
            
            Ez_sim_f = fft(ez_data, n=n_pad) * cfg.dt
            Ez_sim_valid = Ez_sim_f[band_idx]
            H_sim = Ez_sim_valid / J_src_valid
            H_sim_corrected = H_sim * np.exp(1j * omega * cfg.dt)
            H_analytical = -(omega * cfg.mu0 / 4) * sp_special.hankel2(0, k0 * r)
            results["_hankel_data"]["obs_points"][name] = {
                "r": r,
                "H_sim": H_sim_corrected,
                "H_analytical": H_analytical
            }

        process_point("Recorder Full", results["recorder_full"][:, cfg.y1], cfg.x1, cfg.y1)
        process_point("Obs Diag", results["rec_diag"], cfg.x_diag, cfg.y_diag)
        process_point("Obs After", results["rec_after"], cfg.x_after, cfg.y_after)
        process_point("Obs Before", results["rec_before"], cfg.x_before, cfg.y_before)

        # Plotting — Magnitude and Phase only
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
        
        ax = axes[0]
        ax.plot(f_valid, np.abs(J_src_valid) / np.abs(J_src_valid[idx_fc]), label='Source', lw=2, color='lightgray', alpha=1.0)
        
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
        ax.set_title('Magnitude Response')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        axes[1].set_xscale('log')
        axes[1].set_title('Phase Response')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (rad)')
        axes[1].legend()
        axes[1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_error_analysis(results):
        if "_hankel_data" not in results:
            print("Run verify_with_hankel first!")
            return
            
        hd = results["_hankel_data"]
        cfg = results["config"]
        f_valid = hd["f_valid"]
        
        # Print key frequency limits
        f_src_max = cfg.f_c * (1 + 1)  # Source bandwidth edge: f_c + a*sigma_f = 2*f_c
        f_nyquist = cfg.c / (2 * cfg.dx_f)
        f_cutoff = (1 / (np.pi * cfg.dt)) * np.arcsin(cfg.c * cfg.dt / cfg.dx_f)
        print(f"  Source bandwidth:    ~[0, {f_src_max:.3e}] Hz  (2 * f_c)")
        print(f"  Spatial Nyquist:     {f_nyquist:.3e} Hz  ({f_nyquist/cfg.f_c:.1f} * f_c)")
        print(f"  Yee num. cutoff:     {f_cutoff:.3e} Hz  ({f_cutoff/cfg.f_c:.1f} * f_c)")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (name, data) in enumerate(hd["obs_points"].items()):
            H_sim, H_anal = data["H_sim"], data["H_analytical"]
            
            # Normalize both at f_c to remove constant amplitude offset from source injection
            sim_mag = np.abs(H_sim) / np.abs(H_sim[idx_fc])
            anal_mag = np.abs(H_anal) / np.abs(H_anal[idx_fc])
            rel_error = np.abs(sim_mag - anal_mag) / anal_mag
            
            # Phase Error Dispersion
            sim_phase = np.unwrap(np.angle(H_sim))
            anal_phase = np.unwrap(np.angle(H_anal))
            # Compute difference and pin it to 0 at the central frequency
            phase_diff = sim_phase - anal_phase
            phase_diff -= phase_diff[idx_fc]
            phase_error_deg = np.rad2deg(phase_diff)
            
            axes[0].plot(f_valid, rel_error * 100, label=name, lw=1.5, color=colors[i])
            axes[1].plot(f_valid, phase_error_deg, label=name, lw=1.5, color=colors[i])
            
        # Magnitude Error Formatting
        ax = axes[0]
        ax.axvline(cfg.f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({cfg.f_break:.2e} Hz)')
        ax.axvline(cfg.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({cfg.f_c:.2e} Hz)')
        ax.axvline(cfg.f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({cfg.f_min:.2e} Hz)') 
        ax.set_xscale('log')
        ax.set_title('Relative Magnitude Error')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Relative Error (%)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        # Phase Error Formatting
        ax = axes[1]
        ax.axvline(cfg.f_break, color='red', ls=':', alpha=0.5, label=f'f_break ({cfg.f_break:.2e} Hz)')
        ax.axvline(cfg.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c ({cfg.f_c:.2e} Hz)')
        ax.axvline(cfg.f_min, color='green', ls=':', alpha=0.5, label=f'f_min ({cfg.f_min:.2e} Hz)') 
        ax.set_xscale('log')
        ax.set_title('Phase Error Dispersion')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase Error (Degrees)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_grid_spacing(results):
        """Visualizes the dx and dy spacing to verify refinement."""
        cfg = results["config"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot DX spacing
        ax1.plot(cfg.dx, 'o-', markersize=2, label='dx spacing')
        ax1.set_title("X-Grid Spacing (dx)")
        ax1.set_xlabel("Cell Index (i)")
        ax1.set_ylabel("Spacing (m)")
        ax1.grid(True, alpha=0.3)
        
        # Plot DY spacing
        ax2.plot(cfg.dy, 'o-', markersize=2, color='orange', label='dy spacing')
        ax2.set_title("Y-Grid Spacing (dy)")
        ax2.set_xlabel("Cell Index (j)")
        ax2.set_ylabel("Spacing (m)")
        ax2.grid(True, alpha=0.3)
        
        # Highlight waveguide region in dy plot
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        ax2.axvspan(int(cfg.n_air), int(cfg.n_air + cfg.n_clad), color='yellow', alpha=0.1)
        ax2.axvspan(int(cfg.n_air + cfg.n_clad),int(cfg.n_air + cfg.n_clad + cfg.n_core), color='blue'  , alpha=0.1)
        ax2.axvspan(int(cfg.n_air + cfg.n_clad + cfg.n_core), int(cfg.n_air + 2 * cfg.n_clad + cfg.n_core), color='yellow', alpha=0.1)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mesh(results, zoom_x=(400, 600), zoom_y=None):
        """Plots the actual grid lines (the mesh) to see the refinement."""
        cfg = results["config"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))
        
    
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.vlines(nodes_x, ymin=nodes_y.min(), ymax=nodes_y.max(), colors='black', lw=0.5, alpha=0.5)
        ax.hlines(nodes_y, xmin=nodes_x.min(), xmax=nodes_x.max(), colors='blue', lw=0.5, alpha=0.5)

        ax.set_title("FDTD Mesh Visualization")
        ax.set_xlabel("X-position (m)")
        ax.set_ylabel("Y-position (m)")
        ax.legend()
        plt.show()

    @classmethod
    def run_full_analysis(cls, fps=60, frame_skip=3, do_hankel=False, **kwargs):
        data = cls.execute(frame_skip=frame_skip, **kwargs)
        cls.plot_2d_animation(data, fps=fps)
        cls.plot_1d_intensity(data, fps=fps)
        
        cfg = data["config"]
        if getattr(cfg, "grid_refinement", "gradual"):
            cls.plot_grid_spacing(data)
            cls.plot_mesh(data)
        
        if do_hankel:
            cls.verify_with_hankel(data)
            cls.plot_error_analysis(data)
        
        return data






if __name__ == "__main__":
    
    print("Starting simulation...")
    
    # Example 1: Full materials simulation, with grid refinement:
    results = SimulationRunner.run_full_analysis(
        wg_type = "grin",
        frame_skip = 5, 
        finesse = 20, 
        eps_core = 2.22**2, 
        eps_clad = 2.218**2, 
        free_space_sim = True, 
        grid_refinement = "step", # "gradual", "step" or False
        do_hankel = True
    )
    
    # # Free space sim to verify with Hankel
    # results = SimulationRunner.run_full_analysis(
    #     frame_skip = 10, 
    #     wg_type ="step", 
    #     finesse = 10, 
    #     free_space_sim = True, 
    #     grid_refinement = False, 
    #     do_hankel = True, 
    #     hankel_f_min = 0, 
    #     hankel_f_max = 3*299792458,
    # )

    print("Simulation finished.")
