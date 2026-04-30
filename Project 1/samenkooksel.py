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
        self.eps_clad   = 3.9
        self.eps_core   = 11.9
        self.gamma_f    = 8e-15
        self.sigma_wall = getattr(self, "sigma_wall", 3.5e7) # Conductivity of the metallic walls, set to 0 for PEC

        # Apply user overrides before deriving wavelength-dependent values.
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.double_wall = getattr(self, "double_wall", True)

        # Source Parameters
        self.lam_c  = getattr(self, "lam_c", 1) # Central wavelength
        self.f_c    = self.c / self.lam_c
        self.A      = 1.0
        self.a      = 3 # Amount of sigmas between fc and 0 in frequency domain
        self.sig_t  = self.a / (2 * np.pi * self.f_c)
        self.t0     = 4 * self.sig_t
        self.f_break = 2 * self.f_c
        
        # Dimensions expressed in amount of wavelengths
        f = getattr(self, "f", 1.0)
        self.L_wg   = getattr(self, "L_wg", 6) * f * self.lam_c
        self.w_core = getattr(self, "w_core",1.5) * f * self.lam_c
        self.w_clad = f * 1 * self.lam_c
        self.w_air  = f * 0.5 * self.lam_c
        self.d      = f * 2 * self.lam_c
        self.t_m    = f * 0.01 * self.lam_c
        self.Ll     = f * 0.5 * self.lam_c
        self.Lr     = f * 1 * self.lam_c
        self.wall_count = 2 if self.double_wall else 1
        self.L      = self.Ll + self.d + self.wall_count * self.t_m + self.L_wg + self.Lr
        self.W      = 2 * self.w_air + 2 * self.w_clad + self.w_core
        self.T      = self.t0 + 3*self.sig_t + (self.d + self.wall_count * self.t_m + np.sqrt(self.eps_core) * self.L_wg) / self.c
        self.finesse = kwargs.get("finesse", 30)

            
        self.wg_type = getattr(self, "wg_type", "step")
        self.grid_refinement = getattr(self, "grid_refinement", "gradual")
        self.alpha = getattr(self, "alpha", 1.05)
        
        self.n_pml = max(0, int(getattr(self, "n_pml", 30)))
        
        # Build Grids
        self._build_dx()
        self._build_dy()
        
        if self.fci_bc == "PBC" and self.solver_type == "fci":
            self.nx = len(self.dx)
            self.ny = len(self.dy)
        elif self.solver_type == "fci":
            self.nx = len(self.dx)
            self.ny = len(self.dy) + 1
            self.dx = self.dx[:-1] # make sure that we dont have an even matrix
        else:
            self.nx = len(self.dx) + 1
            self.ny = len(self.dy) + 1
        
        
        print(f"Grid: {self.nx} x {self.ny}")
        
        if self.solver_type == "fci":
            self.epsilon_r  = np.ones((self.nx, self.ny))
            self.sigma      = np.zeros((self.nx, self.ny))
            self.gamma      = self.gamma_f
        else:
            self.epsilon_r  = np.ones((self.nx, self.ny))
            self.sigma      = np.zeros((self.nx, self.ny))
            self.gamma      = np.zeros((self.nx, self.ny))

        self.dx_f = self.dx.min()
        self.dy_f = self.dy.min()
        
        self.x0 = getattr(self, "x0", self.n_pml + self.n_Ll)
        self.y0 = getattr(self, "y0", self.ny // 2)
        
        self.A *= (self.dx[self.x0] * self.dy[self.y0])**(-2)
        
        sum_dx = np.cumsum(self.dx)
        pml_offset_x = float(np.sum(self.dx[:self.n_pml])) if self.n_pml > 0 else 0.0
        pml_offset_y = float(np.sum(self.dy[:self.n_pml])) if self.n_pml > 0 else 0.0
        self.pml_offset_x = pml_offset_x
        self.pml_offset_y = pml_offset_y
        self.x_nodes = np.concatenate(([0.0], sum_dx))
        self.y_nodes = np.concatenate(([0.0], np.cumsum(self.dy)))
        self.x_nodes_phys = self.x_nodes - self.pml_offset_x
        self.y_nodes_phys = self.y_nodes - self.pml_offset_y
        self.y_region_spans_phys = self._build_y_region_spans(physical=True)
        self.y_region_spans_index = self._build_y_region_spans(physical=False)
        post_wg_offset = self.wall_count * self.t_m
        
        self.x1 = self.x0 + kwargs["x1"] if "x1" in kwargs else np.argmin(np.abs(sum_dx - (pml_offset_x + self.Ll + self.d + post_wg_offset + self.L_wg + 0.1*self.Lr)))
        self.y1 = self.y0 + kwargs["y1"] if "y1" in kwargs else self.y0
        
        self.f_min = self.c / (sum_dx[self.x1] - sum_dx[self.x0]) #Quasi static regime    

        dist_req = self.d / 2.0
        offset_x = np.searchsorted(np.cumsum(self.dx[self.x0:]), dist_req)
        offset_y = np.searchsorted(np.cumsum(self.dy[self.y0:]), dist_req)
        
        
        
        self.x_diag = getattr(self, "x_diag", self.x0 + offset_x)
        self.y_diag = getattr(self, "y_diag", self.y0 + offset_y)

        self.x_after = getattr(self, "x_after", np.argmin(np.abs(sum_dx - (pml_offset_x + self.Ll + self.d + post_wg_offset + self.L_wg))))
        self.y_after = getattr(self, "y_after", self.y0)
        self.x_start = getattr(self, "x_start", np.argmin(np.abs(sum_dx - (pml_offset_x + self.Ll+ self.d + self.t_m + 0.1*self.L_wg))))
        self.y_start = getattr(self, "y_start", self.y0)

        self.T_start = self.t0 + 4*self.sig_t + (self.d + self.t_m + 0.1*self.L_wg*self.eps_core**(1/2)) / self.c
        
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
        elif self.grid_refinement == "gradual":    
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.L_f_dt, self.n_f_dt = self._L_and_n_fine(self.dx_0, self.t_m, self.alpha)
            self.n_d = int(np.ceil(self.d / self.dx_0 - self.L_f_dt / self.dx_0)) + self.n_f_dt
            self.L_f_twg, self.n_f_twg = self._L_and_n_fine(self.dx_0 / np.sqrt(self.eps_core), self.t_m, self.alpha)
            self.n_wg = int(np.ceil((self.L_wg - self.L_f_twg) * np.sqrt(self.eps_core)/ self.dx_0) + self.n_f_twg)
            if self.double_wall:
                self.L_f_lr, self.n_f_lr = self._L_and_n_fine(self.dx_0, self.t_m, self.alpha)
                self.n_Lr = int(np.ceil(self.Lr / self.dx_0 - self.L_f_lr / self.dx_0)) + self.n_f_lr

                post_wg_wall = np.concatenate([
                    self.alpha ** np.arange(self.n_f_twg, 0, -1) * self.t_m,
                    self.alpha ** np.arange(0, self.n_f_lr + 1) * self.t_m,
                ])
            else:
                self.L_f_wg_Lr, self.n_f_wg_Lr = self._L_and_n_fine(self.dx_0, self.dx_0 / np.sqrt(self.eps_core), self.alpha)
                self.n_Lr = int(np.ceil(self.Lr / self.dx_0 - self.L_f_wg_Lr / self.dx_0)) + self.n_f_wg_Lr

                post_wg_wall = self.alpha ** np.arange(1, self.n_f_wg_Lr + 1) * self.dx_0 / np.sqrt(self.eps_core)
            
            self.dx = np.concatenate([
                np.full(self.n_Ll, self.Ll / self.n_Ll),
                np.full(int(self.n_d - self.n_f_dt), (self.d - self.L_f_dt) / (self.n_d - self.n_f_dt)),
                self.alpha ** np.arange(self.n_f_dt, -1, -1) * self.t_m,
                self.alpha ** np.arange(1, self.n_f_twg + 1) * self.t_m,
                np.full(int(self.n_wg - self.n_f_twg), (self.L_wg - self.L_f_twg) / (self.n_wg - self.n_f_twg)),
                post_wg_wall,
                np.full(int(self.n_Lr - (self.n_f_lr if self.double_wall else self.n_f_wg_Lr)),
                        (self.Lr - (self.L_f_lr if self.double_wall else self.L_f_wg_Lr)) /
                        (self.n_Lr - (self.n_f_lr if self.double_wall else self.n_f_wg_Lr)))
            ])
        elif self.grid_refinement == "step":
            self.n_Ll = int(np.ceil(self.Ll / self.dx_0))
            self.n_d = int(np.ceil(self.d / self.dx_0))
            self.n_wg = int(np.ceil(self.L_wg / (self.dx_0 / np.sqrt(self.eps_core))))
            self.n_Lr = int(np.ceil(self.Lr / self.dx_0))
            self.dx = np.concatenate([
                np.full(self.n_Ll, self.Ll / self.n_Ll),
                np.full(self.n_d, self.d / self.n_d),
                np.array([self.t_m]),
                np.full(self.n_wg, (self.L_wg / self.n_wg)),
                np.array([self.t_m]) if self.double_wall else np.array([], dtype=float),
                np.full(self.n_Lr, self.Lr / self.n_Lr)
            ])
        else:
            raise ValueError(f"Unknown grid_refinement mode: {self.grid_refinement}")

        if self.n_pml > 0:
            left_dx = np.full(self.n_pml, self.dx[0])
            right_dx = np.full(self.n_pml, self.dx[-1])
            self.dx = np.concatenate([left_dx, self.dx, right_dx])
            
        if len(self.dx)%2 == 0:
            self.dx = np.concatenate([self.dx, [self.dx[-1]]])

    def _build_dy(self):
        self.deps_max = getattr(self, "deps_max", 0.1) # percentage of (self.eps_core - self.eps_clad)
        self.dy_0 = self.lam_c / self.finesse
        if not getattr(self, "grid_refinement",True):
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
        elif self.grid_refinement == "gradual":    
            self.L_f_ac, self.n_f_ac = self._L_and_n_fine(self.dy_0, self.dy_0 / np.sqrt(self.eps_clad), self.alpha, max_length=self.w_air)
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
                
            if self.wg_type != "step":
                left_air_coarse = air_coarse
                left_air_taper = air_taper
                right_air_taper = air_taper[::-1]
                right_air_coarse = air_coarse
            else:
                air_coarse_count = max(0, int(self.n_air - self.n_f_ac))
                rem_air_step = max(self.w_air - self.L_f_ac, 0.0)
                if air_coarse_count > 0:
                    coarse_val = rem_air_step / air_coarse_count
                    left_air_coarse = np.full(air_coarse_count, coarse_val)
                    right_air_coarse = np.full(air_coarse_count, coarse_val)
                else:
                    left_air_coarse = np.array([], dtype=float)
                    right_air_coarse = np.array([], dtype=float)

                left_air_taper = self.alpha ** np.arange(self.n_f_ac, 0, -1) * self.dy_0 / np.sqrt(self.eps_clad)
                right_air_taper = self.alpha ** np.arange(1, self.n_f_ac + 1) * self.dy_0 / np.sqrt(self.eps_clad)

            self.dy = np.concatenate([
                left_air_coarse,
                left_air_taper,
                *dy_mid,
                right_air_taper,
                right_air_coarse
            ])
        elif self.grid_refinement == "step":
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
        else:
            raise ValueError(f"Unknown grid_refinement mode: {self.grid_refinement}")

        if self.n_pml > 0:
            left_dy = np.full(self.n_pml, self.dy[0])
            right_dy = np.full(self.n_pml, self.dy[-1])
            self.dy = np.concatenate([left_dy, self.dy, right_dy])
        
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

    def _build_y_region_spans(self, physical=True):
        """Return shaded y-intervals for the full left-to-right stack."""
        edges = self.y_nodes_phys if physical else self.y_nodes
        pml = int(getattr(self, "n_pml", 0))
        n_air = int(getattr(self, "n_air", 0))
        n_clad = int(getattr(self, "n_clad", 0))
        n_core = int(getattr(self, "n_core", 0))

        sequence = [
            ("PML", pml, "gray", 0.08),
            ("Air", n_air, "#9fd3c7", 0.10),
            ("Cladding", n_clad, "yellow", 0.12),
            ("Core", n_core, "blue", 0.12),
            ("Cladding", n_clad, "yellow", 0.12),
            ("Air", n_air, "#9fd3c7", 0.10),
            ("PML", pml, "gray", 0.08),
        ]

        spans = []
        cursor = 0
        seen_labels = set()

        for label, count, color, alpha in sequence:
            next_cursor = cursor + count
            if next_cursor > len(edges) - 1:
                break
            if count > 0:
                spans.append({
                    "left": float(edges[cursor]),
                    "right": float(edges[next_cursor]),
                    "color": color,
                    "alpha": alpha,
                    "label": label if label not in seen_labels else None,
                })
                seen_labels.add(label)
            cursor = next_cursor

        return spans
    
    def _setup_waveguide(self):
        if self.free_space_sim:
            return
        x_shift = int(self.n_pml)
        y_shift = int(self.n_pml)
        self.epsilon_r[int(self.n_Ll + self.n_d + 1 + x_shift):int(self.n_Ll + self.n_d + 1 + x_shift + self.n_wg), int(self.n_air + y_shift):int(- (self.n_air + y_shift))] = self.eps_clad
        if self.wg_type == "step":
            self.epsilon_r[int(self.n_Ll + self.n_d + 1 + x_shift):int(self.n_Ll + self.n_d + 1 + x_shift + self.n_wg), int(self.n_air + self.n_clad + y_shift):int(- (self.n_air + self.n_clad + y_shift))] = self.eps_core
        else:
            core_slice = (slice(int(self.n_Ll + self.n_d + 1 + x_shift), int(self.n_Ll + self.n_d + 1 + x_shift + self.n_wg)),
                          slice(int(self.n_air + self.n_clad + y_shift), int(- (self.n_air + self.n_clad + y_shift))))
            if hasattr(self, "a_eps") and hasattr(self, "b_eps"):
                center_y = self.pml_offset_y + self.W / 2
                eps_val = lambda y: self.eps_core + self.a_eps * (y - center_y)**2 + self.b_eps * (y - center_y)**4
                y_nodes = np.concatenate(([0], np.cumsum(self.dy)))
                core_start = int(self.n_air + self.n_clad + y_shift)
                core_stop = int(self.n_air + self.n_clad + y_shift + self.n_core) + 1
                self.epsilon_r[core_slice] = eps_val(y_nodes[core_start:core_stop])
            else:
                self.epsilon_r[core_slice] = self.eps_core
            
        self.sigma[int(self.n_Ll + self.n_d - 1 + x_shift),   :int(0.4*self.n_core + self.n_air + self.n_clad + y_shift)] = self.sigma_wall
        self.sigma[int(self.n_Ll + self.n_d - 1 + x_shift), - int(0.4*self.n_core + self.n_air + self.n_clad + y_shift):] = self.sigma_wall
        if self.double_wall:
            second_wall_x = int(self.n_Ll + self.n_d + self.n_wg + x_shift)
            self.sigma[second_wall_x,   :int(0.4*self.n_core + self.n_air + self.n_clad + y_shift)] = self.sigma_wall
            self.sigma[second_wall_x, - int(0.4*self.n_core + self.n_air + self.n_clad + y_shift):] = self.sigma_wall

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
        p, m = int(self.cfg.n_pml), getattr(self.cfg, "m_pml", 3)
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

        sub_v = cfg.v_local[1:-1, 1:-1]
        sub_z = cfg.Z_local[1:-1, 1:-1]
        sub_gamma = cfg.gamma[1:-1, 1:-1]
        sub_sigma = cfg.sigma[1:-1, 1:-1]

        self.ap = 2.0 * sub_gamma / cfg.dt + 1.0
        self.am = 2.0 * sub_gamma / cfg.dt - 1.0
        self.coef_n = (1.0 / (sub_v * cfg.dt) - sub_z * sub_sigma / (2.0 * self.ap))
        self.coef_p = (1.0 / (sub_v * cfg.dt) + sub_z * sub_sigma / (2.0 * self.ap))
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
        self.Jc[ix, iy] = (self.am * self.Jc[ix, iy] + cfg.sigma[ix, iy] * cfg.Z_local[ix, iy] * avg_Ez_ddot) / self.ap

        
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

        p, m = int(self.cfg.n_pml), getattr(self.cfg, "m_pml", 3)
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
            self.lhs = self.LHS.tocsc()
            if self.cfg.multi:
                sp.factorize(self.lhs)
            else:
                self.solve_func = spla.factorized(self.LHS.tocsc())
        else:
            print("Pre-factoring system (Schur complement)...")

            nx_n, ny_n = self.nx_n, self.ny_n
            N        = self.len_ez                       # ez-block size = nx_n * ny_n
            len_hx   = self.len_hx
            len_hy   = self.len_hy
            top_len  = 2*len_hx + 2*len_hy

            # Diagonal block inverses (L11, L33, L55, L66, L88 are all sp.diags)
            L11_inv = diag(1.0 / L11.diagonal())
            L33_inv = diag(1.0 / L33.diagonal())
            L55_inv = diag(1.0 / L55.diagonal())
            L66_inv = diag(1.0 / L66.diagonal())
            L88_inv = diag(1.0 / L88.diagonal())

            # L22 = kron(Ix, Ay) @ diag(byp). In row-major ordering this is
            # block-diagonal with nx_n blocks of size ny_n. The PBC corner of Ay
            # lives *inside* each block, not between them, so block-diagonality
            # is preserved. Each block is dense ny_n x ny_n: trivial to invert.
            Ay_dense = Ay.toarray()
            byp_2d   = byp.reshape(nx_n, ny_n)
            inv_blocks_22 = [np.linalg.inv(Ay_dense * byp_2d[i, :][np.newaxis, :])
                             for i in range(nx_n)]
            L22_inv = sp.block_diag(inv_blocks_22, format='csc')

            # L44 = kron(Ax, Iy) @ diag(bz). The PBC corner in Ax escapes each
            # row-major block, so we permute to column-major where the kron
            # identity P @ kron(Ax, Iy) @ P.T = kron(Iy, Ax) restores block-
            # diagonality. We get ny_n blocks of size nx_n.
            Ax_dense = Ax.toarray()
            bz_2d    = bz.reshape(nx_n, ny_n)
            inv_blocks_44 = [np.linalg.inv(Ax_dense * bz_2d[:, j][np.newaxis, :])
                             for j in range(ny_n)]
            L44_inv_perm = sp.block_diag(inv_blocks_44, format='csc')

            perm_idx = np.arange(N).reshape(nx_n, ny_n).T.ravel()
            P = sp.csc_matrix((np.ones(N), (np.arange(N), perm_idx)), shape=(N, N))
            L44_inv = (P.T @ L44_inv_perm @ P).tocsc()

            if getattr(self.cfg, 'verify_schur', False):
                rng = np.random.default_rng(0)
                x = rng.standard_normal(N)
                err22 = np.linalg.norm(L22 @ (L22_inv @ x) - x)
                err44 = np.linalg.norm(L44 @ (L44_inv @ x) - x)
                print(f"  L22_inv residual: {err22:.2e}, L44_inv residual: {err44:.2e}")

            # Outer Schur eliminates the (hx, hx_dot, hy, hy_dot) block. The
            # only nonzero of (M_BL @ M11_inv @ M_TR) lives at the (ez_ddot, ez)
            # position and equals -K, so the outer-Schur correction adds K to
            # that block:
            K = (L71 @ L11_inv @ L12 @ L22_inv @ L25
               + L73 @ L33_inv @ L34 @ L44_inv @ L45).tocsc()

            # Two further Schur eliminations on the ez/ez_dot/jz blocks
            # collapse to a single N x N system:
            #   S2 = L77 + K @ L55_inv @ L56 @ L66_inv @ L67 - L78 @ L88_inv @ L87
            S2 = (L77
                + K @ L55_inv @ L56 @ L66_inv @ L67
                - L78 @ L88_inv @ L87).tocsc()
            print(f"  Factoring S2: {S2.shape[0]} x {S2.shape[1]}, nnz={S2.nnz}")
            S2_factor = spla.factorized(S2)

            def apply_M11_inv(b):
                b1 = b[:len_hx]
                b2 = b[len_hx:2*len_hx]
                b3 = b[2*len_hx:2*len_hx + len_hy]
                b4 = b[2*len_hx + len_hy:2*len_hx + 2*len_hy]
                u2 = L22_inv @ b2
                u1 = L11_inv @ (b1 - L12 @ u2)
                u4 = L44_inv @ b4
                u3 = L33_inv @ (b3 - L34 @ u4)
                return np.concatenate([u1, u2, u3, u4])

            # S_TL = [[L55, L56], [0, L66]] (upper triangular block)
            def apply_S_TL_inv(b):
                b_ez    = b[:N]
                b_ezdot = b[N:]
                u_ezdot = L66_inv @ b_ezdot
                u_ez    = L55_inv @ (b_ez - L56 @ u_ezdot)
                return np.concatenate([u_ez, u_ezdot])

            # S_inner = [[L77+correction, L78], [L87, L88]]; eliminate L88 row
            def apply_S_inner_inv(b):
                b1 = b[:N]
                b2 = b[N:]
                v1 = S2_factor(b1 - L78 @ (L88_inv @ b2))
                v2 = L88_inv @ (b2 - L87 @ v1)
                return np.concatenate([v1, v2])

            # S_BL is zero everywhere except the (ez_ddot, ez) block (= K),
            # S_TR is zero everywhere except the (ez_dot, ez_ddot) block (= L67).
            # Inline their actions to skip the trivial zero rows.
            def apply_S_inv(b):
                b_top = b[:2*N]      # ez, ez_dot
                b_bot = b[2*N:]      # ez_ddot, jz
                x_top_tmp = apply_S_TL_inv(b_top)
                rhs_bot = np.empty(2*N)
                rhs_bot[:N] = b_bot[:N] - K @ x_top_tmp[:N]
                rhs_bot[N:] = b_bot[N:]
                u_bot = apply_S_inner_inv(rhs_bot)
                rhs_top = np.empty(2*N)
                rhs_top[:N] = b_top[:N]
                rhs_top[N:] = b_top[N:] - L67 @ u_bot[:N]
                u_top = apply_S_TL_inv(rhs_top)
                return np.concatenate([u_top, u_bot])

            M_TR = self.LHS[:top_len, top_len:].tocsc()
            M_BL = self.LHS[top_len:, :top_len].tocsc()

            def schur_solve(b):
                b_top = b[:top_len]
                b_bot = b[top_len:]
                u_bot = apply_S_inv(b_bot - M_BL @ apply_M11_inv(b_top))
                u_top = apply_M11_inv(b_top - M_TR @ u_bot)
                return np.concatenate([u_top, u_bot])

            self.solve_func = schur_solve
        

    def step(self, t):
        b = self.RHS.dot(self.u)

        src_val = self.cfg.A * np.cos(2 * np.pi * self.cfg.f_c * (t - self.cfg.t0)) * \
                  np.exp(-0.5 * ((t - self.cfg.t0) / self.cfg.sig_t)**2)

        dx = self.cfg.dx[self.cfg.x0]
        dy = self.cfg.dy[self.cfg.y0]
        src_idx = self.idx_ez.start + self.cfg.x0 * self.ny_n + self.cfg.y0
        b[src_idx] -= src_val * dx * dy
        if self.cfg.multi:
            self.u = ps.solve(self.lhs,b)
        else:
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
        recorder_start_full = np.zeros((config.nt, config.ny), dtype=np.float32)
        
        if "all" in recorders or "diag" in recorders:
            rec_diag = np.zeros(config.nt, dtype=np.float32)
        if "all" in recorders or "after" in recorders:
            rec_after = np.zeros(config.nt, dtype=np.float32)
        if "all" in recorders or "start" in recorders:
            rec_start = np.zeros(config.nt, dtype=np.float32)
        
        frame_idx = 0
        for it in tqdm(range(config.nt), desc=f"Simulating (nt={config.nt}, solver={config.solver_type})"):
            t = it * config.dt
            solver.step(t)
            
            recorder_full[it, :] = solver.Ez[config.x1, :]
            recorder_start_full[it, :] = solver.Ez[config.x_start, :]
            
            if "all" in recorders or "diag" in recorders:
                rec_diag[it] = solver.Ez[config.x_diag, config.y_diag]
            if "all" in recorders or "after" in recorders:
                rec_after[it] = solver.Ez[config.x_after, config.y_after]
            if "all" in recorders or "start" in recorders:
                rec_start[it] = solver.Ez[config.x_start, config.y_start]
            
            if it % frame_skip == 0 and frame_idx < n_frames:
                field_history[frame_idx] = solver.Ez
                recorder_plane[frame_idx, :] = solver.Ez[config.x1, :]
                frame_idx += 1
        result = {
                "config": config,
                "history": field_history,
                "recorder": recorder_plane,
                "recorder_full": recorder_full,
                "recorder_start_full": recorder_start_full,
                "frame_skip": frame_skip,
                "times": np.arange(config.nt) * config.dt,
                "z_norm": getattr(solver, 'Z_local', config.Z_local)
            }

        if "all" in recorders or "diag" in recorders:
            result["rec_diag"] = rec_diag

        if "all" in recorders or "after" in recorders:
            result["rec_after"] = rec_after

        if "all" in recorders or "start" in recorders:
            result["rec_start"] = rec_start

        return result

# ==============================================================================
# 4. Simulation Analyzer
# ==============================================================================
class SimulationAnalyzer:
    @staticmethod
    def _y_axis_with_layers(cfg, physical=True):
        y_nodes = getattr(cfg, "y_nodes_phys" if physical else "y_nodes", None)
        if y_nodes is None:
            y_nodes = np.concatenate(([0.0], np.cumsum(cfg.dy)))
            if physical:
                y_nodes = y_nodes - float(np.sum(cfg.dy[: int(getattr(cfg, "n_pml", 0))]))
        return y_nodes, getattr(cfg, "y_region_spans_phys" if physical else "y_region_spans_index", [])

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
        if "all" in recorders or "start" in recorders:
            process_point("Obs start", results["rec_start"], cfg.x_start, cfg.y_start)

        return results["_hankel_data"]

    @staticmethod
    def _shade_invalid_frequency_regions(ax, f_valid, f_min, f_break, alpha=0.08):
        positive_freqs = f_valid[np.isfinite(f_valid) & (f_valid > 0)]
        if positive_freqs.size == 0:
            return

        left_edge = float(np.min(positive_freqs))
        right_edge = float(np.max(positive_freqs))

        if f_min > left_edge:
            ax.axvspan(left_edge, f_min, color='red', alpha=alpha, lw=0)
        if f_break < right_edge:
            ax.axvspan(f_break, right_edge, color='red', alpha=alpha, lw=0)

    @staticmethod
    def _compare_hankel_verification(results_list):
        base_cfg = results_list[0]["config"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        source_plotted = False
        analytical_plotted = False
        y_max = 0

        ordered_results = []
        for i, res in enumerate(results_list):
            cfg = res["config"]
            label = getattr(cfg, "label", cfg.solver_type.upper())
            ordered_results.append(("hankel verification" in label.lower(), i, res, label))
        ordered_results.sort(key=lambda item: (item[0], item[1]))

        for _, original_index, res, label in ordered_results:
            hd = SimulationAnalyzer._compute_hankel_data(res)
            cfg = res["config"]
            f_valid = hd["f_valid"]
            idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))

            if not source_plotted:
                axes[0].plot(f_valid, np.abs(hd["J_src_valid"]) / np.abs(hd["J_src_valid"][idx_fc]), label='Source', lw=2, color='lightgray')
                source_plotted = True

            if "Recorder Full" not in hd["obs_points"]:
                continue

            label = getattr(cfg, "label", cfg.solver_type.upper())
            data_main = hd["obs_points"]["Recorder Full"]
            H_sim_c = data_main["H_sim"]
            H_anal = data_main["H_analytical"]
            norm_sim = np.abs(H_sim_c[idx_fc])
            norm_anal = np.abs(H_anal[idx_fc])
            color = colors[original_index]
            
            if not analytical_plotted:
                axes[0].plot(f_valid, np.abs(H_anal) / norm_anal, '--', label='Analytical', lw=2, color='black')
                axes[1].plot(f_valid, np.unwrap(np.angle(H_anal)), '--', label='Analytical', lw=2, color='black')
                analytical_plotted = True
                
            axes[0].plot(f_valid, np.abs(H_sim_c) / norm_sim, label=label, lw=2, color=color)
            axes[1].plot(f_valid, np.unwrap(np.angle(H_sim_c)), label=label, lw=2, color=color)

            

            y_max = max(y_max, np.max(np.abs(H_sim_c[f_valid < cfg.f_break]) / norm_sim))

        for ax in axes:
            SimulationAnalyzer._shade_invalid_frequency_regions(ax, f_valid, base_cfg.f_min, base_cfg.f_break)
            ax.axvline(base_cfg.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        axes[0].set_ylim(0, 1.3 * y_max if y_max > 0 else 1.0)
        axes[0].set_title(f'Magnitude Response [{label} vs comparison]')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Normalized Magnitude')
        axes[1].set_title(f'Phase Response [{label} vs comparison]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (rad)')
        axes[0].set_xlim(f_valid[0], f_valid[-1])
        axes[1].set_xlim(f_valid[0], f_valid[-1])
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

        ordered_results = []
        for i, res in enumerate(results_list):
            cfg = res["config"]
            label = getattr(cfg, "label", cfg.solver_type.upper())
            ordered_results.append(("hankel verification" in label.lower(), i, res, label))
        ordered_results.sort(key=lambda item: (item[0], item[1]))

        for _, original_index, res, label in ordered_results:
            hd = SimulationAnalyzer._compute_hankel_data(res)
            cfg = res["config"]
            f_valid = hd["f_valid"]
            idx_fc = np.argmin(np.abs(f_valid - cfg.f_c))
            color = colors[original_index]

            if "Recorder Full" not in hd["obs_points"]:
                continue

            data = hd["obs_points"]["Recorder Full"]
            H_sim, H_anal = data["H_sim"], data["H_analytical"]
            sim_mag = np.abs(H_sim) / np.abs(H_sim[idx_fc])
            anal_mag = np.abs(H_anal) / np.abs(H_anal[idx_fc])
            eps = np.finfo(float).eps
            rel_mag_error = np.abs(sim_mag - anal_mag) / np.maximum(anal_mag, eps)

            mask = (f_valid > cfg.f_min) & (f_valid < cfg.f_break)
            if np.any(mask):
                max_val = np.max([max_val, np.max(rel_mag_error[mask])])

            rms_rel_mag_error = np.sqrt(np.mean(rel_mag_error[mask]**2)) if np.any(mask) else 0
            print(f"{label}: RMS Relative MagnitudeError in valid band = {np.round(rms_rel_mag_error*100, 2)}%")

            phase_diff = np.unwrap(np.angle(H_sim)) - np.unwrap(np.angle(H_anal))
            phase_diff -= phase_diff[idx_fc]
            phase_error_deg = np.rad2deg(phase_diff)
            
            rms_phase_error = np.sqrt(np.mean(phase_error_deg[mask]**2)) if np.any(mask) else 0
            print(f"{label}: RMS Phase Error in valid band = {np.round(rms_phase_error, 2)} degrees")

            if np.any(mask):
                min_phase_error = np.min([min_phase_error, np.min(phase_error_deg[mask])])
                max_phase_error = np.max([max_phase_error, np.max(phase_error_deg[mask])])

            axes[0].plot(f_valid, rel_mag_error * 100, label=label, lw=1.5, color=color)
            axes[1].plot(f_valid, phase_error_deg, label=label, lw=1.5, color=color)

        for ax in axes:
            SimulationAnalyzer._shade_invalid_frequency_regions(ax, f_valid, base_cfg.f_min, base_cfg.f_break)
            ax.axvline(base_cfg.f_c, color='blue', ls=':', alpha=0.5, label=f'f_c')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        axes[0].set_title('Relative Magnitude Error [comparison]')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Relative Error (%)')
        axes[0].set_ylim(0, 300 * max_val if max_val > 0 else 1.0)
        axes[0].set_xlim(f_valid[0], f_valid[-1])
        axes[1].set_title('Phase Error Dispersion [comparison]')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase Error (Degrees)')
        axes[1].set_ylim(1.5 * min_phase_error, 1.5 * max_phase_error if max_phase_error != 0 else 1.0)
        axes[1].set_xlim(f_valid[0], f_valid[-1])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_2d_animation(results, fps=40, color_sensitivity=0.01):
        cfg = results["config"]
        hist = results["history"]
        nodes_x = np.concatenate(([0], np.cumsum(cfg.dx)))[:cfg.nx]
        nodes_y = np.concatenate(([0], np.cumsum(cfg.dy)))[:cfg.ny]
        X, Y = np.meshgrid(nodes_x, nodes_y)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        
        label = getattr(cfg, "label", cfg.solver_type.upper())
        
        z_norm = np.array(results["z_norm"]) # Using fetched z_norm
        vmax = np.max(np.abs(hist * z_norm)) * color_sensitivity
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
        if "all" in recorders or "start" in recorders:
            scat5 = ax.scatter(nodes_x[cfg.x_start], nodes_y[cfg.y_start], color='yellow', s=30, zorder=2, marker='x', label='Obs start')
        
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

            if "all" in recorders or "start" in recorders:
                out.append(scat5)

            return tuple(out)

        interval_ms = max(1, int(1000 / fps))
        ani = FuncAnimation(fig, update, frames=len(hist), interval=interval_ms, blit=True)
        results["ani_2d"] = ani
        ax.set_title(f"Field Animation ({label})")
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
    def plot_grid_spacing(results):
        """Visualize the dx and dy spacing to verify mesh refinement."""
        cfg = results["config"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(cfg.dx, 'o-', markersize=2, label='dx spacing')
        ax1.set_title("X-Grid Spacing (dx)")
        ax1.set_xlabel("Cell Index (i)")
        ax1.set_ylabel("Spacing (m)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(cfg.dy, 'o-', markersize=2, color='orange', label='dy spacing')
        ax2.set_title("Y-Grid Spacing (dy)")
        ax2.set_xlabel("Cell Index (j)")
        ax2.set_ylabel("Spacing (m)")
        ax2.grid(True, alpha=0.3)

        for region in getattr(cfg, "y_region_spans_index", []):
            ax2.axvspan(region["left"], region["right"], color=region["color"], alpha=region["alpha"], label=region["label"])

        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cumulative_energy_flux(*results_list, time_index=-1, use_squared=True, ref=False):
        """Plot the exit time-integrated energy profile normalized per result.

        The plotted curve is the recorder at x=x1 integrated over time and then
        normalized by the total integrated energy entering the waveguide at its
        beginning.
        """
        if len(results_list) == 1 and isinstance(results_list[0], (list, tuple)):
            results_list = tuple(results_list[0])
        if len(results_list) == 0:
            raise ValueError("plot_cumulative_energy_flux requires at least one results dict.")

        first_cfg = results_list[0]["config"]
        first_y_nodes, spans = SimulationAnalyzer._y_axis_with_layers(first_cfg, physical=True)
        first_lam = getattr(first_cfg, "lam_c", 1.0) or 1.0
        first_y_nodes_lambda = first_y_nodes / first_lam

        fig, ax = plt.subplots(figsize=(10, 4))

        for region in spans:
            ax.axvspan(region["left"] / first_lam, region["right"] / first_lam, color=region["color"], alpha=region["alpha"], label=region["label"])
        
        for res in results_list:
            cfg = res["config"]
            y_nodes = getattr(cfg, "y_nodes_phys", None)
            if y_nodes is None:
                y_nodes = np.concatenate(([0], np.cumsum(cfg.dy)))
                y_nodes = y_nodes - float(np.sum(cfg.dy[: int(getattr(cfg, "n_pml", 0))]))
            y = y_nodes[:res["recorder_full"].shape[1]]
            y_lambda = y / (getattr(cfg, "lam_c", 1.0))

            end_rec = res["recorder_full"]
            end_signal = end_rec ** 2 if use_squared else end_rec
            end_profile = np.trapezoid(end_signal, dx=cfg.dt, axis=0)

            end_profile_weighted = end_profile
            
            if ref:
                ref_cfg = ref["config"]
                y_nodes_ref = getattr(ref_cfg, "y_nodes_phys", None)
                if y_nodes_ref is None:
                    y_nodes_ref = np.concatenate(([0], np.cumsum(ref_cfg.dy)))
                    y_nodes_ref = y_nodes_ref - float(np.sum(ref_cfg.dy[: int(getattr(ref_cfg, "n_pml", 0))]))
                y_ref = y_nodes_ref[:ref["recorder_full"].shape[1]]
                
                end_ref = ref["recorder_full"]
                end_signal_ref = end_ref ** 2 if use_squared else end_ref
                end_profile_ref = np.trapezoid(end_signal_ref, dx=ref_cfg.dt, axis=0)
                
                # Interpolate end_profile_ref to match current result's y-axis
                end_profile_ref_interp = np.interp(y, y_ref, end_profile_ref)
                end_profile_weighted = end_profile / end_profile_ref_interp
            
            label = getattr(cfg, "label", cfg.solver_type.upper())
            ax.plot(y_lambda, end_profile_weighted, lw=2, label=f"{label}")

        ax.set_xlabel(r"Relative position $y / \lambda_c$")
        ax.set_ylabel(r"Normalized $\int E_z(y,t)^2\,dt$")
        ax.set_title(r"Cumulative $E_z^2$ Profile at Waveguide Exit")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

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
    
    # Runs a single Yee simulation in free-space (no waveguide core) and plots
    # the grid spacing, the permittivity colormap, and the 2D field animation.
    # Useful as a quick sanity check that the solver and domain are set up correctly.
    Single_test = False
    if Single_test:
        t0 = time.time()
        res = SimulationRunner.execute(
            solver_type="yee",
            L_wg = 0.1,
            w_core=0.1,
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            recorders=["after"],
            label = r"Yee $\lambda/10$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        SimulationAnalyzer.plot_grid_spacing(res)
        SimulationAnalyzer.plot_eps_r_colormap(res)
        SimulationAnalyzer.plot_2d_animation(res)
    
    
    # Sweeps PML thickness (n_pml = 10, 20, 30, 40) and polynomial order (m = 3, 4, 5)
    # in free-space to find the combination that minimises boundary reflections.
    PML_test_Yee = False
    if PML_test_Yee:
        n_list = [10, 20, 30, 40]
        m_list = [3, 4, 5]
        results_by_m = {m: [] for m in m_list}

        for m in m_list:
            for n in n_list:
                t0 = time.time()
                res = SimulationRunner.execute(
                    solver_type="yee",
                    frame_skip=10,
                    finesse=30,
                    n_pml=n,
                    m_pml=m,
                    free_space_sim=True,
                    do_hankel=True,
                    grid_refinement=False,
                    recorders=["after"],
                    label=rf"p={n}, m={m}"
                )
                t1 = time.time()
                print(f"YEE executed in {t1-t0:.2f} seconds.")
                results_by_m[m].append(res)

        # Compare results grouped by `m` (keeps original grouping: m=3 and m=4)
        for m in m_list:
            group = results_by_m[m]
            if len(group) >= 3:
                SimulationAnalyzer.compare_recorders(*group)
            elif group:
                SimulationAnalyzer.compare_recorders(*group)
        
    # Runs the same free-space simulation with both the Yee (FDTD) and FCI (collocated)
    # solvers side-by-side and overlays their recorder time signals for direct comparison.
    # The Hankel reference is enabled to verify numerical dispersion against the analytical solution.
    FCI_vs_YEE = True
    if FCI_vs_YEE:
        t0 = time.time()
        res_yee = SimulationRunner.execute(
            f = 0.5,
            solver_type="yee",
            frame_skip=10,
            finesse=20,
            free_space_sim=True,
            do_hankel=True,
            L_wg = 1,
            alpha = 1.2,
            grid_refinement = 'gradual',
            recorders=["after"],
            label = r"Yee"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_fci = SimulationRunner.execute(
            f = 0.5,
            solver_type = "fci",
            schur = False,
            multi = False,
            frame_skip = 10,
            L_wg = 1,
            finesse = 20,
            free_space_sim = True,
            grid_refinement = 'step',
            do_hankel = True,
            recorders = ["after"],
            label = "FCI"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")

        SimulationAnalyzer.plot_2d_animation(res_fci)
        
        SimulationAnalyzer.plot_2d_animation(res_yee)
        
        SimulationAnalyzer.compare_recorders(res_fci, res_yee)
    
    # Tests Yee solver convergence by running the same free-space problem at three
    # spatial resolutions (finesse = λ/10, λ/20, λ/30) and comparing the recorder outputs.
    Finesse_YEE = False
    if Finesse_YEE:
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
        
        SimulationAnalyzer.plot_2d_animation(res_yee_10)
        SimulationAnalyzer.plot_2d_animation(res_yee_20)
        SimulationAnalyzer.plot_2d_animation(res_yee_30)      
        SimulationAnalyzer.compare_recorders(res_yee_10, res_yee_20, res_yee_30)

    # Same convergence test as Finesse_YEE but for the FCI solver (finesse = λ/10, λ/20, λ/30).
    Finesse_FCI = False
    if Finesse_FCI:
        t0 = time.time()
        res_fci_10 = SimulationRunner.execute(
            solver_type="fci",
            schur = False,
            multi = False,
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
            schur = False,
            multi = False,
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
            multi = False,
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
        SimulationAnalyzer.compare_recorders(res_fci_10, res_fci_20)

    # Runs a single FCI free-space simulation at low resolution (λ/10) to deliberately expose phase velocity errors.
    Phase_error_FCI = False
    if Phase_error_FCI:
        t0 = time.time()
        res_fci_10 = SimulationRunner.execute(
            solver_type="fci",
            schur = False,
            multi = False,
            frame_skip=10,
            finesse=10,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = False,
            label = r"FCI $\lambda/10$"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_fci_10)

        SimulationAnalyzer.compare_recorders(res_fci_10)

    # Compares uniform (step) vs. geometrically graded (gradual) grid stretching for the Yee solver
    # by sweeping the grading ratio α ∈ {2, 1.5, 1.2, 1.05}. 
    Grid_refinement_Yee = False
    if Grid_refinement_Yee:    
        t0 = time.time()
        res_yee_step = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            do_hankel=True,
            grid_refinement = "step",
            recorders=["after"],
            label = r"Step"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_yee_gradual_2 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            alpha = 2,
            do_hankel=True,
            grid_refinement = "gradual",
            recorders=["after"],
            label = r"$\alpha =2$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_yee_gradual_1p5 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            alpha = 1.5,
            do_hankel=True,
            grid_refinement = "gradual",
            recorders=["after"],
            label = r"$\alpha = 1.5$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_yee_gradual_1p2 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            alpha = 1.2,
            do_hankel=True,
            grid_refinement = "gradual",
            recorders=["after"],
            label = r"$\alpha = 1.2$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_yee_gradual_1p05 = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            free_space_sim=True,
            alpha = 1.05,
            do_hankel=True,
            grid_refinement = "gradual",
            recorders=["after"],
            label = r"$\alpha = 1.05$"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
    
        SimulationAnalyzer.plot_2d_animation(res_yee_step)
        SimulationAnalyzer.plot_2d_animation(res_yee_gradual_2)
        SimulationAnalyzer.plot_2d_animation(res_yee_gradual_1p5)
        SimulationAnalyzer.plot_2d_animation(res_yee_gradual_1p2)
        SimulationAnalyzer.plot_2d_animation(res_yee_gradual_1p05)
        SimulationAnalyzer.compare_recorders(res_yee_step, res_yee_gradual_2, res_yee_gradual_1p5, res_yee_gradual_1p2, res_yee_gradual_1p05)

    # Compares a step-index waveguide against a GRIN (graded-index) waveguide using a free-space run as the flux reference. 
    Grin_vs_step_Yee = False
    if Grin_vs_step_Yee:
        t0 = time.time()
        res_reference = SimulationRunner.execute(
            lam_c = 1e-6,
            solver_type = "yee",
            frame_skip = 10,
            finesse = 30,
            free_space_sim = True,
            L_wg = 10,
            grid_refinement = False,
            do_hankel = False,
            recorders = ["after"],
            label = "Reference"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
         
        t0 = time.time()
        res_step = SimulationRunner.execute(
            lam_c = 1e-6,
            solver_type = "yee",
            frame_skip = 10,
            finesse = 30,
            free_space_sim = False,
            L_wg = 10,
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
            lam_c = 1e-6,
            solver_type = "yee",
            deps_max = 0.1,
            frame_skip = 10,
            finesse = 30,
            free_space_sim = False,
            L_wg = 10,
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
        SimulationAnalyzer.plot_cumulative_energy_flux(res_step, res_grin, ref=res_reference)
        SimulationAnalyzer.compare_recorders(res_step, res_grin)

    # Sweeps the carrier wavelength over four decades (λ = 1 m, 100 nm, 1 nm, 1 pm) in a
    # this confirms that the Drude media works at low frequencies
    Wavelength_Sweep_Yee = False
    if Wavelength_Sweep_Yee:
        t0 = time.time()
        res_1m = SimulationRunner.execute(
            lam_c = 1,
            solver_type = "yee",
            frame_skip = 10,
            eps_core = 2.5**2,
            finesse = 20,
            double_wall = False,
            free_space_sim = False,
            grid_refinement = 'gradual',
            wg_type = 'step',
            do_hankel = False,
            recorders = ["after"],
            label = r"$\lambda = 1$ m"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_100nm = SimulationRunner.execute(
            lam_c = 1e-7,
            solver_type = "yee",
            frame_skip = 10,
            eps_core = 2.5**2,
            finesse = 20,
            free_space_sim = False,
            grid_refinement = 'gradual',
            wg_type = 'step',
            do_hankel = False,
            recorders = ["after"],
            label = r"$\lambda = 100$ nm"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_1nm = SimulationRunner.execute(
            lam_c = 1e-9,
            solver_type = "yee",
            frame_skip = 10,
            eps_core = 2.5**2,
            finesse = 20,
            free_space_sim = False,
            grid_refinement = 'gradual',
            wg_type = 'step',
            do_hankel = False,
            recorders = ["after"],
            label = r"$\lambda = 1$ nm"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        t0 = time.time()
        res_1pm = SimulationRunner.execute(
            lam_c = 1e-12,
            solver_type = "yee",
            frame_skip = 10,
            eps_core = 2.5**2,
            finesse = 20,
            free_space_sim = False,
            grid_refinement = 'gradual',
            wg_type = 'step',
            do_hankel = False,
            recorders = ["after"],
            label = r"$\lambda = 1$ pm"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_1m)
        SimulationAnalyzer.plot_2d_animation(res_100nm)
        SimulationAnalyzer.plot_2d_animation(res_1nm)
        SimulationAnalyzer.plot_2d_animation(res_1pm)
        SimulationAnalyzer.plot_cumulative_energy_flux(res_1m, res_100nm, res_1nm, res_1pm)

    # Repeats the FCI vs Yee head to head but with a step-index waveguide
    FCI_vs_YEE_stepwg = False
    if FCI_vs_YEE_stepwg:
        t0 = time.time()
        res_fci_schur = SimulationRunner.execute(
            solver_type = "fci",
            schur = False,
            frame_skip = 10,
            finesse = 10,
            free_space_sim = False,
            grid_refinement = "step",
            wg_type = "step",
            do_hankel = False,
            recorders = ["after"],
            label = "FCI"
        )
        t1 = time.time()
        print(f"FCI executed in {t1-t0:.2f} seconds.")

        SimulationAnalyzer.plot_2d_animation(res_fci_schur)

        t0 = time.time()
        res_yee = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            sigma_wall = 0,
            free_space_sim=False,
            do_hankel=True,
            grid_refinement = "gradual",
            wg_type = "step",
            recorders=["after"],
            label = r"Yee"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_yee)
        
        SimulationAnalyzer.compare_recorders(res_fci_schur, res_yee)
             
    # Tests the Drude dispersive-medium extension of the Yee solver at λ = 10 nm
    Drude_test = False
    if Drude_test:
        t0 = time.time()
        res_drude = SimulationRunner.execute(
            solver_type="yee",
            frame_skip=10,
            finesse=30,
            eps_core=1,
            eps_clad=1,
            w_core = 0.1,
            free_space_sim=False,
            grid_refinement = "gradual",
            wg_type = "step",
            do_hankel=True,
            recorders=["after"],
            lam_c = 1e-8,
            label = r"Yee with Drude"
        )
        t1 = time.time()
        print(f"YEE executed in {t1-t0:.2f} seconds.")
        
        SimulationAnalyzer.plot_2d_animation(res_drude)
        SimulationAnalyzer.compare_recorders(res_drude)