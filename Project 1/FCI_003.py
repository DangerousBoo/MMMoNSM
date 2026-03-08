import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import LinAlgError



# Parameters
a = 3
Nt = 150
Nx = 100
Ny = 100
lambda_0 = 1

# Constants
eps0 = 8.854e-12
mu0 = 4*np.pi*1e-7
c = 1/np.sqrt(eps0*mu0)
dx = lambda_0 / 20
dy = lambda_0 / 20
dt = 1/(c*np.sqrt(1/dx**2 + 1/dy**2))
f_c = c/lambda_0
sig_t = a / (2 * np.pi * f_c)
t0 = 4 * sig_t
src = lambda t:5 * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/sig_t)**2)
c_idx = 2*Nx*Ny + (Nx//2)*Ny + (Ny//2)
neighbors = [c_idx, c_idx+1, c_idx-1, c_idx+Ny, c_idx-Ny]
weights = [0.5, 0.125, 0.125, 0.125, 0.125]
eps_val, mu_val, sigma_val = 8.854e-12, 1.256e-6, 0.001








def get_operators(n, d):
    """Constructs the 1D operators from Equation (2)"""
    # Ix: Identity (n+1 x n+1)
    Ix = sp.eye(n + 1, format='csr')
    # I_hat_x: (n+1 x n)
    I_hx = sp.eye(n + 1, n, k=0, format='csr')
    # A_hat_x: (n x n+1)
    A_hx = sp.eye(n, n + 1, k=0) + sp.eye(n, n + 1, k=1)
    # Ax: (n x n)
    Ax = sp.eye(n, n) + sp.eye(n, n, k=1)
    # A_tilde_x: (n+1 x n+1)
    Atx = (sp.eye(n + 1, n + 1) + sp.eye(n + 1, n + 1, k=-1)).tolil()
    Atx[n, n] = 2
    # Dx: (n x n+1)
    Dx = (sp.eye(n, n + 1, k=1) - sp.eye(n, n + 1, k=0)) / d
    # D_tilde_x: (n+1 x n)
    Dtx = (sp.eye(n + 1, n, k=0) - sp.eye(n + 1, n, k=-1)).tolil()
    Dtx[n, n-1] = -2
    return Ix, I_hx, A_hx.tocsr(), Ax.tocsr(), Atx.tocsr(), Dx.tocsr(), Dtx.tocsr() / d

Ix, Ihx, Ahx, Ax, Atx, Dx, Dtx = get_operators(Nx, dx)
Iy, Ihy, Ahy, Ay, Aty, Dy, Dty = get_operators(Ny, dy)


len_ex = (Nx + 1) * Ny
len_ey = Nx * (Ny + 1)
len_hz = (Nx + 1) * (Ny + 1)

Exx_p = (eps_val/dt + sigma_val/2) * sp.eye((Nx+1)*Ny)
Exx_m = (eps_val/dt - sigma_val/2) * sp.eye((Nx+1)*Ny)
Eyy_p = (eps_val/dt + sigma_val/2) * sp.eye(Nx*(Ny+1))
Eyy_m = (eps_val/dt - sigma_val/2) * sp.eye(Nx*(Ny+1))
Mzz   = (mu_val/dt) * sp.eye((Nx+1)*(Ny+1))

L11 = sp.kron(Ix, Ay) @ Exx_p
L12 = sp.kron(Ihx, Ahy) * (sigma_val/2)  
L13 = -sp.kron(Ix, Dy)

L21 = sp.kron(Ahx, Ihy) * (sigma_val/2) 
L22 = sp.kron(Ax, Iy) @ Eyy_p
L23 = sp.kron(Dx, Iy)

L31 = -sp.kron(Atx, Dty)
L32 = sp.kron(Dtx, Aty)
L33 = sp.kron(Atx, Aty) @ Mzz

R11 = sp.kron(Ix, Ay) @ Exx_m
R12 = sp.kron(Ihx, Ahy) * (-sigma_val/2)
R13 = sp.kron(Ix, Dy)

R21 = sp.kron(Ahx, Ihy) * (-sigma_val/2)
R22 = sp.kron(Ax, Iy) @ Eyy_m
R23 = -sp.kron(Dx, Iy)

R31 = sp.kron(Atx, Dty)
R32 = -sp.kron(Dtx, Aty)
R33 = sp.kron(Atx, Aty) @ Mzz

LHS = sp.bmat([[L11, L12, L13],
               [L21, L22, L23],
               [L31, L32, L33]], format='csr')

RHS = sp.bmat([[R11, R12, R13],
               [R21, R22, R23],
               [R31, R32, R33]], format='csr')

u = np.zeros(len_ex + len_ey + len_hz)











# Indices to slice the solution vector u
idx_ex = slice(0, len_ex)
idx_ey = slice(len_ex, len_ex + len_ey)
idx_hz = slice(len_ex + len_ey, len_ex + len_ey + len_hz)

# --- 2. Solver and Plotter Setup ---
hz_rec = np.zeros(Nt)
fig, ax = plt.subplots()
movie = []
# Define source location in the Hz grid (center)
src_x, src_y = (Nx + 1) // 2, (Ny + 1) // 2
# Index in the Hz portion of the vector
hz_source_local_idx = src_x * (Ny + 1) + src_y
hz_source_global_idx = len_ex + len_ey + hz_source_local_idx

for i in range(Nt):
    t = i * dt
    
    # 1. Update Source 
    current_src = src(t)
    
    # 2. Build the B vector (RHS * u + source)
    b = RHS.dot(u)
    b[hz_source_global_idx] += current_src
    
    # 3. Solve Implicit System
    u = spla.spsolve(LHS, b)

    # 4. Extract and Reshape Hz for visualization
    hz_2d = u[idx_hz].reshape((Nx + 1, Ny + 1))
    hz_rec[i] = hz_2d[src_x, src_y]

    # 5. Animation frames
    if i % 2 == 0:
        txt = ax.text(0.5, 1.05, f'Step: {i}/{Nt}', ha="center", transform=ax.transAxes)
        vlimit = 0.000005 
        img = ax.imshow(hz_2d.T, animated=True, cmap='RdBu', origin='lower',
                        extent=[0, Nx*dx, 0, Ny*dy],
                        vmin=-vlimit, vmax=vlimit)
        movie.append([txt, img])

    if i % 10 == 0:
        print(f"Iterating: {i}/{Nt}")

ani = ArtistAnimation(fig, movie, interval=50, blit=True)
plt.colorbar(img, ax=ax, label='Hz Field Amplitude')
plt.show()

plt.figure()
plt.plot(np.arange(Nt)*dt, hz_rec)
plt.title("Hz Field at Source Location")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()