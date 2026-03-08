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

dx = lambda_0 / 30
dy = lambda_0 / 30
# Constants
eps0 = 8.854e-12
mu0 = 4*np.pi*1e-7
c = 1/np.sqrt(eps0*mu0)

dt = 2/(c*np.sqrt(1/dx**2 + 1/dy**2))
f_c = c/lambda_0
sig_t = a / (2 * np.pi * f_c)
t0 = 4 * sig_t
src = lambda t:5 * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/sig_t)**2)
eps_val, mu_val, sigma_val = 








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



len_hx = (Nx + 1) * Ny
len_hy = Nx * (Ny + 1)
len_ez = (Nx + 1) * (Ny + 1)

# --- 2. Material Matrices (Sigma now applied to Ez blocks) ---
Mxx = (mu_val/dt) * sp.eye(len_hx)
Myy = (mu_val/dt) * sp.eye(len_hy)
Ezz_p = (eps_val/dt + sigma_val/2) * sp.eye(len_ez)
Ezz_m = (eps_val/dt - sigma_val/2) * sp.eye(len_ez)

L11 = sp.kron(Ix, Ay) @ Mxx
L12 = None
L13 = sp.kron(Ix, Dy)           

L21 = None
L22 = sp.kron(Ax, Iy) @ Myy
L23 = -sp.kron(Dx, Iy)          

L31 = sp.kron(Atx, Dty)      
L32 = -sp.kron(Dtx, Aty)        
L33 = sp.kron(Atx, Aty) @ Ezz_p

LHS = sp.bmat([[L11, L12, L13],
               [L21, L22, L23],
               [L31, L32, L33]], format='csr')

R11 = sp.kron(Ix, Ay) @ Mxx
R12 = None
R13 = -sp.kron(Ix, Dy)

R21 = None
R22 = sp.kron(Ax, Iy) @ Myy
R23 = sp.kron(Dx, Iy)

R31 = -sp.kron(Atx, Dty)
R32 = sp.kron(Dtx, Aty)
R33 = sp.kron(Atx, Aty) @ Ezz_m

RHS = sp.bmat([[R11, R12, R13],
               [R21, R22, R23],
               [R31, R32, R33]], format='csr')

u = np.zeros(len_hx + len_hy + len_ez)











# --- 5. Solver and Plotter Setup ---
idx_ez = slice(len_hx + len_hy, len_hx + len_hy + len_ez)
ez_rec = np.zeros(Nt)
fig, ax = plt.subplots()
movie = []

# Define source location in the Ez grid (center)
src_x, src_y = (Nx + 1) // 2, (Ny + 1) // 2
ez_source_global_idx = len_hx + len_hy + (src_x * (Ny + 1) + src_y)

# Optimization: Pre-factorize for a 100x improvement in loop speed
solve_func = spla.factorized(LHS)

for i in range(Nt):
    t = i * dt
    b = RHS.dot(u)
    b[ez_source_global_idx] += src(t)
    
    # Solve system using pre-factorized matrix
    u = solve_func(b)

    # Extract Ez for visualization
    ez_2d = u[idx_ez].reshape((Nx + 1, Ny + 1))
    ez_rec[i] = ez_2d[src_x, src_y]

    if i % 2 == 0:
        txt = ax.text(0.5, 1.05, f'Step: {i}/{Nt}', ha="center", transform=ax.transAxes)
        vlimit = 0.1
        img = ax.imshow(ez_2d.T, animated=True, cmap='RdBu', origin='lower',
                        extent=[0, Nx*dx, 0, Ny*dy],
                        vmin=-vlimit, vmax=vlimit)
        movie.append([txt, img])

    if i % 10 == 0:
        print(f"Iterating: {i}/{Nt}")

ani = ArtistAnimation(fig, movie, interval=50, blit=True)
plt.colorbar(img, ax=ax, label='Ez Field Amplitude')
plt.show()

plt.figure()
plt.plot(np.arange(Nt)*dt, ez_rec)
plt.title("Ez Field at Source Location")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()