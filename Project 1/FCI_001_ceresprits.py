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









# Init fields
Ez = np.zeros((Nx,Ny))
Hx = np.zeros((Nx,Ny))
Hy = np.zeros((Nx,Ny))

# Diag matrices
Inx = sp.diags_array([1], shape = (Nx,Nx), dtype = None)
Iny = sp.diags_array([1], shape = (Ny,Ny), dtype = None)

# Averaging operators
Ax = sp.diags_array(np.ones(3), offsets = np.array([0,1,-Nx+1]), dtype = None, shape = (Nx, Nx)).multiply(0.5)
Ay = sp.diags_array(np.ones(3), offsets = np.array([0,1,-Ny+1]), dtype = None, shape = (Ny, Ny)).multiply(0.5)

# Difference operators
Dx = sp.diags_array([-1,1,1], offsets = np.array([0,1,-Nx+1]), dtype = None, shape = (Nx, Nx)).multiply(1/dx)
Dy = sp.diags_array([-1,1,1], offsets = np.array([0,1,-Ny+1]), dtype = None, shape = (Ny, Ny)).multiply(1/dy)

# # Delta matrices
# del_x = sp.diags_array([1], shape = (Nx,Nx), dtype = None).multiply(dx)
# del_y = sp.diags_array([1], shape = (Ny,Ny), dtype = None).multiply(dy)


# A (weight) matrices ONLY FOR SCALAR EPS & MU
M_hx = sp.kron(Inx, Ay).multiply(mu0)   
M_hy = sp.kron(Ax, Iny).multiply(mu0)   
M_ez = sp.kron(Ax, Ay).multiply(eps0)  

M = sp.block_diag((M_hx, M_hy, M_ez), format='csr')


# "K"url matrices fzo
K_hx_ez =  sp.kron(Inx, Dy)
K_hy_ez = -sp.kron(Dx, Iny) 
K_ez_hx = -sp.kron(Ax,  Dy)  
K_ez_hy =  sp.kron(Dx,  Ay)  

K = sp.block_array([
    [None,    None,    K_hx_ez],
    [None   , None,    K_hy_ez],
    [K_ez_hx, K_ez_hy, None   ]
    ], format='csr')

LHS = M/dt + K/2
RHS = M/dt - K/2

u = np.concatenate([Hx.flatten(), Hy.flatten(), Ez.flatten()])



E_rec = np.zeros((Nt))
fig, ax = plt.subplots()
plt.axis('equal')
plt.xlim([1, Nx+1])
plt.ylim([1, Ny+1])
movie = []
current_vector = np.zeros(3 * Nx * Ny)
source_idx = 2*Nx*Ny + (Nx//2)*Ny + (Ny//2)

for i in range(Nt):
    print(f"Iterating: {i}/{Nt}")
    t = i * dt
    
    src_val = (src(t) + src(t + dt)) / 2
    current_vector.fill(0)
    for idx, w in zip(neighbors, weights):
        current_vector[idx] = src_val * w
    b = RHS.dot(u) + current_vector
    u = spla.spsolve(LHS, b)


    Ez_new = u[2*Nx*Ny:].reshape(Nx, Ny)
    E_rec[i] = Ez_new[Nx//2, Ny//2] # Record at source or observation point
    

    txt = ax.text(0.5, 1.05, f'Step: {i}/{Nt}', 
                  size=plt.rcParams["axes.titlesize"],
                  ha="center", transform=ax.transAxes)
    v_max = np.max(np.abs(Ez_new)) if np.max(np.abs(Ez_new)) > 0 else 1
    img = ax.imshow(Ez_new, animated=True, cmap='RdBu', origin='lower', 
                    vmin=-v_max*0.2, vmax=v_max*0.2) # Look at 20% of peak
    movie.append([txt, img])

my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000, blit=True)
plt.show()
plt.plot(E_rec)
plt.show()
