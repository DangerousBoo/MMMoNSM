import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Constants
eps0 = 8.854e-12
mu0 = 4*np.pi*1e-7
c = 1/np.sqrt(eps0*mu0)

nt = 50

# Grid
Nx = 200
Ny = 200
lambda_0 = 1

dx = lambda_0 / 30
dy = lambda_0 / 30

dt = 4/(c*np.sqrt(1/dx**2 + 1/dy**2))

# Source
f_c = c/lambda_0
a = 3
sig_t = a / (2 * np.pi * f_c)
t0 = 4 * sig_t
src = lambda t:5 * np.cos(2*np.pi*f_c*(t-t0)) * np.exp(-0.5*((t-t0)/sig_t)**2)



# Init fields
Ez = np.zeros((Nx,Ny))
Hx = np.zeros((Nx,Ny))
Hy = np.zeros((Nx,Ny))

# Diag matrices
x_1 = sp.diags_array([1], shape = (Nx,Nx), dtype = None)
y_1 = sp.diags_array([1], shape = (Ny,Ny), dtype = None)

# Averaging operators
Ax = sp.diags_array(np.ones(3), offsets = np.array([0,1,-Nx+1]), dtype = None, shape = (Nx, Nx)).multiply(0.5)
Ay = sp.diags_array(np.ones(3), offsets = np.array([0,1,-Ny+1]), dtype = None, shape = (Ny, Ny)).multiply(0.5)

# Difference operators
Dx = sp.diags_array([-1,1,1], offsets = np.array([0,1,-Nx+1]), dtype = None, shape = (Nx, Nx)).multiply(1/dx)
Dy = sp.diags_array([-1,1,1], offsets = np.array([0,1,-Ny+1]), dtype = None, shape = (Ny, Ny)).multiply(1/dy)

H_curl = sp.block_array([[None, sp.kron(x_1, Ay)],
                   [-sp.kron(Ax, y_1), None]])
E_curl = sp.block_array([[sp.kron(x_1, Dy)],
                    [-sp.kron(Dx, y_1)]])

E_avg = sp.kron(Ax, Ay) * eps0 / dt
H_avg = sp.block_array([[sp.kron(x_1, Ay) * eps0 / dt, None], 
                 [None,  sp.kron(Ax, y_1) * mu0 / dt ]])

LHS = sp.block_array([[None     , H_curl  ],
               [E_avg , None       ],
               [E_curl, H_avg   ]])

RHS = sp.block_array([[None       , - H_curl  ],
               [E_avg   , None         ],
               [- E_curl, H_avg     ]]) 



PrevFields = sp.block_array([[sp.csr_array(Ez.reshape(-1,1))],
                             [sp.csr_array(Hx.reshape(-1,1))],
                             [sp.csr_array(Hy.reshape(-1,1))]])

E_rec = np.zeros((nt))



fig, ax = plt.subplots()
plt.axis('equal')
plt.xlim([1, Nx+1])
plt.ylim([1, Ny+1])
movie = []
timeseries = np.zeros((nt,1))

tmax = nt

for i in range(nt):
    print(f"{i}/{nt}")
    t = (i-1)*dt
    timeseries[i, 0] = t
    current = np.zeros((Nx,Ny))
    current[Nx//2,Ny//2] = (src(t+dt)+src(t))/2 
    current = sp.block_array([[sp.csr_array(np.zeros((2*Nx*Ny,1)))],
                              [sp.csr_array(current.reshape(-1,1))],
                              [sp.csr_array(np.zeros((2*Nx*Ny,1)))]])

    NextFields = spla.lsqr(LHS, (RHS.dot(PrevFields.reshape(-1,1)) + current.toarray()))
    PrevFields = NextFields[0]

    E_rec[i] = NextFields[0][Nx*Nx//2+Ny//3]
    artists = [
        ax.text(0.5,1.05,'%d/%d' % (i, nt), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, ),
        ax.imshow(NextFields[0][:Nx*Ny].reshape(Nx,Ny))
        # ax.imshow(p_ref.T, vmin=-0.02*A, vmax=0.02*A),
        ]
    movie.append(artists)



my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                blit=True)
plt.show()

plt.plot(E_rec)
plt.show()
