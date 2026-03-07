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

nt = 200

# Grid
Nx = 200
Ny = 200
lambda_0 = 1

dx = lambda_0 / 30
dy = lambda_0 / 30

dt = 1/(c*np.sqrt(1/dx**2 + 1/dy**2))


# Init fields
Ez = np.zeros(Nx*Ny).T
Hx = np.zeros(Nx*Ny).T
Hy = np.zeros(Nx*Ny).T

# Diag matrices
x_1 = sp.diags_array([1], shape = (Nx,Nx), dtype = None)
y_1 = sp.diags_array([1], shape = (Ny,Ny), dtype = None)

# Averaging operators
Ax = sp.diags_array(np.ones(3), offsets = np.array([0,1,-Nx+1]), dtype = None, shape = (Nx, Nx))/2
Ay = sp.diags_array(np.ones(3), offsets = np.array([0,1,-Ny+1]), dtype = None, shape = (Ny, Ny))/2

# Difference operators
Dx = sp.diags_array([-1,1,1], offsets = np.array([0,1,-Nx+1]), dtype = None, shape = (Nx, Nx))/dx
Dy = sp.diags_array([-1,1,1], offsets = np.array([0,1,-Ny+1]), dtype = None, shape = (Ny, Ny))/dy

H_curl = sp.bmat([[0, sp.kron(x_1, Ay).multiply(0.5)],
                   [-sp.kron(Ax, y_1), 0]])
E_curl = sp.bmat([[sp.kron(x_1, Dy)],
                    [-sp.kron(Dx, y_1)]])

E_avg = sp.kron(Ax, Ay) * eps0 / dt
H_avg = sp.bmat([[sp.kron(x_1, Ay) * eps0 / dt, 0], 
                 [0,  sp.kron(Ax, y_1) * mu0 / dt ]])

LHS = sp.bmat([[0     , H_curl  ],
               [E_avg , 0       ],
               [E_curl, H_avg   ]])

RHS = sp.bmat([[0       , - H_curl  ],
               [E_avg   , 0         ],
               [- E_curl, H_avg     ]]) 

PrevFields = sp.bmat([[Ez],
                      [Hx],
                      [Hy]])

print(PrevFields.toarray())
recorder = np.zeros()

for _ in range(nt):
    NextFields = spla.spsolve(LHS,RHS * PrevFields)



