import scipy.sparse as sp
import numpy as np
d = np.array([1,1,1,1,1,1])
n = 6
inv_d = sp.diags(1.0 / d)

D1_core = sp.diags([1, -1], [0, -1], shape=(n, n-1))
D2_core = sp.diags([-1, 1], [0, 1], shape=(n, n+1))
D1 = inv_d @ D1_core
D2 = inv_d @ D2_core

# Convert and print the dense version
print(D2.toarray())