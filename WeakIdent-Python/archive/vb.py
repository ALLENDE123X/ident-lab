# save as make_burgers.py in WeakIdent-Python/ and run: python make_burgers.py
import numpy as np

# 1D grid and time
nx, nt = 256, 201
x = np.linspace(-1, 1, nx)
t = np.linspace(0, 1.0, nt)
dx = x[1]-x[0]
dt = t[1]-t[0]
nu = 0.01

# initial condition: smooth bump
u = np.zeros((nt, nx))
u0 = np.exp(-50*(x+0.5)**2) - np.exp(-50*(x-0.5)**2)
u[0] = u0

# simple periodic finite-difference time stepping (explicit)
for n in range(nt-1):
    up = np.roll(u[n], -1)
    um = np.roll(u[n], 1)
    ux  = (up - um) / (2*dx)
    uxx = (up - 2*u[n] + um) / (dx*dx)
    u[n+1] = u[n] + dt*(-u[n]*ux + nu*uxx)

np.save('dataset-Python/burgers_viscous.npy', {'u': u, 'x': x, 't': t})
print('saved to dataset-Python/burgers_viscous.npy')
