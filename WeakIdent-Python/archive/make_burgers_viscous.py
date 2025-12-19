import numpy as np

# grid and params
nx = 256
x = np.linspace(-1.0, 1.0, nx)
dx = x[1] - x[0]
nu = 0.01
T_end = 1.0

# initial condition
u0 = np.exp(-50*(x+0.5)**2) - np.exp(-50*(x-0.5)**2)
umax = max(1e-6, np.max(np.abs(u0)))
dt_cfl = 0.4 * min(dx/umax, dx*dx/(2*nu))
nt = int(np.ceil(T_end / dt_cfl)) + 1
t = np.linspace(0.0, T_end, nt)
dt = t[1] - t[0]

u = np.zeros((nt, nx))
u[0] = u0

# Stable Lax-Friedrichs + explicit diffusion, periodic
for n in range(nt - 1):
    u_n = u[n]
    up = np.roll(u_n, -1)
    um = np.roll(u_n,  1)
    f = 0.5 * u_n**2
    fp = 0.5 * up**2
    fm = 0.5 * um**2
    conv = - (dt/(2*dx)) * (fp - fm) + 0.5*(up - 2*u_n + um) * (dt/dx) * 0.0
    diff = nu * dt * (up - 2*u_n + um) / (dx*dx)
    u[n+1] = u_n + conv + diff

# Re-orient to (Nx, Nt) expected by WeakIdent (space first, then time)
u_space_time = u.T  # shape (nx, nt)

# xs should be an object array of [x (Nx,1), t (1,Nt)]
x_col = x.reshape(-1, 1)
t_row = t.reshape(1, -1)

# true_coefficients should be an object array of length n (here n=1),
# where each element is a 2D array of rows [beta_u, d_x, d_t, coef].
# Burgers (viscous): u_t = -u u_x + nu u_xx = -(1/2)(u^2)_x + nu u_{xx}
true_coefs_u = np.array([
    [2., 1., 0., -0.5],   # (u^2)_x with coefficient -1/2 equals -u u_x
    [1., 2., 0.,  nu],    # u_{xx} with viscosity nu
])

# Pack arrays exactly as WeakIdent expects
A1 = np.array([u_space_time], dtype=object)    # shape (1,)
A2 = np.array([x_col, t_row], dtype=object)    # shape (2,)
A3 = np.array([true_coefs_u], dtype=object)    # shape (1,)

out_path = 'dataset-Python/burgers_viscous.npy'
with open(out_path, 'wb') as f:
    np.save(f, A1, allow_pickle=True)
    np.save(f, A2, allow_pickle=True)
    np.save(f, A3, allow_pickle=True)

print('Wrote', out_path)
print('Saved shapes:', A1.shape, A2.shape, A3.shape)
