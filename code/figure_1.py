# Reproduces figure 1: demonstrates the need for multiple grids in
# resolving the solution via characteristics to
#
#  u_t + a(x) u_x = 0
#
# on x \in [-1,1] with periodic boundary conditions.

export_pdf = False

import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from utils import lserk, lserk_step

# Wavespeed
a = lambda xx: -np.sin(np.pi*xx)*(1 - np.sin(np.pi*xx)**2)

# Initial data
u0 = lambda xx: np.sin(4*np.pi*xx)

T = 1

# Choose dt for stability
amax = np.max(np.abs(a(np.linspace(-1,1,1e3))))
dt = 1e-2/amax
N = np.ceil(T/dt)
dt = T/N

N = 100
# Particles at t=0
x0 = np.linspace(-1, 1, N+1)[:-1]
x0 += (x0[1]-x0[0])/2.

# Particles at t=T
yT = x0.copy()

# Evolve x particles to time T
f = lambda tt, yy: a(yy)
xT = lserk_step(0, x0, f, dt, N, lserk)

# Evolve y particles back to time 0
f = lambda tt, yy: -a(yy)
y0 = lserk_step(0, yT, f, dt, N, lserk)

rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'semibold'
rcParams['mathtext.fontset'] = 'dejavuserif'
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.plot(xT, u0(x0), 'r.-')
plt.xlim(-1,1)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Only $X_j$ particles')

plt.subplot(2, 3, 2)
plt.plot(yT, u0(y0), 'r.-')
plt.xlim(-1,1)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Only $Y_j$ particles')

plt.subplot(2, 3, 3)
ordering = np.argsort(np.hstack([xT, yT]))
plt.plot(np.hstack([xT,yT])[ordering], u0(np.hstack([x0,y0])[ordering]), 'r.-')
plt.xlim(-1,1)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Both $X_j$ and $Y_j$ particles')

axinset = (-1e-3, 1e-3)

plt.subplot(2, 3, 4)
plt.plot(xT, u0(x0), 'r.-')
plt.xlim(axinset)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Only $X_j$ particles')
plt.xticks([-1e-3, 0, 1e-3])

plt.subplot(2, 3, 5)
plt.plot(yT, u0(y0), 'r.-')
plt.xlim(axinset)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Only $Y_j$ particles')
plt.xticks([-1e-3, 0, 1e-3])

plt.subplot(2, 3, 6)
ordering = np.argsort(np.hstack([xT, yT]))
plt.plot(np.hstack([xT,yT])[ordering], u0(np.hstack([x0,y0])[ordering]), 'r.-')
plt.xlim(axinset)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Both $X_j$ and $Y_j$ particles')
plt.xticks([-1e-3, 0, 1e-3])

plt.subplots_adjust(wspace = 0.4, hspace=0.5)

if export_pdf:
    plt.savefig(fname="../particle-plot.pdf", format='pdf')

plt.show()
