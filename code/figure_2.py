# Reproduces figure 2: convergence study in time for solution via
# characteristics to
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
a = lambda xx: -np.sin(np.pi*xx)

# Initial data
u0 = lambda xx: np.sin(np.pi*xx)

T = 1

# Choose coarsest dt for stability
amax = np.max(np.abs(a(np.linspace(-1,1,1e3))))
dt = 1e-1/amax
N = np.int(np.ceil(T/dt))
dt = T/N

M = 1e3
# Particles at t=T
xT = np.linspace(-1, 1, M+1)[:-1]
xT += (xT[1]-xT[0])/2.

# For convergence study: various values of dt
Ns = np.round(np.logspace(np.log10(N), 2+np.log10(N), 11))
dts = T/Ns
Ns = np.array(Ns, dtype=int)

Nfine = Ns[-1]*10
dtfine = T/Nfine

# Evolve particles back to time 0
f = lambda tt, yy: -a(yy)
x0fine = lserk_step(0, xT, f, dtfine, Nfine, lserk)

# The "exact" solution
ufine = u0(x0fine)

# Solve same problem for different values of dt
errors = np.zeros(dts.shape)
for (i,dt) in enumerate(dts):

    x0 = lserk_step(0, xT, f, dts[i], Ns[i], lserk)
    errors[i] = np.max(np.abs(ufine - u0(x0)))

rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'semibold'

plt.loglog(dts, errors, 'r.-')
# Add a line with slope 4 
xs = np.array([np.min(dts)*4, np.min(dts)*8])
ys = xs**4
ys *= 1e-9/ys[0]
plt.loglog(xs, ys, 'k--')
plt.xlabel(r'Timestep $h$')
plt.title(r'$\Vert u(x,T) - u_h(x,T)\Vert_{L^\infty([-1,1])}$')
plt.gca().annotate(r'Slope$=4$', xy=(np.min(dts)*6, 10*np.min(ys)), horizontalalignment='right')

if export_pdf:
    plt.savefig(fname="../convergence-plot.pdf", format='pdf')

plt.show()
