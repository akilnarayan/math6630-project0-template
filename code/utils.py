# Miscellenous files for solving via characteristics

import numpy as np

# Here's a particular low-storage explicit RK scheme:
lserk = dict()
lserk['a'] = np.array([0.,
                       -567301805773./1357537059087.,
                       -2404267990393./2016746695238.,
                       -3550918686646./2091501179385.,
                       -1275806237668./842570457699.])

lserk['b'] = np.array([1432997174477./9575080441755.,
                       5161836677717./13612068292357.,
                       1720146321549./2090206949498.,
                       3134564353537./4481467310338.,
                       2277821191437./14882151754819.])

lserk['c'] = np.array([0.0,
                       1432997174477.0/9575080441755.0,
                       2526269341429.0/6820363962896.0,
                       2006345519317.0/3224310063776.0,
                       2802321613138.0/2924317926251.0,
                       1.0])

p = 5;

def lserk_step(t0, y0, f, dt, N, rk, history=False):
    """
    Takes N discrete time-steps of length dt for the ODE

       y' = f(t,y),
       y(t0) = y0

    using the low-storage explicit Runge Kutta method defined by the
    (assumed diagonal) Butcher tableau coefficients in the dict rk. Returns
    the value of y at t = t0 + N*dt as a numpy vector. y0 is assumed to be a
    numpy vector.

    If history is True, also returns the entire trajectory history as a 2-D
    numpy array.
    """

    t = t0
    M = y0.size
    y = y0.copy()
    if history:
        yout = np.zeros([M, N+1])
        yout[:,0] = y0
    else:
        yout = y0.copy()

    for n in range(N): # For each timestep
        ky = np.zeros(M)
        for p in range(rk['a'].size): # For each stage
            stage_time = t + dt*rk['c'][p]
            ky = rk['a'][p]*ky + dt*f(stage_time,y)
            y += rk['b'][p]*ky

        t += dt
        if history:
            yout[:,n+1] = y

    if not history:
        yout = y

    return yout

def sawtooth_wave(x):
    """
    Evaluates a sawtooth wave of amplitude 1 on [-1,1].
    """

    y = np.zeros(x.shape)

    mask = x<-0.5
    y[mask] = 2*(x[mask]+1)

    mask = np.abs(x) <= 0.5
    y[mask] = -2*x[mask]

    mask = x>0.5
    y[mask] = -2*(x[mask]-1)

    return y
