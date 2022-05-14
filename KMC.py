import os
import sys

import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as c

# plt.style.use('science')

# Dispersion of Si
data = pd.read_csv('dataSi.txt', names=['frequency', 'DoS', 'velocity', 'frequency_width', 'relaxation_time', 'polarization'], sep=' ')

# parameters of the simulated system
Teq = 300  # reference temperature
Tinit = 300 # init temperature
Tl = 301 # left-side high temperature boundary
Tr = 299 # right-side low temperature boundary
L = 2e-7 # length of the film
N = 10000 # phonon numbers
Nx = 20 # number of spatial cells
Ntt = 60 # number of time steps
dt = 5e-12 # time step
tmax = Ntt * dt
tt = np.arange(0, tmax + dt, dt) # time 
Dx = L / Nx # length of the spatial cell
xx = np.arange(Dx / 2, L + Dx / 2, Dx) # center coordinates of the spatial cells

de_dT = (c.hbar * data.frequency / Teq)**2 * np.exp(c.hbar * data.frequency / (c.k * Teq)) / c.k / (np.exp(c.hbar * data.frequency / (c.k * Teq)) - 1)**2 # hbar * omega * df/dT
T = np.zeros((Ntt, Nx)) # temperature
Qx = np.zeros((Ntt, Nx)) # heat flux

# build cumulative distribution function for sampling

@nb.jit(nopython=True)
def rand_choice_nb(arr, cumsum_prob):
    """Sample an element in arr with the probability given in prob.
    """
    return arr[np.searchsorted(cumsum_prob, np.random.random(), side="right")]

base = data.DoS * de_dT * data.frequency_width
coll = base / data.relaxation_time
vel = base * data.velocity
cumsum_base = np.cumsum(base) / np.sum(base)
cumsum_coll = np.cumsum(coll) / np.sum(coll)
cumsum_vel = np.cumsum(vel) / np.sum(vel)
C = np.sum(base)

enrgInit = L*C*np.abs(Tinit - Teq)
enrgLeft = np.sum(vel) * tmax * np.abs(Tl - Teq) / 4
enrgRight = np.sum(vel) * tmax * np.abs(Tr - Teq) / 4

enrgTot = enrgInit + enrgLeft + enrgRight 

Eeff = enrgTot / N

# start simulation
left = 0
right = 0
for particle in range(N):
    Ri = np.random.rand()
    if Ri < enrgInit / enrgTot:
        x0 = L * np.random.rand()
        phonon_index = rand_choice_nb(data.index.values, cumsum_base.values)
        R = 2 * np.random.rand() - 1
        vx = data.loc[phonon_index, 'velocity'] * Ri
        psign = np.sign(Tinit - Teq)
    elif Ri > enrgInit / enrgTot and Ri < 1 - enrgRight / enrgTot:
        x0 = 0
        phonon_index = rand_choice_nb(data.index.values, cumsum_vel.values) 
        R = np.sqrt(np.random.rand())
        vx = data.loc[phonon_index, 'velocity'] * R
        psign = np.sign(Tl - Teq)
        left += 1
    else:
        x0 = L
        phonon_index = rand_choice_nb(data.index.values, cumsum_vel.values) 
        R = -np.sqrt(np.random.rand())
        vx = data.loc[phonon_index, 'velocity'] * R
        psign = np.sign(Tr - Teq)
        right += 1
    t0 = np.random.rand() * tmax
    finished = False
    im = np.where(tt >= t0)[0][0]
    while not finished:
        Delta_t = - data.loc[phonon_index, 'relaxation_time'] * np.log(np.random.rand())
        t1 = t0 + Delta_t 
        x1 = x0 + Delta_t * vx 
        while (im < Ntt and t0 <= tt[im] and t1 > tt[im]):
            x_ = x0 + (tt[im] - t0) * vx
            indx = int(np.floor(x_ / L * Nx))
            if (indx < Nx) and (indx >= 0):
                T[im, indx] = T[im, indx] + psign * Eeff / C / Dx
                Qx[im, indx] = Qx[im, indx] + psign * Eeff * vx / Dx 
            im += 1

        phonon_index = rand_choice_nb(data.index.values, cumsum_coll.values)

        R = 2 * np.random.rand() - 1 
        vx = data.loc[phonon_index, 'velocity'] * R 
        x0 = x1
        t0 = t1
        if (t0 > tmax) or (x0 < 0) or (x0 > L):
            finished = True
            
plt.scatter(xx * 1e9, T[1, :] + Teq, s=15, facecolors='none', edgecolors='#E41A1C', marker='s', label='t = {} ps'.format(dt * 1e12))
plt.scatter(xx * 1e9, T[6, :] + Teq, s=15, facecolors='none', edgecolors='blue', marker='^', label='t = {} ps'.format(dt * 1e12 * 6))
plt.scatter(xx * 1e9, T[-1, :] + Teq, s=15, facecolors='none', edgecolors='#A65628', marker='o', label='t = {} ps'.format(dt * 1e12 * Ntt))
plt.legend()
plt.ylim(299, 301)
plt.ylabel('$T$ (K)')
plt.xlabel('$x$ (nm)')
plt.savefig('jojo.png')

for i in range(40):
    print('acquired effective k is {} W/mK'.format(np.average(Qx[-i, :]) / (2 / L)))
