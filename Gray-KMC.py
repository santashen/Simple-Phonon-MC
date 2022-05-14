import os
import sys

import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as c

# plt.style.use('science')

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
T = np.zeros((Ntt, Nx))
Qx = np.zeros((Ntt, Nx))

# gray-medium approximation
C = 690 * 2330
vg = 6400 
MFP = 43.7e-9
tau = MFP / vg

enrgInit = L*C*np.abs(Tinit - Teq)
enrgLeft = C * vg * tmax * np.abs(Tl - Teq) / 4
enrgRight = C * vg * tmax * np.abs(Tr - Teq) / 4

enrgTot = enrgInit + enrgLeft + enrgRight 

Eeff = enrgTot / N

# start simulation
for particle in range(N):
    Ri = np.random.rand()
    if Ri < enrgInit / enrgTot:
        x0 = L * np.random.rand()
        R = 2 * np.random.rand() - 1 
        vx = vg * R 
        psign = np.sign(Tinit - Teq)
    elif Ri > enrgInit / enrgTot and Ri < 1 - enrgRight / enrgTot:
        x0 = 0
        R = np.sqrt(np.random.rand()) 
        vx = vg * R
        psign = np.sign(Tl - Teq)
    else: # 此时从右墙发射
        x0 = L
        R = -np.sqrt(np.random.rand())
        vx = vg * R
        psign = np.sign(Tr - Teq)
    t0 = np.random.rand() * tmax 
    finished = False
    im = np.where(tt >= t0)[0][0] 
    while not finished:
        Delta_t = - tau * np.log(np.random.rand())
        t1 = t0 + Delta_t 
        x1 = x0 + Delta_t * vx 
        while (im < Ntt and t0 <= tt[im] and t1 > tt[im]):
            x_ = x0 + (tt[im] - t0) * vx
            indx = int(np.floor(x_ / L * Nx))
            if (indx < Nx) and (indx >= 0):
                T[im, indx] = T[im, indx] + psign * Eeff / C / Dx
                Qx[im, indx] = Qx[im, indx] + psign * Eeff * vx / Dx 
            im += 1

        R = 2 * np.random.rand() - 1 
        vx = vg * R 
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
