from dataclasses import dataclass

import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as c
from scipy.interpolate import interp1d

# plt.style.use('science')

# Dispersion of Si
data = pd.read_csv('dataSi.txt', names=['frequency', 'DoS', 'velocity', 'frequency_width', 'relaxation_time', 'polarization'], sep=' ')

# parameters of the simulated system
Teq = 300 # reference temperature
Tl = 301 # left-side high temperature boundary
Tr = 299 # right-side low temperature boundary
L = 2e-7 # length of the film
Nx = 30 # number of spatial cells
dt = 5e-12 # time step
Ntt = 60 # number of time steps
Dx = L / Nx # length of the spatial cell
xx = np.arange(Dx / 2, L + Dx / 2, Dx) # center coordinates of the spatial cells
x_lines = np.arange(Dx, L+Dx, Dx) # right borders of the spatial cells, for convenience to identify phonon distribution
Temperature = np.zeros((Ntt, Nx)) # temperature
PesudoTemperature = np.zeros((Ntt, Nx)) # pesudo temperature
drift_projection = np.zeros(len(data)) # to identify the heat flux

@dataclass
class Phonon:
    x: float
    next_x: float
    mu: float
    index: int
    sign: int
    
# build the look-up tables for energy-temperature conversion
# Bose-enstein distribution
def f(data, T):
    return 1 / (np.exp(c.hbar * data.frequency / (c.k * T)) - 1)

# \int_0^{\omega_m} \hbar\omega D(\omega) f(T) - f(T_eq) \mathrm{d}\omega
def GetEnergyDeviation(data, T):
    return np.sum(c.hbar * data.frequency * data.DoS * (f(data, T) - f(data, Teq)) * data.frequency_width)

# \int_0^{\omega_m} \hbar\omega D(\omega)/\tau f(T) - f(T_eq) \mathrm{d}\omega
def GetPesudoEnergyDeviation(data, T):
    return np.sum(c.hbar * data.frequency * data.DoS * (f(data, T) - f(data, Teq)) / data.relaxation_time * data.frequency_width)

# map local energy density to temperature
T_range = np.linspace(298.5, 301.5, 50)
EnergyToT = interp1d([GetEnergyDeviation(data, T) for T in T_range], T_range)
PesudoEnergyToT = interp1d([GetPesudoEnergyDeviation(data, T) for T in T_range], T_range)

# build cumulative distribution function for sampling
@nb.jit(nopython=True)
def rand_choice_nb(arr, cumsum_prob):
    """Sample an element in arr with the probability given in prob.
    """
    return arr[np.searchsorted(cumsum_prob, np.random.random(), side="right")]

de_dT = (c.hbar * data.frequency / Teq)**2 * np.exp(c.hbar * data.frequency / (c.k * Teq)) / c.k / (np.exp(c.hbar * data.frequency / (c.k * Teq)) - 1)**2 # hbar * omega * df/dT，注意这里多乘了一个hbar * omega，因为模拟的是声子能量团

base = data.DoS * de_dT * data.frequency_width
coll = base / data.relaxation_time
vel_hot = data.DoS * data.velocity * c.hbar * data.frequency * data.frequency_width * (f(data, Tl) - f(data, Teq))
vel_cold = data.DoS * data.velocity * c.hbar * data.frequency * data.frequency_width * (f(data, Teq) - f(data, Tr))

# scatter
cumsum_coll = np.cumsum(coll) / np.sum(coll)
# left-side hot boundary emission
cumsum_hot = np.cumsum(vel_hot) / np.sum(vel_hot)
# right-side cold boundary emission
cumsum_cold = np.cumsum(vel_cold) / np.sum(vel_cold)

# determine the representative energy of one phonon bundle, it's somewhat causal
Eeff = GetEnergyDeviation(data, Tl) * L / 20000
q_hot = np.sum(data.DoS * data.velocity * c.hbar * data.frequency * (f(data, Tl) - f(data, Teq)) * data.frequency_width) / 4 * dt
q_cold = - np.sum(data.DoS * data.velocity * c.hbar * data.frequency * (f(data, Tr) - f(data, Teq)) * data.frequency_width) / 4* dt
q_hot_emitted_per_dt = int(q_hot / Eeff)
q_cold_emitted_per_dt = int(q_cold / Eeff)

# start simulation
phonons = []

for t in range(Ntt):
    ############# phonon drift #############
    
    remove_indexes = []
    drift_projection = np.zeros(len(data))
    for i, phonon in enumerate(phonons):
        phonon.next_x = phonon.x + data.loc[phonon.index, "velocity"] * phonon.mu * dt
        if phonon.next_x < 0:
            phonon.next_x = 0
            remove_indexes.append(i)
        elif phonon.next_x > L:
            phonon.next_x = L
            remove_indexes.append(i)
        drift_projection[phonon.index] += phonon.sign * (phonon.next_x - phonon.x)
    for index in remove_indexes[::-1]:
        phonons.pop(index)

    ############# boundary emission #############

    # left-side hot boundary
    for i in range(q_hot_emitted_per_dt):
        mu = np.sqrt(np.random.rand())
        phonon_index = rand_choice_nb(data.index.values, cumsum_hot.values)
        phonon = Phonon(0, 0, mu, phonon_index, 1)
        time = dt * np.random.rand()
        phonon.next_x = phonon.x + data.loc[phonon.index, "velocity"] * phonon.mu * time
        if phonon.next_x >= L:
            phonon.next_x = L
            drift_projection[phonon.index] += phonon.sign * (phonon.next_x - phonon.x)
        else:
            drift_projection[phonon.index] += phonon.sign * (phonon.next_x - phonon.x)
            phonons.append(phonon)

    # right-side cold boundary
    for i in range(q_cold_emitted_per_dt):
        mu = -np.sqrt(np.random.rand())
        phonon_index = rand_choice_nb(data.index.values, cumsum_cold.values)
        phonon = Phonon(L, 0, mu, phonon_index, -1)
        time = dt * np.random.rand()
        phonon.next_x = phonon.x + data.loc[phonon.index, "velocity"] * phonon.mu * time
        if phonon.next_x <= 0:
            phonon.next_x = 0
            drift_projection[phonon.index] += phonon.sign * (phonon.next_x - phonon.x)
        else:
            drift_projection[phonon.index] += phonon.sign * (phonon.next_x - phonon.x)
            phonons.append(phonon)

    # update phonon position
    for phonon in phonons:
        phonon.x = phonon.next_x

    heat_flux = np.sum(drift_projection) / dt * Eeff / L

    ############# calculate temperature #############
    scatter_index_arrays = [[] for _ in range(len(xx))]
    Energy = np.zeros(Nx)
    PesudoEnergy = np.zeros(Nx)
    for index, phonon in enumerate(phonons):
        position_index = np.searchsorted(x_lines, phonon.x)
        Energy[position_index] += phonon.sign * Eeff / Dx
        PesudoEnergy[position_index] += (
            phonon.sign * Eeff / data.loc[phonon.index, "relaxation_time"] / Dx
        )
        Temperature[t] = EnergyToT(Energy)
        PesudoTemperature[t] = PesudoEnergyToT(PesudoEnergy)

        if np.random.rand() < 1 - np.exp(
            -dt / data.loc[phonon.index, "relaxation_time"]
        ):
            scatter_index_arrays[position_index].append(index)

    ############# scatter #############
    remove_indexes = []
    for position_index, scatter_index_array in enumerate(scatter_index_arrays):
        net_scatter_phonon_number = abs(
            int(np.sum([phonons[i].sign for i in scatter_index_arrays[position_index]]))
        )
        phonon_sign = np.sign(Temperature[t, position_index] - Teq)
        scattered_phonons = np.random.choice(
            scatter_index_array, net_scatter_phonon_number, replace=False
        )
        for phonon_index in scattered_phonons:
            phonons[phonon_index] = Phonon(
                phonons[phonon_index].x,
                phonons[phonon_index].next_x,
                2 * np.random.rand() - 1,
                rand_choice_nb(data.index.values, cumsum_coll.values),
                phonon_sign,
            )
        remove_indexes.append(set(scatter_index_array) - set(scattered_phonons))
    remove_set = set()
    for subset in remove_indexes:
        remove_set = remove_set | subset
    for index in list(sorted(remove_set))[::-1]:
        phonons.pop(index)
    print('current effective k is {} W/mK'.format(heat_flux / (2/L)))

plt.scatter(xx * 1e9, Temperature[0, :], s=15, facecolors='none', edgecolors='#E41A1C', marker='s', label='t = {} ps'.format(dt * 1e12))
plt.scatter(xx * 1e9, Temperature[5, :], s=15, facecolors='none', edgecolors='blue', marker='^', label='t = {} ps'.format(dt * 1e12 * 6))
plt.scatter(xx * 1e9, Temperature[-1, :], s=15, facecolors='none', edgecolors='#A65628', marker='o', label='t = {} ps'.format(dt * 1e12 * Ntt))
plt.legend()
plt.ylim(299, 301)
plt.ylabel('$T$ (K)')
plt.xlabel('$x$ (nm)')
plt.savefig('jojo.png')