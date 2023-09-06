import numpy as np
import matplotlib
font = 12
#matplotlib.rcParams['axes.linewidth']=1.5
matplotlib.rcParams['axes.labelsize']=font
matplotlib.rcParams['axes.titlesize']=font
matplotlib.rcParams['xtick.labelsize']=font
matplotlib.rcParams['ytick.labelsize']=font
import matplotlib.pyplot as plt

tmin, tmax = 0.0, 10.0
m = 1.
l = 1.
g = 9.8
torq_max = 1.5
target = -1.

dt = 0.01

t = np.arange(tmin, tmax, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)
torque = np.zeros_like(t)
torque[1:] = torq_max

for i in range(1, len(t)):
    omega_t = (torque[i] - m * g * l * np.sin(theta[i - 1])) / (m * l * l)
    omega[i] = omega[i - 1] + dt * omega_t
    theta[i] = theta[i - 1] + dt * omega[i] + 0.5 * dt ** 2 * omega_t

fig, ax = plt.subplots(1, 1, figsize=(5, 1.5))
ax.plot(t, theta, 'k', label='Angle')
ax.plot(t, torque, 'b', label='Torque')

ax.axhline(np.pi, c='r', ls='--', label='Goal')
ax.axhline(-np.pi, c='r', ls='--')
ax.axhline(0, c='k', ls='--', lw=0.5, zorder=-1)
ax.set_xlim([tmin, tmax])
#ax.set_ylim([-np.pi - 0.2, np.pi + 0.2])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.legend()
plt.tight_layout()
plt.savefig('pendulum_greedy.svg')
plt.show()

