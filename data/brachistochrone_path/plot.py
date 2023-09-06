import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
import numpy as np
import matplotlib
font = 12
#matplotlib.rcParams['axes.linewidth']=1.5
matplotlib.rcParams['axes.labelsize']=font
matplotlib.rcParams['axes.titlesize']=font
matplotlib.rcParams['xtick.labelsize']=font
matplotlib.rcParams['ytick.labelsize']=font
import matplotlib.pyplot as plt
import tensorflow as tf

n_output = 2 # x, y
tmin, tmax = 0.0, 1.0
n1, n2 = 1., 2.
ckpts = [500, 1000, 1500, 2000]

def ode(t, u):
    return u[:, 0:1] # Dummy ODE

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_tf(x, y):
    return 1. / (1. + tf.exp(-y))

geom = dde.geometry.TimeDomain(tmin, tmax)
data = dde.data.PDE(geom, ode, [])
net = dde.nn.FNN([1] + [64] * 3 + [n_output], "tanh", "Glorot normal")
net.apply_output_transform(sigmoid_tf)

model = dde.Model(data, net)
model.compile("adam", lr=0.01)
model.restore("saved_model-3692.ckpt")

t = np.linspace(tmin, tmax, 501)
uu = model.predict(np.array([t]).T)

r = 0.5729
theta = np.linspace(0, np.arccos(1 - 1 / r), 101)
x = r * (theta - np.sin(theta))
y = 1 - r * (1 - np.cos(theta))

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(uu[:, 0], uu[:, 1], 'b', lw=4, zorder=1, label='PINN')
ax.plot(x, y, c='orange', ls='--', lw=1.5, zorder=2, label='Analytic')

cs = ['darkgray', 'k', 'r', 'g']
for ic, ckpt in enumerate(ckpts):
    alpha = 0.1 + 0.9 * (ckpt - min(ckpts)) / (max(ckpts) - min(ckpts))
    model.restore(f"checkpoint-{ckpt}.ckpt")
    uu = model.predict(np.array([t]).T)
    ax.plot(uu[:, 0], uu[:, 1], c=cs[ic], lw=1, zorder=0)#, label=f'ckpt_{ckpt}')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
ax.set_aspect('equal', 'box')

plt.tight_layout
plt.savefig('curve.svg')
plt.show()

