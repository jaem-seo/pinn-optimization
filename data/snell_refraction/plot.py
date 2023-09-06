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
model.restore("saved_model-2232.ckpt")

t = np.linspace(tmin, tmax, 501)
uu = model.predict(np.array([t]).T)

yy = np.linspace(0, 1, 101)
xx = np.arctan(2. * np.tan(np.pi * yy)) / np.pi
xx[len(xx) // 2 + 1:] += 1

X, Y = np.meshgrid(xx, yy)
R = n1 + (n2 - n1) * 0.5 * (1. - np.cos(2. * np.pi * Y))

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
z = ax.contourf(X, Y, R, levels=30, alpha=0.5, zorder=-1)
plt.colorbar(z)
ax.plot(uu[:, 0], uu[:, 1], 'b', lw=4, zorder=1, label='PINN')
ax.plot(xx, yy, c='orange', ls='--', lw=1.5, zorder=2, label='Analytic')

#ax.scatter([0], [0], s=80, marker='o', color='k', zorder=3)
#ax.scatter([1], [1], s=80, marker='o', color='r', zorder=3)

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
ax.set_xticks([0.0, 0.5, 1.0])

plt.tight_layout
plt.savefig('snell.svg')
plt.show()

