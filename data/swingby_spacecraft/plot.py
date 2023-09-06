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
bh_xygm = [
    [-0.5, -1.0, 0.5],
    [-0.2, 0.4, 1.0],
    [0.8, 0.3, 0.5],
]
x0, y0 = -1., -1.
x1, y1 = 1., 1.
m0 = 1.
T = 2.22

def ode(t, u):
    return u[:, 0:1] # Dummy ODE

def tanh_tf(x, y):
    return tf.tanh(y)

geom = dde.geometry.TimeDomain(tmin, tmax)
data = dde.data.PDE(geom, ode, [])
net = dde.nn.FNN([1] + [64] * 3 + [n_output], "tanh", "Glorot normal")
net.apply_output_transform(tanh_tf)

model = dde.Model(data, net)
model.compile("adam", lr=0.01)
model.restore("saved_model-3215.ckpt")

t = np.linspace(tmin, tmax, 501)
dt = t[1] - t[0]
uu = model.predict(np.array([t]).T)
x, y = uu[:, 0], uu[:, 1]
x_t = np.gradient(x) / dt / T
y_t = np.gradient(y) / dt / T
x_tt = np.gradient(x_t) / dt / T
y_tt = np.gradient(y_t) / dt / T

fgx, fgy = [], []
for xtmp, ytmp, gmtmp in bh_xygm:
    fgx.append(-gmtmp * m0 * (x - xtmp) / ((x - xtmp) ** 2 + (y - ytmp) ** 2) ** 1.5)
    fgy.append(-gmtmp * m0 * (y - ytmp) / ((x - xtmp) ** 2 + (y - ytmp) ** 2) ** 1.5)

fgx.append(-m0 * x_tt)
fgy.append(-m0 * y_tt)

print(np.array(fgx)[:, ::50])

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(uu[:, 0], uu[:, 1], 'b', lw=3, zorder=1, label='Trajectory')

ax.quiver(x[::50][1:-1], y[::50][1:-1], np.sum(fgx[:-1], axis=0)[::50][1:-1], np.sum(fgy[:-1], axis=0)[::50][1:-1], color='gray', scale=20., width=0.01, label='Gravity/20')
#ax.quiver(x[::50][1:-1], y[::50][1:-1], np.sum(fgx, axis=0)[::50][1:-1], np.sum(fgy, axis=0)[::50][1:-1], color='g', scale=1., width=0.01)
ax.quiver(x[::50][1:-1], y[::50][1:-1], np.sum(fgx, axis=0)[::50][1:-1], np.sum(fgy, axis=0)[::50][1:-1], color='orange', scale=1., width=0.01, label='Thurst')

for xtmp, ytmp, gmtmp in bh_xygm:
    ax.scatter(xtmp, ytmp, s=gmtmp * 200, c='k', marker='o')

ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc='lower right')
ax.set_aspect('equal', 'box')

#plt.tight_layout()
plt.savefig('path.svg')
#plt.show()


fig, ax = plt.subplots(1, 1, figsize=(6, 3))
cs = ['r', 'k', 'g']
for i, (xtmp, ytmp, gmtmp) in enumerate(bh_xygm):
    ax.plot(t, np.sqrt(fgx[i] ** 2 + fgy[i] ** 2), lw=1, c=cs[i], label=f'Gravity by Object {i+1}')

ax.plot(t, np.sqrt(np.sum(fgx[:-1], axis=0) ** 2 + np.sum(fgy[:-1], axis=0) ** 2), lw=4, c='gray', label='Total gravity')
ax.plot(t[2:-2], np.sqrt(fgx[-1] ** 2 + fgy[-1] ** 2)[2:-2], lw=1.5, c='b', ls='--', label='Required force for the trajectory')
ax.plot(t[2:-2], np.sqrt(np.sum(fgx, axis=0) ** 2 + np.sum(fgy, axis=0) ** 2)[2:-2], lw=2, c='orange', label='Thrust')

ax.axhline(0, c='k', ls='--', lw=0.5)
ax.set_ylim([None, 5.5])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.legend(ncols=2, fontsize=9.5)
#plt.tight_layout()
plt.savefig('force.svg')
plt.show()
