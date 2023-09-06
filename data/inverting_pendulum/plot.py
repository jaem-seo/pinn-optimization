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
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad

def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black',zorder=None,sign=1):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          #theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=2.5,color=color_,zorder=zorder)
    ax.add_patch(arc)


    #========Create the arrow head
    if sign >= 0:
        endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))
    else:
        endX=centX+(radius/2)*np.cos(rad(angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/3,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_,
            zorder=zorder
        )
    )
    #ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort


n_output = 2 # theta, torq_norm
tmin, tmax = 0.0, 10.0
torq_max = 1.5
m = 1.
l = 1.
g = 9.8
target = -1.

t_plot = [2., 5.2, 6.4, 8., 9.5]

def ode(t, u):
    return u[:, 0:1] # Dummy ODE

geom = dde.geometry.TimeDomain(tmin, tmax)
data = dde.data.PDE(geom, ode, [])
net = dde.nn.FNN([1] + [64] * 3 + [n_output], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=0.01)
model.restore("saved_model-9804.ckpt")

t = np.linspace(tmin, tmax, 501)
uu = model.predict(np.array([t]).T)

fig, ax = plt.subplots(1, 1, figsize=(5, 1.5))
ax.plot(t, uu[:, 0], 'k', label='Angle')
ax.plot(t, torq_max * np.tanh(uu[:, 1]), 'b', label='Torque')

ax.axhline(np.pi, c='r', ls='--', label='Goal')
ax.axhline(-np.pi, c='r', ls='--')
ax.axhline(0, c='k', ls='--', lw=0.5, zorder=-1)
for tt in t_plot:
    ax.axvline(tt, c='darkgray', lw=0.5)
ax.set_xlim([tmin, tmax])
#ax.set_ylim([-np.pi - 0.2, np.pi + 0.2])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.legend()
plt.tight_layout()
plt.savefig('pendulum_pinn.svg')
#plt.show()

'''
# Check
dt = t[1] - t[0]

theta = np.zeros_like(t)
omega = np.zeros_like(t)
torque = np.tanh(uu[:, 1]) * torq_max
x = uu[:, 0]
x_t = np.append([0.], np.diff(x)) / dt
x_tt = np.append([0.], np.diff(x_t)) / dt
y = m * g * l * np.sin(x) + m * l * l * x_tt

for i in range(1, len(t)):
    omega_t = (torque[i] - m * g * l * np.sin(theta[i - 1])) / (m * l * l)
    omega[i] = omega[i - 1] + dt * omega_t
    theta[i] = theta[i - 1] + dt * omega[i] + 0.5 * dt ** 2 * omega_t

fig, ax = plt.subplots(1, 1, figsize=(5, 2))
ax.plot(t, uu[:, 0], 'k', label='Angle')
ax.plot(t, theta, 'k--', label='Angle')
ax.plot(t, torque, 'b', label='Torque')
ax.plot(t, y, 'b--', label='Torque')

ax.axhline(np.pi, c='r', ls='--', label='Goal')
ax.axhline(-np.pi, c='r', ls='--')
ax.axhline(0, c='k', ls='--', lw=0.5, zorder=-1)
ax.set_xlim([tmin, tmax])
#ax.set_ylim([-np.pi - 0.2, np.pi + 0.2])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.legend()
plt.tight_layout()
plt.show()
'''

theta = uu[:, 0]
torque = torq_max * np.tanh(uu[:, 1])
di = 10

x = np.linspace(-np.pi, np.pi, 101)

fig, axs = plt.subplots(1, len(t_plot), sharey=True, figsize=(5, 1))
for it, tt in enumerate(t_plot):
    idx = np.abs(t - tt).argmin()
    xx, yy = theta[idx], torque[idx]
    xxm, yym = theta[idx - di], torque[idx - di]
    xxmm, yymm = theta[idx - 2 * di], torque[idx - 2 * di]
    axs[it].plot(l * np.cos(x), l * np.sin(x), 'k--', lw=0.5)
    axs[it].plot([0, l * np.sin(xx)], [0, -l * np.cos(xx)], 'gray', lw=3)
    axs[it].scatter([l * np.sin(xx)], [-l * np.cos(xx)], marker='o', color='k', s=30, zorder=10)
    axs[it].plot([0, l * np.sin(xxm)], [0, -l * np.cos(xxm)], 'gray', lw=3, alpha=0.5)
    axs[it].scatter([l * np.sin(xxm)], [-l * np.cos(xxm)], marker='o', color='k', s=30, zorder=10, alpha=0.5)
    axs[it].plot([0, l * np.sin(xxmm)], [0, -l * np.cos(xxmm)], 'gray', lw=3, alpha=0.2)
    axs[it].scatter([l * np.sin(xxmm)], [-l * np.cos(xxmm)], marker='o', color='k', s=30, zorder=10, alpha=0.2)

    if it == len(t_plot) - 1:
        axs[it].plot([0, 0], [0, l], 'r--', lw=1.5, zorder=25)

    angle = 120
    theta2 = 300
    #drawCirc(axs[it], 0.3 * np.abs(yy), 0, 0, angle, theta2, color_='b', zorder=20, sign=np.sign(yy))
    drawCirc(axs[it], 0.4 * np.abs(yy), 0, 0, angle, theta2, color_='b', zorder=20, sign=np.sign(yy))
    
    axs[it].set_xlim([-1.2, 1.2])
    axs[it].set_ylim([-1.2, 1.2])
    axs[it].axis('off')
    axs[it].set_aspect('equal', 'box')

#plt.tight_layout()
plt.savefig('snap.svg')
#plt.show()


fig, axs = plt.subplots(2, 1, figsize=(1, 2))

x = np.linspace(-np.pi / 6, np.pi / 6)
axs[0].plot(l * np.sin(x), -l * np.cos(x), 'k--', lw=0.5)
axs[0].plot([0, l * np.sin(-np.pi / 8)], [0, -l * np.cos(-np.pi / 8)], 'gray', lw=3, alpha=0.3)
axs[0].scatter([l * np.sin(-np.pi / 8)], [-l * np.cos(-np.pi / 8)], marker='o', color='k', s=30, zorder=10, alpha=0.3)
axs[0].plot([0, l * np.sin(np.pi / 8)], [0, -l * np.cos(np.pi / 8)], 'gray', lw=3, alpha=1.)
axs[0].scatter([l * np.sin(np.pi / 8)], [-l * np.cos(np.pi / 8)], marker='o', color='k', s=30, zorder=10, alpha=1.)
drawCirc(axs[0], 0.4, 0, 0, angle, theta2, color_='b', zorder=20, sign=1)
axs[0].set_ylim([-1.4, 0.3])
axs[0].axis('off')
axs[0].set_aspect('equal', 'box')

x = np.linspace(-np.pi, np.pi, 101)
axs[1].plot(l * np.cos(x), l * np.sin(x), 'k--', lw=0.5)
axs[1].plot([0, 0], [0, l], 'gray', lw=3)
axs[1].scatter([0], [l], marker='o', color='k', s=30, zorder=10)
axs[1].set_ylim([-1.4, 1.4])
axs[1].axis('off')
axs[1].set_aspect('equal', 'box')

plt.savefig('intro.svg')
#plt.show()


#fig, axs = plt.subplots(1, 1, figsize=(3, 1))
fig, axs = plt.subplots(1, 1, figsize=(3, 1.5))

with open('loss.dat', 'r') as f:
    a = f.readlines()

step, phys_loss, const_loss, goal_loss = [], [], [], []
for line in a[1:]:
    step.append(float(line.split()[0]))
    phys_loss.append(float(line.split()[1]))
    const_loss.append(sum(list(map(float, line.split()[2:5]))))
    goal_loss.append(float(line.split()[5]))

axs.plot(step, phys_loss, 'b', label='Physics loss')
axs.plot(step, const_loss, 'r', label='Constraint loss')
axs.plot(step, goal_loss, 'g', label='Goal loss')
axs.plot(step, np.sum([phys_loss, const_loss, goal_loss], axis=0), 'k', lw=2, label='Sum')

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_yscale('log')
axs.set_yticks([1e-8, 1e-4, 1e0])
#axs.set_yticks([1e-4, 1e-1, 1e2])

plt.tight_layout()
#plt.legend(fontsize=8, ncols=2)

plt.savefig('loss.svg')
plt.show()
