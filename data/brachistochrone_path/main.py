#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
dde.backend.tf.random.set_random_seed(seed)

# Set hyperparameters
n_output = 2 # x, y

num_domain = 1000

n_adam = 2000

lr = 1e-3 # for Adam
loss_weights = [1., 0.01, 1., 1., 1., 1.]

# Set physical parameters
tmin, tmax = 0.0, 1.0 # normalized time
T = dde.Variable(1.0) # end time

x0, y0 = 0., 1.
x1, y1 = 1., 0.
m0 = 1.
g = 9.8

def ode(t, u):
    x, y = u[:, 0:1], u[:, 1:2]
    x_T = dde.grad.jacobian(x, t) / T
    y_T = dde.grad.jacobian(y, t) / T
    ode1 = m0 * g * y0 - (m0 * g * y + 0.5 * m0 * (x_T ** 2 + y_T ** 2))
    ode2 = T * tf.ones_like(x)
    return [ode1, ode2]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_tf(x, y):
    return 1. / (1. + tf.exp(-y))

def boundary_left(t, on_boundary):
    return on_boundary * np.isclose(t[0], tmin)

def boundary_right(t, on_boundary):
    return on_boundary * np.isclose(t[0], tmax)

geom = dde.geometry.TimeDomain(tmin, tmax)
bc0x = dde.icbc.DirichletBC(geom, lambda t: np.array([x0]), boundary_left, component=0)
bc0y = dde.icbc.DirichletBC(geom, lambda t: np.array([y0]), boundary_left, component=1)
bc1x = dde.icbc.DirichletBC(geom, lambda t: np.array([x1]), boundary_right, component=0)
bc1y = dde.icbc.DirichletBC(geom, lambda t: np.array([y1]), boundary_right, component=1)
data = dde.data.PDE(geom, ode, [bc0x, bc0y, bc1x, bc1y], num_domain=num_domain, num_boundary=2)

net = dde.nn.FNN([1] + [64] * 3 + [n_output], "tanh", "Glorot normal")
net.apply_output_transform(sigmoid_tf)

resampler = dde.callbacks.PDEPointResampler(period=100)
variable = dde.callbacks.VariableValue(T, period=10)
ckpts = dde.callbacks.ModelCheckpoint('checkpoint', period=500)

model = dde.Model(data, net)
model.compile("adam", lr=lr, loss_weights=loss_weights, external_trainable_variables=T)
losshistory, train_state = model.train(display_every=10, iterations=n_adam, callbacks=[resampler, variable, ckpts])
model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=T)
losshistory, train_state = model.train(display_every=10, callbacks=[resampler, variable, ckpts])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save('saved_model')

t = np.linspace(tmin, tmax, 101)
uu = model.predict(np.array([t]).T)

plt.plot(uu[:, 0], uu[:, 1], 'b')
r = 0.5729
theta = np.linspace(0, np.arccos(1 - 1 / r), 101)
x = r * (theta - np.sin(theta))
y = 1 - r * (1 - np.cos(theta))
plt.plot(x, y, 'k--')

plt.show()
