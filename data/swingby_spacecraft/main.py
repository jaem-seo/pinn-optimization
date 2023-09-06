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
loss_weights = [1., 1., 1., 1., 1., 1.]

# Set physical parameters
tmin, tmax = 0.0, 1.0 # normalized time
T = dde.Variable(1.0) # end time
bh_xygm = [
    [-0.5, -1.0, 0.5],
    [-0.2, 0.4, 1.0],
    [0.8, 0.3, 0.5],
]


x0, y0 = -1., -1.
x1, y1 = 1., 1.
m0 = 1.

def ode(t, u):
    x, y = u[:, 0:1], u[:, 1:2]
    x_T = dde.grad.jacobian(x, t) / T
    x_TT = dde.grad.jacobian(x_T, t) / T
    y_T = dde.grad.jacobian(y, t) / T
    y_TT = dde.grad.jacobian(y_T, t) / T
    
    ode_x = m0 * x_TT
    ode_y = m0 * y_TT
    for xtmp, ytmp, gmtmp in bh_xygm:
        ode_x += gmtmp * m0 * (x - xtmp) / ((x - xtmp) ** 2 + (y - ytmp) ** 2) ** 1.5
        ode_y += gmtmp * m0 * (y - ytmp) / ((x - xtmp) ** 2 + (y - ytmp) ** 2) ** 1.5
        
    return [ode_x, ode_y]

def tanh_tf(x, y):
    return tf.tanh(y)

def initial(_, on_initial):
    return on_initial

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
net.apply_output_transform(tanh_tf)

resampler = dde.callbacks.PDEPointResampler(period=100)
variable = dde.callbacks.VariableValue(T, period=10)

model = dde.Model(data, net)
model.compile("adam", lr=lr, loss_weights=loss_weights, external_trainable_variables=T)
losshistory, train_state = model.train(display_every=10, iterations=n_adam, callbacks=[resampler, variable])
model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=T)
losshistory, train_state = model.train(display_every=10, callbacks=[resampler, variable])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save('saved_model')

t = np.linspace(tmin, tmax, 101)
uu = model.predict(np.array([t]).T)

plt.plot(uu[:, 0], uu[:, 1], 'k')
for xtmp, ytmp, gmtmp in bh_xygm:
    plt.scatter(xtmp, ytmp, s=gmtmp * 500, c='r', marker='o')
plt.show()
