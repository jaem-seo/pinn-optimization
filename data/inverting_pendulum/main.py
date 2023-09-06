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
n_output = 2 # theta, torq_norm

num_domain = 1000

n_adam = 5000

lr = 2e-2 # for Adam
loss_weights = [1., 10., 1., 1., 1.]

# Set physical parameters
tmin, tmax = 0.0, 10.0
m = 1.
l = 1.
g = 9.8
torq_max = 1.5
target = -1.

class Custom_BC(dde.icbc.BC):
    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = dde.icbc.boundary_conditions.npfunc_range_autocache(dde.utils.return_tensor(func))
        
    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        theta = outputs[:, 0:1]
        goal = tf.cos(theta)
        return goal[beg:end, self.component:self.component + 1] - values

def ode(t, u):
    theta, torq_norm = u[:, 0:1], tf.tanh(u[:, 1:2])
    torq = torq_max * torq_norm
    theta_t = dde.grad.jacobian(theta, t)
    theta_tt = dde.grad.jacobian(theta_t, t)
    
    ode = m * l * l * theta_tt - (torq - m * g * l * tf.sin(theta))
    return ode

def initial(_, on_initial):
    return on_initial

def boundary_left(t, on_boundary):
    return on_boundary * np.isclose(t[0], tmin)

def boundary_right(t, on_boundary):
    return on_boundary * np.isclose(t[0], tmax)

geom = dde.geometry.TimeDomain(tmin, tmax)
ic1 = dde.icbc.IC(geom, lambda t: np.array([0.]), initial, component=0)
ic2 = dde.icbc.IC(geom, lambda t: np.array([0.]), initial, component=1)
ic3 = dde.icbc.NeumannBC(geom, lambda t: np.array([0.]), boundary_left, component=0)
opt = Custom_BC(geom, lambda t: np.array([target]), boundary_right) # custom ICBC
data = dde.data.PDE(geom, ode, [ic1, ic2, ic3, opt], num_domain=num_domain, num_boundary=2)

net = dde.nn.FNN([1] + [64] * 3 + [n_output], "tanh", "Glorot normal")

resampler = dde.callbacks.PDEPointResampler(period=100)
#dde.optimizers.config.set_LBFGS_options(ftol=np.nan, gtol=np.nan, maxiter=8000, maxfun=8000)

model = dde.Model(data, net)
model.compile("adam", lr=lr, loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=10, iterations=n_adam, callbacks=[resampler])
model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=10)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save('saved_model')

t = np.linspace(tmin, tmax, 101)
uu = model.predict(np.array([t]).T)
plt.plot(t, uu[:, 0])
plt.plot(t, np.tanh(uu[:, 1]))
plt.show()
