import numpy as np
import keras
import matplotlib.pyplot as plt

m, l, g = 1., 1., 9.8
torq_max = 1.5
tend = 10.
dt = 0.01
weights_path = 'saved_actor'

# Load model
actor = keras.models.Sequential()
actor.add(keras.layers.Flatten(input_shape=(1,) + (3,)))
actor.add(keras.layers.Dense(64, activation = 'tanh'))
actor.add(keras.layers.Dense(64, activation = 'tanh'))
actor.add(keras.layers.Dense(64, activation = 'tanh'))
actor.add(keras.layers.Dense(1, activation = 'tanh'))
actor.load_weights(weights_path)

# Inference
time = np.arange(0, tend + dt, dt)
theta = np.zeros_like(time)
omega = np.zeros_like(time)
torque = np.zeros_like(theta)

for it in range(1, len(time)):
    action = actor.predict(np.array([[[time[it-1], theta[it-1], omega[it-1]]]]))
    torque[it] = action * torq_max
    omega_t = (torque[it] - m * g * l * np.sin(theta[it-1])) / (m * l * l)
    omega[it] = omega[it-1] + dt * omega_t
    theta[it] = theta[it-1] + dt * omega[it] + 0.5 * dt ** 2 * omega_t

fig, ax = plt.subplots(1, 1, figsize=(5, 1.5))
ax.plot(time, theta, 'k', label='Angle')
ax.plot(time, torque, 'b', label='Torque')

ax.axhline(np.pi, c='r', ls='--', label='Goal')
ax.axhline(-np.pi, c='r', ls='--')
ax.axhline(0, c='k', ls='--', lw=0.5, zorder=-1)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('rl_evolution.svg')
plt.show()