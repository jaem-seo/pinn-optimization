import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
import myenv

np.random.seed(0)
tf.random.set_seed(0)
env = myenv.Env()
n_actions = env.action_space.shape[0]

# Actor
actor = keras.models.Sequential()
actor.add(keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(keras.layers.Dense(64, activation = 'tanh'))
actor.add(keras.layers.Dense(64, activation = 'tanh'))
actor.add(keras.layers.Dense(64, activation = 'tanh'))
actor.add(keras.layers.Dense(n_actions, activation = 'tanh'))
actor.summary()

# Critic
action_input = keras.layers.Input(shape=(n_actions,), name='action_input')
observation_input = keras.layers.Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = keras.layers.Flatten()(observation_input)
x = keras.layers.Concatenate()([action_input, flattened_observation])
x = keras.layers.Dense(64, activation = 'tanh')(x)
x = keras.layers.Dense(64, activation = 'tanh')(x)
x = keras.layers.Dense(64, activation = 'tanh')(x)
x = keras.layers.Dense(1, activation = 'linear')(x)
critic = keras.Model(inputs=[action_input, observation_input], outputs=x)
critic.summary()

# Compile, train, and save
memory = SequentialMemory(limit=100000, window_length=1)
agent = DDPGAgent(nb_actions=n_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=500,
                  gamma=.99, batch_size=64, target_model_update=1e-3)
agent.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])
agent.fit(env, nb_steps=100000, visualize=False, verbose=0)
agent.save_weights('saved', overwrite=True)

# Test
import testenv
env2 = testenv.Env()
agent.test(env2, nb_episodes=1, visualize=False)