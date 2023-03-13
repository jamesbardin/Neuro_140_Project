import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Pong-v0", render_mode = "rgb_array")


def preprocess_observation(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float32).ravel()


# Define Q-table
q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])

# Set hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.9999
episodes = 50

# Define the Q-learning algorithm
for episode in range(episodes):
    # Initialize the state
    state = preprocess_observation(env.reset()[0])

    done = False
    while not done:
        # Choose an action
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state.astype(int), :])


        # Take the action and observe  next state
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_observation(next_state)

        # Update the Q-value
        # q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]))
        state_int = state.astype(int)
        # bellman equation
        q_table[state_int, action] = (1 - alpha) * q_table[state_int, action] + alpha * (reward + gamma * np.max(q_table[next_state.astype(int), :]))


        # Update the state
        state = next_state

        # Decay the epsilon value
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay


# Train the agent
print("Entering Training")
for episode in range(episodes):
    state = preprocess_observation(env.reset()[0])
    done = False
    print("Entering episode " + str(episode) )
    while not done:
        action = np.argmax(q_table[state.astype(int), :])
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_observation(next_state)
        state = next_state
        if episode % 10 == 0:
            screen = env.render()
            plt.imshow(screen)
env.close()
