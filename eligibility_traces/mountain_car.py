import gym
import random
import numpy as np
import itertools

from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def eps_greedy_action(Q, s, eps=0.2):
    if np.random.rand() > eps:
        # greedy
        return np.argmax(Q[s[0], s[1]])
    else:
        #random
        return random.choice([0, 1, 2])

def smooth(y, box_pts=100):
    """Smoothing filter"""
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

def obtain_value_function(Q):
    return np.max(Q, axis=2)

def discretize(x, offset=1.2, abs_range=1.8, n=20):
    x = x.copy()
    return (int(round((n-1) * (x[0]+offset)/abs_range)), int(round((n-1) * (x[1]+.07)/.14)))

def lambda_q_learning(Q, episodes, env, alpha, gamma, l=0.2, eps=0.0):
    episode_lengths = []
    trajectory = []
    value_functions = []
    for episode in tqdm(range(episodes), desc='lambda Q-Learning'):
        if episode % 20 == 0:
            value_functions.append(obtain_value_function(Q))
        E = np.zeros_like(Q)
        s = discretize(env.reset())
        a = eps_greedy_action(Q, s, eps=eps)
        done = False
        episode_length = 0
        s_ = [0, 0]
        while not done:
            episode_length += 1
            a = eps_greedy_action(Q, s, eps=eps)
            s_, reward, done, _ = env.step(a)
            if episode > episode - 5:
                env.render()
            if episode == episodes - 1:
                trajectory.append([19 * (s_[0]+1.2)/1.8, 19 * (s_[1]+.07)/.14])
            s_ = discretize(s_)
            a_ = eps_greedy_action(Q, s_, eps=eps) 
            a_star = np.argmax(Q[s_])
            delta = reward + gamma * Q[s_[0], s_[1], a_star] - Q[s[0], s[1], a]
            E[s[0], s[1], a] += 1
            for s0, s1, a in itertools.product(range(20), range(20), range(3)):
                Q[s0, s1, a] = Q[s0, s1, a] + alpha * delta * E[s0, s1, a]
                E[s0, s1, a] = gamma * l * E[s0, s1, a] if a_ == a_star else 0
            s = s_
            a = a_
        episode_lengths.append(episode_length)
    return Q, episode_lengths, np.array(trajectory), value_functions

def main():
    from gym.envs.classic_control.mountain_car import MountainCarEnv
    env = MountainCarEnv()
    env.actions=[0,1,2]
    AGGREGATED_STATES = 20
    Q = np.zeros((AGGREGATED_STATES, AGGREGATED_STATES, len(env.actions)))
    episodes = 100
    eps = .0
    n = 20
    Q_learned, rewards, trajectory, value_functions = lambda_q_learning(Q.copy(), episodes, env, alpha=0.1, gamma=0.99, eps=eps)
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i, img in enumerate(value_functions):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(trajectory[:,1], trajectory[:,0], 'bo')
    ax.imshow(obtain_value_function(Q_learned), cmap='ocean')
    plt.show()
    plt.figure()
    plt.plot(smooth(rewards))
    plt.show()


if __name__ == '__main__':
    main()

"""env = gym.make('MountainCar-v0')
env.reset()
env.render()
while True:
    a = env.action_space.sample()
    r, _, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        env.reset()"""