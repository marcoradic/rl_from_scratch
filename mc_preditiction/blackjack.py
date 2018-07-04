import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import LightSource
import numpy as np
from collections import defaultdict
import random
import itertools


def monte_carlo_prediciton():
    env = gym.make('Blackjack-v0')
    episodes = 500000

    value = np.zeros((22, 11)) -1.
    value_ace = np.zeros_like(value) -1.
    returns = defaultdict(list)
    returns_ace = defaultdict(list)

    sum_returns = defaultdict(int)
    sum_returns_ace = defaultdict(int)
    for episode in tqdm(range(episodes)):
        obs = env.reset()
        done = False
        # no useable ace
        history = []
        rewards = 0
        # useable ace
        history_ace = []
        while not done:
            player_sum, dealer_card, usable_ace = obs
            obs, reward, done, _ = env.step(0 if player_sum >= 20 else 1)
            if player_sum >= 12:
                if usable_ace:
                    history_ace.append((player_sum, dealer_card))
                else:
                    history.append((player_sum, dealer_card))
                if done:
                    rewards = reward
        g = rewards #-1 if -1 in rewards else max(rewards)
        for element in history:
            returns[element].append(g)
            sum_returns[element] += g
            value[element[0], element[1]] = float(sum_returns[element] / len(returns[element]))
        for element in history_ace:
            returns_ace[element].append(g)
            sum_returns_ace[element] += g
            value_ace[element[0], element[1]] = float(sum_returns_ace[element] / len(returns_ace[element]))
    print(value)
    print(len(value))
    print(value_ace)
    print(len(value_ace))

    x,y = np.meshgrid(np.arange(11),np.arange(22))

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = plt.subplot(121, projection='3d')
    ax.plot_wireframe(x,y, value_ace, cmap='coolwarm')
    ax.set_zlim(-1.01, 1.01)
    ax.set_title('$V$ usable ace')
    ax.set_ylim(12, 21)
    ax.set_xlim(1,10)

    ax = plt.subplot(122, projection='3d')
    ax.plot_wireframe(x,y, value, cmap='coolwarm')
    ax.set_zlim(-1.01, 1.01)
    ax.set_title('$V$ no usable ace')
    ax.set_ylim(12, 21)
    ax.set_xlim(1,10)

    plt.show()

class ExploringStarts(object):

    def __init__(self, episodes):
        self.Q = defaultdict(lambda : np.random.uniform(-1,1))
        self.policy = defaultdict(lambda : np.random.randint(0,2))
        for s in itertools.product(range(12,22), range(1,11), [0,1]):
            self.policy[s] = 0 if s[0] >= 20 else 1
        # [player_sum, dealer_card, usable_ace]
        # self.state_space = [x for x in itertools.product(range(12, 22), range(1, 11), [0,1])]
        # [hit, stick]
        self.action_space = [0, 1]
        self.env = gym.make('Blackjack-v0')
        self.episodes = range(episodes)
        self.returns = defaultdict(list)
        self.returns_sum = defaultdict(int)
        self.returns_ace = defaultdict(list)
        self.returns_ace_sum = defaultdict(int)
    
    def run(self,):
        for episode in tqdm(self.episodes):
            start_action = self._random_start()
            obs = self.env.reset()
            history = []
            history_ace = []
            g = 0
            start = True
            done = False
            while not done:
                player_sum, dealer_card, usable_ace = obs
                if player_sum < 12:
                    action = 1
                elif start:
                    action = start_action
                    start = False
                else:
                    action = self.policy[obs]
                obs, reward, done, _ = self.env.step(action)
                if player_sum >= 12:
                    if usable_ace:
                        history_ace.append((obs, action))
                    else:
                        history.append((obs, action))
                    if done:
                        g = reward
            for element in history:
                self.returns[element].append(g)
                self.returns_sum[element] += g
                self.Q[element] = float(self.returns_sum[element] / len(self.returns[element]))
                self.policy[element[0]] = int(self.Q[(element[0], 0)] < self.Q[(element[0],1)])
            for element in history_ace:
                self.returns_ace[element].append(g)
                self.returns_ace_sum[element] += g
                self.Q[element] = float(self.returns_ace_sum[element] / len(self.returns_ace[element]))
                self.policy[element[0]] = int(self.Q[(element[0], 0)] < self.Q[(element[0], 1)])
        return self.Q, self.policy
    
    def _random_start(self,):
        return random.choice(self.action_space)
    
    def plot_policy(self):
        policy = np.zeros((22,11))
        for key in self.policy.keys():
            if not key[2] and key[0] < 22:
                policy[key[0], key[1]] = self.policy[key]
        print('Policy no usable ace')
        plt.imshow(policy)
        plt.show()
        for key in self.policy.keys():
            if key[2] and key[0] < 22:
                policy[key[0], key[1]] = self.policy[key]
        print('Policy usable ace')
        plt.imshow(policy)
        plt.show()

    def plot_value(self):
        value = np.zeros((22,11)) -1.0
        value_ace = np.zeros_like(value) -1.0
        for key in itertools.product(range(12,22), range(1, 11)):
            value[key[0], key[1]] = max(self.Q[((key[0], key[1], False), 0)], self.Q[((key[0], key[1], False), 1)])
            value_ace[key[0], key[1]] = max(self.Q[((key[0], key[1], True), 0)], self.Q[((key[0], key[1], True), 1)])
        #plt.imshow(value)
        #plt.show()
        x,y = np.meshgrid(np.arange(11),np.arange(22))
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = plt.subplot(121, projection='3d')
        ax.plot_surface(x,y, value_ace, cmap='coolwarm')
        ax.set_zlim(-1.01, 1.01)
        ax.set_title('$V$ usable ace')
        ax.set_ylim(12, 21)
        ax.set_xlim(1,10)

        ax = plt.subplot(122, projection='3d')
        ax.plot_surface(x,y, value, cmap='coolwarm')
        ax.set_zlim(-1.01, 1.01)
        ax.set_title('$V$ no usable ace')
        ax.set_ylim(12, 21)
        ax.set_xlim(1,10)

        plt.show()

def main():
    #monte_carlo_prediciton()
    es = ExploringStarts(500000)
    Q, policy = es.run()
    es.plot_policy()
    es.plot_value()

if __name__ == '__main__':
    main()