from collections import defaultdict
import random
import itertools

from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

SIZE = (8,8)

def eps_greedy_action(Q, s, eps=0.2):
    if np.random.rand() > eps:
        # greedy
        return np.argmax(Q[s])
    else:
        #random
        return np.random.randint(0, Q.shape[1])

class modlist(object):
    """modulo list with capacity n"""
    def __init__(self, n):
        self.n = n
        self._list = []
        self.counter = 0
    
    def put(self, item):
        if self.counter < self.n + 1 :
            self._list.append(item)
        else:
            self._list[self.counter%(self.n+1)] = item
        self.counter += 1

    def __getitem__(self, idx):
        return self._list.__getitem__(idx%(self.n+1))

    def __repr__(self):
        return repr(self._list)
    
    def __str__(self):
        return str(self._list)

def n_step_sarsa(Q, n, episodes, env, alpha, gamma, eps=0.2):
    episode_lengths= []
    for episode in tqdm(range(episodes), desc=f'n={n}, a={alpha:.1f}'):
        state_storage = modlist(n)
        reward_storage = modlist(n)
        action_storage = modlist(n)
        s = env.reset()
        state_storage.put(s)
        a = eps_greedy_action(Q, s, eps=eps)
        action_storage.put(a)
        reward_storage.put(0)
        T = np.inf
        tau = -1
        t = 0
        episode_length = 0
        while tau != T - 1:
            episode_length += 1
            if t < T:
                s_, reward, done, _ = env.step(action_storage[t])
                state_storage.put(s_)
                reward_storage.put(reward)
                if done:
                    T = t + 1
                else:
                    a_ = eps_greedy_action(Q, s_, eps=eps)
                    action_storage.put(a_)
            tau = t - n + 1 # tau is the time whose estimate is being updated
            if tau >= 0:
                G = sum([gamma**(i-tau-1) * reward_storage[i] for i in range(tau+1,min(tau+n,T)+1)])
                if tau + n < T:
                    G += gamma**n * Q[state_storage[tau+n], action_storage[tau+n]]
                Q[state_storage[tau],action_storage[tau]] += alpha * (G - Q[state_storage[tau], action_storage[tau]])
            t += 1
        episode_lengths.append(episode_length)
    return Q, episode_lengths

def obtain_value_function(Q):
    v = np.zeros((Q.shape[0],))
    for i in range(Q.shape[0]):
        v[i] = np.max(Q[i])
    return v.reshape(SIZE)

def obtain_policy(Q):
    policy = np.zeros((Q.shape[0],))
    for i in range(Q.shape[0]):
        policy[i] = np.argmax(Q[i])
    return policy.reshape(SIZE)

def smooth(y, box_pts=100):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

def pretty_print(v, policy, title=''):
    symbols = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }
    p = np.copy(policy).astype(str)
    for num, sym in symbols.items():
        p[policy == num] = sym
    print(p)
    fig, ax = plt.subplots()
    for x, y in itertools.product(range(SIZE[0]), range(SIZE[1])):
        ax.text(x, y, p[y,x], color='black', size='x-large', bbox=dict(facecolor='white', alpha=0.5))
    ax.imshow(v, cmap='gray', animated=True)
    print(v)
    plt.title(f'Value function {title}')
        
def main():
    from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
    env = FrozenLakeEnv(is_slippery=False, map_name=f'{SIZE[0]}x{SIZE[1]}')
    env.actions = [0,1,2,3]
    #Q = np.zeros((SIZE[0]*SIZE[1], len(env.actions)))
    Q = np.random.uniform(low=0, high=1, size=(SIZE[0]*SIZE[1], len(env.actions)))
    print(obtain_value_function(Q))
    episodes = 10000
    eps = .3
    Qs, episode_lengths = [], []
    """for n in range(1,15):
        plt.figure()
        for alpha in np.arange(.1, 1., .1):
            Q_learned, episode_length = n_step_sarsa(Q.copy(), n, episodes, env, alpha, 0.8, eps=eps)
            Qs.append(Q_learned)
            episode_lengths.append(episode_length)
            plt.plot(smooth(episode_length), label=f'$n=${n}, $\\alpha=${alpha:.1f}')
        plt.legend()"""
    n = 4
    Q_learned, episode_lengths_qlearning = n_step_sarsa(Q.copy(), n, episodes, env, 0.1, 0.8, eps=eps)
    plt.legend()
 
    value = obtain_value_function(Q_learned)
 
    policy = obtain_policy(Q_learned)
 
    pretty_print(value, policy, f'${n}$-step Sarsa')
    plt.show()
 
 

if __name__ == '__main__':
    main()
    