from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from collections import defaultdict
import random
import itertools
from tqdm import tqdm

SIZE = (4,4)

def eps_greedy_action(Q, s, eps=0.2):
    if np.random.rand() > eps:
        # greedy
        return np.argmax(Q[s])
    else:
        #random
        return np.random.randint(0, Q.shape[1])

def q_learning(Q, episodes, env, alpha, gamma, eps=0.2):
    episode_lengths = []
    for episode in tqdm(range(episodes), desc='Q-Learning'):
        s = env.reset()
        done = False
        episode_length = 0
        while not done:
            episode_length += 1
            a = eps_greedy_action(Q, s, eps=eps)
            s_, reward, done, _ = env.step(a)
            Q[s,a] = Q[s,a] + alpha * (reward + gamma * max(Q[s_,]) - Q[s, a])
            s = s_
        episode_lengths.append(episode_length)
    return Q, episode_lengths

def sarsa(Q, episodes, env, alpha, gamma, eps=0.2):
    episode_lengths = []
    for episode in tqdm(range(episodes), desc='SARSA'):
        s = env.reset()
        # choose a
        a = eps_greedy_action(Q, s, eps=eps)
        done = False
        episode_length = 0
        while not done:
            episode_length += 1
            s_, reward, done, _ = env.step(a)
            a_ = eps_greedy_action(Q, s_, eps=eps if episode < 20000 else .0)
            Q[s,a] = Q[s,a] + alpha * (reward + gamma * Q[s_, a_] - Q[s, a])
            s, a = s_, a_
        episode_lengths.append(episode_length)
    return Q, episode_lengths

def expected_sarsa(Q, episodes, env, alpha, gamma, eps=0.2):
    random_probability = eps/len(env.actions)
    episode_lengths = []
    for episode in tqdm(range(episodes), desc='Expected SARSA'):
        s = env.reset()
        done = False
        episode_length = 0
        while not done:
            episode_length += 1
            a = eps_greedy_action(Q, s, eps=eps)
            s_, reward, done, _ = env.step(a)
            max_action = eps_greedy_action(Q, s_, eps=eps)
            expectation = sum([(1-eps) * Q[s_, A] if A == max_action else random_probability * Q[s_, A] for A in env.actions])
            Q[s,a] = Q[s,a] + alpha * (reward + gamma * expectation - Q[s, a])
            s = s_
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
        ax.text(x, y, p[y,x], color='black', size='xx-large', bbox=dict(facecolor='white', alpha=0.5))
    ax.imshow(v, cmap='gray', animated=True)
    print(v)
    plt.title(f'Value function {title}')
        
def main():
    from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
    env = FrozenLakeEnv(is_slippery=False, map_name=f'{SIZE[0]}x{SIZE[1]}')
    env.actions = [0,1,2,3]
    Q = np.zeros((SIZE[0]*SIZE[1], len(env.actions)))
    episodes = 50000
    eps = .1
    #Q_sarsa, episode_lengths_sarsa = sarsa(Q, episodes, env, 0.1, 0.9, eps=eps)
    Q = np.zeros((SIZE[0]*SIZE[1], len(env.actions)))
    Q_qlearning, episode_lengths_qlearning = q_learning(Q.copy(), episodes, env, 0.1, 0.9, eps=eps)
    #Q = np.zeros((SIZE[0]*SIZE[1], len(env.actions)))
    #Q_expected, episode_lengths_expected = expected_sarsa(Q.copy(), episodes, env, 0.1, 0.9, eps=eps)

#    v_sarsa = obtain_value_function(Q_sarsa)
    v_qlearning = obtain_value_function(Q_qlearning)
 #   v_expected = obtain_value_function(Q_expected)


    #policy_sarsa = obtain_policy(Q_sarsa)
    policy_qlearning = obtain_policy(Q_qlearning)
    #policy_expected = obtain_policy(Q_expected)


#    pretty_print(v_sarsa, policy_sarsa, 'SARSA')
    pretty_print(v_qlearning, policy_qlearning, 'Q-Learning')
 #   pretty_print(v_expected, policy_expected, 'Expected SARSA')

    plt.figure()
 #   plt.plot(smooth(episode_lengths_sarsa), label='SARSA')
    plt.plot(smooth(episode_lengths_qlearning), label='Q-Learning')
#    plt.plot(smooth(episode_lengths_expected), label='Expected SARSA')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
    