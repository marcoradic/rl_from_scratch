from environment import CleaningRobotEnvironment
import numpy as np


def value_iteration(env, threshold, gamma):
    # initialize
    value_function = np.zeros_like(env.states)
    policy = np.zeros_like(env.states)
    # improve
    delta = np.Infinity
    count = 0
    while delta >= threshold:
        count += 1
        delta = 0
        for s in env.states:
            old_value = value_function[s]
            action_values = []
            for a in env.actions:
                partial_val = 0
                for s_ in env.states:
                    partial_val += env.p(s, a, s_) * (env.r(s, a, s_) + gamma * value_function[s_])
                action_values.append(partial_val)
            policy[s] = env.actions[np.argmax(action_values)]
            value_function[s] = max(action_values)
            delta = max(delta, abs(old_value - value_function[s]))
    # return policy
    print(f'Termination after {count} iterations')
    return policy

def evaluate(env):
    for gamma in [0.1 * x for x in range(10)]:
        print('\ngamma = ', gamma)
        policy =  value_iteration(env, 1, gamma)
        symbols = {-1: ':arrow_left:' , 1: '→'}
        for action in policy:
            print(symbols[action] + '  ', end='')
        print('')


def main():
    """
    Exercise Sheet 3 - Ex3
    """
    environment = CleaningRobotEnvironment()
    evaluate(environment)

if __name__ == '__main__':
    main()
