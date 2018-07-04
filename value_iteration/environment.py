import sys
import numpy as np

class CleaningRobotEnvironment(object):
    '''
    CleaningRobotEnvironment environment

    states:
        +--+--+--+--+--+--+
        | 0| 1| 2| 3| 4| 5|
        +--+--+--+--+--+--+
    actions:
        a = -1 -> left
        a = 1  -> right
    '''

    def __init__(self):
        self.states = np.arange(6)
        self.actions = np.array([-1,1])

    def r(self, s, a, s_):
        return 5 * int(s == 4 and a == 1 and s_ == 5) + int(s == 1 and a == -1 and s_ == 0)

    def p(self, s, a, s_):
        return 1 if s_ == max(0, min(5, s + a)) else 0
