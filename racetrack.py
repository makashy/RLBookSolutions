
import numpy as np
from IPython.display import clear_output
# import matplotlib.pyplot as plt

OBSTACLE = 1
GOAL = 2
START = 3

P_REWARD = 10
N_REWARD = -1

ZERO_ACCELERATION = 4

# pylint: disable=invalid-name


class Agent():
    """Agent to find best direction to goal.

    Arguments:
        grid_map: Shape of map
        constant_speed_probability: The probability at each time step that the velocity increments are both zero
                          (making the environment nondeterministic)
    """

    def __init__(self,
                 grid_map,
                 constant_speed_probability=0):
        self.grid_map = grid_map
        self.constant_speed_probability = constant_speed_probability
        self.A = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

    def debug(self, data):
        print(data)
        # sleep(0.5)
        clear_output(wait=True)

    def random_start(self):
        """ Returning a random point from all start points"""
        start_points = np.where(self.grid_map == START)
        index = np.int16(np.random.uniform(0, start_points[0].shape[0]))
        return (start_points[0][index], start_points[1][index])

    def deterministic_move(self, s, a):
        """Returns next state and immediate reward
        follows by state s and action a"""

        # next velocity
        v_n = tuple(np.array(s[2:]) + np.array(self.A[a]))
        # velocity limmit between 0 and 5
        v_n = (min(5, max(-5, v_n[0])), min(5, max(-5, v_n[1])))
        # next location
        l_n = tuple(np.array(s[:2]) + np.array(v_n))
        # location limmit in grid_map
        l_n = (min(self.grid_map.shape[0]-1, max(0, l_n[0])),
               min(self.grid_map.shape[1]-1, max(0, l_n[1])))

        # immediate reward
        r = N_REWARD
        if self.grid_map[l_n] == OBSTACLE:
            l_n = self.random_start()
            v_n = (0, 0)
        if self.grid_map[l_n] == GOAL:
            r = P_REWARD
        s_n = l_n + v_n
        return s_n, r

    def environment(self, s, a):
        
        if np.random.uniform() < self.constant_speed_probability:
            return self.deterministic_move(s, ZERO_ACCELERATION)
        return self.deterministic_move(s, a)

    def episode_generator(self):

        # first state (random)
        s = self.random_start() + (0, 0)
        # a log list for episode
        episode = list()

        while True:
            a = np.int16(np.random.uniform(0, 9))
            s, r = self.environment(s, a)
            episode.append(s + (a, r, 1.0/9))
            if self.grid_map[s[:2]] == GOAL:
                return episode

    def value_iteration(self, iteration):

        # 1. Initialization
        Q = np.ones(
            shape=(self.grid_map.shape[0], self.grid_map.shape[1], 11, 11, 9)) * ZERO_ACCELERATION
        C = np.zeros(
            shape=(self.grid_map.shape[0], self.grid_map.shape[1], 11, 11, 9))
        pi = np.ones(
            shape=(self.grid_map.shape[0], self.grid_map.shape[1], 11, 11)) * ZERO_ACCELERATION

        x = 0
        # 2. Policy Evaluation  & Policy Improvement
        while x < iteration:
            self.debug(x/np.float(iteration))
            gamma = 0.9
            x = x + 1
            episode = self.episode_generator()
            G = 0
            W = 1
            for e in reversed(episode):
                r = e[5]
                a = e[4]
                p = e[6]
                G = r + gamma*G
                C[e[:-2]] = C[e[:-2]] + W
                Q[e[:-2]] = Q[e[:-2]] + W/C[e[:-2]]*(G - Q[e[:-2]])
                pi[e[:-3]] = np.argmax(Q[e[:-3]])
                if pi[e[:-3]] != a:
                    break
                W = W * 1 / p
        return pi

    # def arrow(self, i, j, pi):
    #     """ Map for policy"""
    #     if self.map[i, j] == 0:
    #         return self.A[pi[i, j]]
    #     return tuple([0, 0])

    # def plot_result(self, pi, V):
    #     X = np.arange(0, self.map_shape[0], 1)
    #     Y = np.arange(0, self.map_shape[1], 1)
    #     direction = np.array(
    #         [[self.arrow(i, j, pi)
    #           for j in range(self.map_shape[1])]
    #          for i in range(self.map_shape[0])])

    #     q = plt.quiver(X, Y, direction[:, :, 1], direction[:, :, 0])
    #     plt.quiverkey(
    #         q,
    #         X=1.1,
    #         Y=1.1,
    #         U=10,
    #         label='Quiver key, length = 10',
    #         labelpos='E')

    #     plt.imshow(V)
    #     plt.gca().invert_yaxis()
    #     y = np.copy(self.goal[1])
    #     x = np.copy(self.goal[0])
    #     if x < 0:
    #         x = self.map_shape[0] + x
    #     if y < 0:
    #         y = self.map_shape[1] + y
    #     plt.plot(
    #         x,
    #         y,
    #         '*',
    #         ms=10,
    #         mfc='yellow',
    #     )
    #     plt.hot()
    #     plt.colorbar()
    #     plt.title("Policy in arrows, State Values in colors")
