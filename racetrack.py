
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

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
        map_shape: Shape of map
        goal: Location of goal
        theta: A small threshold determining accuracy of estimation
        obstacle_list: List of obstacles
        alpha: Probability of random obstacle
        slip_probability: The probability of slip in the direction of movement
                          (making the environment nondeterministic)
    """

    def __init__(self,
                 grid_map,
                 theta=0.001,
                 constant_speed_probability=0):
        self.grid_map = grid_map
        self.theta = theta
        self.constant_speed_probability = constant_speed_probability
        self.A = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        self.S = [(x, y, vx, vy) for x in range(0, self.grid_map.shape[0]) for y in range(0, self.grid_map.shape[1])
                  for vx in range(-5, 6) for vy in range(-5, 6)]

    def random_start(self):
        start_points = np.where(self.grid_map == START)
        index = np.int16(np.random.uniform(0, start_points[0].shape[0]))
        return (start_points[0][index], start_points[1][index])

    def deterministic_move(self, s, a):

        # next velocity
        v_n = tuple(np.array(self.S[s][2:]) + np.array(self.A[a]))
        # velocity limmit between 0 and 5
        v_n = (min(5, max(-5, v_n[0])), min(5, max(-5, v_n[1])))
        # next location
        l_n = tuple(np.array(self.S[s][:2]) + np.array(v_n))
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
        s_n = self.S.index(l_n + v_n)
        return s_n, r

    def environment(self, s, a):
        if np.random.uniform() < self.constant_speed_probability:
            return self.deterministic_move(s, ZERO_ACCELERATION)
        return self.deterministic_move(s, a)

    def episode_generator(self):

        # first state (random)
        s = self.S.index(self.random_start() + (0, 0))
        # a log list for episode
        episode = list()

        while True:
            a = np.int16(np.random.uniform(0, 9))
            s, r = self.environment(s, a)
            episode.append((s, a, r, 1.0/9))
            if self.grid_map[self.S[s][:2]] == GOAL:
                return episode

    # def value_iteration(self):

    #     # 1. Initialization
    #     S = [(x, y, vx, vy) for x in range(0, self.grid_map.shape[0]) for y in range(0, self.grid_map.shape[1])
    #          for vx in range(0, 11) for vy in range(0, 11)]

    #     # V = np.random.rand(self.grid_map.size[0], self.grid_map.size[1], 6, 6)
    #     Q = np.zeros(shape=(self.grid_map[0], self.grid_map[1], 11, 11, 9))
    #     C = np.zeros(shape=(self.grid_map[0], self.grid_map[1], 11, 11, 9))
    #     pi = [np.int16(np.random.uniform(0, 10)) for state in S]

    #     x = 0
    #     # 2. Policy Evaluation  & Policy Improvement
    #     while True:
    #         delta = 0.0
    #         x = x + 1
    #         episode = self.episode_generator
    #         G = 0
    #         W = 1
    #         for e in reversed(episode):
    #             r = e[6]
    #             s = e[:4]
    #             G = r + gamma*G
    #             # C[] = C + W

    #             if self.map[s] == GOAL:
    #                 V[s] = 0
    #             elif self.map[s] == OBSTACLE:
    #                 V[s] = -1
    #             else:
    #                 maximum_pi = 0
    #                 for a in self.A:
    #                     summation = 0
    #                     state_reward = self.environment(s, a)
    #                     for s_r in state_reward:
    #                         summation = summation + s_r[2] * (
    #                             s_r[1] + 0.9 * V[s_r[0]])
    #                     Q[s[0], s[1], self.A.index(a)] = summation
    #                     if summation > maximum_pi:
    #                         maximum_pi = summation
    #                         V[s] = summation
    #                         pi[s] = self.A.index(a)

    #             delta = max(delta, abs(v - V[s]))

    #         pi_set = np.append(pi_set, [pi], 0)
    #         V_set = np.append(V_set, [V], 0)
    #         if delta < self.theta:
    #             break

    #         if x > 5000:
    #             print("Probably there are some closed area in the map!")
    #             break
    #     return V_set, pi_set, Q





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
