""" RL Project 1"""

import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

OBSTACLE = 1
GOAL = 2

P_REWARD = 10
N_REWARD = 0

# pylint: disable=invalid-name


class GridAgent():
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
                 map_shape=(10, 10),
                 goal=(20, 20),
                 theta=0.001,
                 obstacle_list=None,
                 alpha=0,
                 slip_probability=0):
        self.map_shape = (map_shape[0] + 2, map_shape[1] + 2)
        self.theta = theta
        self.goal = (goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.alpha = alpha
        self.slip_probability = slip_probability
        self.map = self.map_creator(self.map_shape)
        self.A = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def debug(self, data):
        print(data)
        # sleep(0.5)
        clear_output(wait=True)

    def map_creator(self, map_shape):
        gridmap = np.zeros(map_shape, dtype=np.int32)
        gridmap[0, :] = OBSTACLE
        gridmap[-1, :] = OBSTACLE
        gridmap[:, 0] = OBSTACLE
        gridmap[:, -1] = OBSTACLE
        gridmap[self.goal] = GOAL

        # random obstacles
        for x_value in range(1, map_shape[0] - 1):
            for y_value in range(1, map_shape[1] - 1):
                if np.random.uniform() < self.alpha:
                    gridmap[x_value, y_value] = OBSTACLE

        # specific obstacles
        if self.obstacle_list is not None:
            for obstacle in self.obstacle_list:
                gridmap[obstacle] = OBSTACLE

        return gridmap

    def deterministic_move(self, s, a, slip, p):
        for i in range(1, slip + 2):

            # next state
            s_n = tuple(np.array(s) + np.array(a) * i)

            # immediate reward
            r = N_REWARD

            if self.map[s_n] == OBSTACLE:
                s_n = s
                break

            if self.map[s_n] == GOAL:
                r = P_REWARD

            if self.map[s] == GOAL:
                s_n = s
                r = 0
                break

        return s_n, r, p

    def environment(self, s, a):
        case_1 = self.deterministic_move(
            s, a, slip=0, p=1 - self.slip_probability)
        case_2 = self.deterministic_move(s, a, slip=1, p=self.slip_probability)
        return case_1, case_2

    def value_iteration(self):

        # 1. Initialization
        S = [(i, j)
             for j in range(self.map_shape[0])
             for i in range(self.map_shape[1])]

        V = np.random.rand(self.map_shape[0], self.map_shape[1])
        Q = np.zeros(shape=(self.map_shape[0], self.map_shape[1], 4))
        pi = np.zeros(shape=(self.map_shape), dtype=np.int)
        pi_set = np.zeros(
            shape=(1, self.map_shape[0], self.map_shape[1]), dtype=np.int)
        V_set = np.zeros(shape=(0, self.map_shape[0], self.map_shape[1]))
        V_set = np.append(V_set, [V], 0)

        x = 0
        # 2. Policy Evaluation  & Policy Improvement
        while True:
            delta = 0.0
            x = x + 1
            for s in reversed(S):
                v = V[s]

                if self.map[s] == GOAL:
                    V[s] = 0
                elif self.map[s] == OBSTACLE:
                    V[s] = -1
                else:
                    maximum_pi = 0
                    for a in self.A:
                        summation = 0
                        state_reward = self.environment(s, a)
                        for s_r in state_reward:
                            summation = summation + s_r[2] * (
                                s_r[1] + 0.9 * V[s_r[0]])
                        Q[s[0], s[1], self.A.index(a)] = summation
                        if summation > maximum_pi:
                            maximum_pi = summation
                            V[s] = summation
                            pi[s] = self.A.index(a)

                delta = max(delta, abs(v - V[s]))

            pi_set = np.append(pi_set, [pi], 0)
            V_set = np.append(V_set, [V], 0)
            if delta < self.theta:
                break

            if x > 5000:
                print("Probably there are some closed area in the map!")
                break
        return V_set, pi_set, Q

    def arrow(self, i, j, pi):
        """ Map for policy"""
        if self.map[i, j] == 0:
            return self.A[pi[i, j]]
        return tuple([0, 0])

    def plot_result(self, pi, V):
        X = np.arange(0, self.map_shape[0], 1)
        Y = np.arange(0, self.map_shape[1], 1)
        direction = np.array(
            [[self.arrow(i, j, pi)
              for j in range(self.map_shape[1])]
             for i in range(self.map_shape[0])])

        q = plt.quiver(X, Y, direction[:, :, 1], direction[:, :, 0])
        plt.quiverkey(
            q,
            X=1.1,
            Y=1.1,
            U=10,
            label='Quiver key, length = 10',
            labelpos='E')

        plt.imshow(V)
        plt.gca().invert_yaxis()
        y = np.copy(self.goal[1])
        x = np.copy(self.goal[0])
        if x < 0:
            x = self.map_shape[0] + x
        if y < 0:
            y = self.map_shape[1] + y
        plt.plot(
            x,
            y,
            '*',
            ms=10,
            mfc='yellow',
        )
        plt.hot()
        plt.colorbar()
        plt.title("Policy in arrows, State Values in colors")
