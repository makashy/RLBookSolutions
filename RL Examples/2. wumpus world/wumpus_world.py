""" A Wumpus World Simulator"""

from random import randrange

import ipywidgets as widgets
from ipywidgets import interact_manual
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

PIT = 9
WUMPUS = 8
DEAD_WUMPUS = 7
ARROW = 6
GOAL = 5
AGENT = 1
NULL = 0

REWARD_WIN = 100
REWARD_DEATH = -100
REWARD_WUMPUS_KILLING = 20
REWARD_ARROW_FINDING = 5

MOVE_DIRECTION = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

# pylint: disable=invalid-name


class agent():
    """ simulator of wumpus world"""

    def __init__(self, length, width, pit_num, wumpus_num, arrow_num):

        if length * width < pit_num + wumpus_num + arrow_num + 1 + 1:
            raise NameError(
                "Number of pits, wumpuses, and arrows exceed the size of the map! Initiate another world.")
        self.length = length
        self.width = width
        self.pit_num = pit_num
        self.wumpus_num = wumpus_num
        self.arrow_num = arrow_num
        self.goal_location = (length-1, width-1)
        self.agent_init_location = (0, 0)
        self.map = self.map_generator(
            length, width, pit_num, wumpus_num, arrow_num)
        self.episode_map = self.map.copy()
        self.agent = (0, 0)
        self.agent_arrow = 0
        self.arrow_possession_states = 2
        self.num_action = 4  # up, down, left, right
        self.A = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def debug(self, data):
        print(data)
        clear_output(wait=True)

    def random_location(self, forbiden_locations, length, width):
        """ Generates a random location without interference with previous locations"""
        while True:
            location = (randrange(0, length), randrange(0, width))
            if location not in forbiden_locations:
                return location

    def map_generator(self, length, width, pit_num, wumpus_num, arrow_num):
        """ Creates a map """
        wumpus_map = np.zeros(shape=(length, width))

        pit_loc = tuple()
        for _ in range(pit_num):
            pit_loc = pit_loc + (self.random_location(pit_loc + (self.goal_location,
                                                                 self.agent_init_location),
                                                      self.length, self.width),)

        arrow_loc = tuple()
        for _ in range(arrow_num):
            arrow_loc = arrow_loc + (self.random_location(arrow_loc + pit_loc +
                                                          (self.goal_location,
                                                           self.agent_init_location),
                                                          self.length, self.width),)

        wumpus_loc = tuple()
        for _ in range(wumpus_num):
            wumpus_loc = wumpus_loc + (self.random_location(wumpus_loc + arrow_loc + pit_loc +
                                                            (self.goal_location,
                                                             self.agent_init_location),
                                                            self.length, self.width),)

        wumpus_map[self.goal_location] = GOAL

        for pit in pit_loc:
            wumpus_map[pit] = PIT

        for arrow in arrow_loc:
            wumpus_map[arrow] = ARROW

        for wumpus in wumpus_loc:
            wumpus_map[wumpus] = WUMPUS

        return wumpus_map

    def deterministic_move(self, location, move, playing_game=False):

        new_location = tuple(np.array(location) + np.array(self.A[move]))

        if new_location[0] < 0 or \
           new_location[1] < 0 or \
           new_location[0] > self.length-1 or \
           new_location[1] > self.width-1:

            new_location = tuple(location)

        if playing_game:
            self.agent = new_location

        # immediate reward
        reward = -1

        if self.episode_map[new_location] == PIT:
            reward = REWARD_DEATH

        if self.episode_map[new_location] == WUMPUS:
            if self.agent_arrow > 0:
                reward = REWARD_WUMPUS_KILLING
                self.episode_map[new_location] = DEAD_WUMPUS
                self.agent_arrow = self.agent_arrow - 1
                if playing_game:
                    print('you killed an wumpus!')

            else:
                reward = REWARD_DEATH
                if playing_game:
                    print("YOU LOSE!")

        if self.episode_map[new_location] == GOAL:
            reward = REWARD_WIN
            if playing_game:
                print("YOU WON!")

        if self.episode_map[new_location] == ARROW:
            self.agent_arrow = self.agent_arrow + 1
            self.episode_map[new_location] = NULL
            reward = REWARD_ARROW_FINDING

        if reward in (REWARD_DEATH, REWARD_WIN) and playing_game:
            self.__init__(self.length, self.width, self.pit_num,
                          self.wumpus_num, self.arrow_num)

        if bool(self.arrow_num):
            new_state = new_location + (1,)
        else:
            new_state = new_location + (0,)
        # new_state = new_location + (bool(self.arrow_num),)
        # print("new_state"+str(new_state))

        return new_state, reward

    def plot_map(self, playing_game=False):
        """ Plots current state of wumpus world"""
        plot_map = self.map.copy()
        if playing_game:
            plot_map = self.episode_map.copy()
            plot_map[self.agent] = AGENT
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(plot_map)
        plt.gca().invert_yaxis()
        ax.set_xticks(np.arange(0.5, self.length + 0.5, 1))
        ax.set_yticks(np.arange(0.5, self.width + 0.5, 1))
        plt.grid()

    def manual_game(self, move_option):
        move = MOVE_DIRECTION[move_option]
        self.deterministic_move(self.agent, move, playing_game=True)
        self.plot_map(playing_game=True)

    def interactive_game(self):
        """ Builds a set of buttons for playing the game"""
        radio_buttons = widgets.RadioButtons(
            options=['UP', 'RIGHT', 'LEFT', 'DOWN'],
            description='movement:',
            disabled=False
        )
        interact_manual(self.manual_game, move_option=radio_buttons)

    def episode_generator(self):

        # first state
        s = (0, 0, 0)
        # a log list for episode
        episode = list()

        while True:
            a = np.int16(np.random.uniform(0, 4))
            s, r = self.deterministic_move(s[:-1], a)
            episode.append(s + (a, r, 1.0/4))
            if r in (REWARD_DEATH, REWARD_WIN):
                return episode

    def off_policy_MC_control(self,
                              iteration,
                              Q=None,
                              C=None,
                              G_log=None,
                              gamma=1):

        if Q is None:
            Q = np.random.rand(self.length, self.width,
                               self.arrow_possession_states, self.num_action)
        if C is None:
            C = np.zeros(
                shape=(self.length, self.width, self.arrow_possession_states, self.num_action))
        if G_log is None:
            G_log = np.zeros(0)
        pi = np.argmax(Q, axis=3)

        count = 0
        # 2. Policy Evaluation  & Policy Improvement
        while count < iteration:
            self.agent_arrow = 0
            self.episode_map = self.map.copy()
            # self.debug(count / np.float(iteration))
            gamma = 1
            count = count + 1
            episode = self.episode_generator()
            G = 0
            W = 1
            for e in reversed(episode):
                r = e[4]
                self.debug(r)
                a = e[3]
                p = e[5]
                G = r + gamma*G
                C[e[:-2]] = C[e[:-2]] + W
                Q[e[:-2]] = Q[e[:-2]] + W/C[e[:-2]]*(G - Q[e[:-2]])
                pi[e[:-3]] = np.argmax(Q[e[:-3]])
                if pi[e[:-3]] != a:
                    break
                W = W * 1 / p
            G_log = np.append(G_log, G)
        return Q, C, G_log, pi

    def epsilon_greedy(self, Q, s, num_action, epsilon):
        if np.random.uniform() < epsilon:
            a = randrange(0, num_action)
        else:
            a = Q[s].argmax()
        return a

    def sarsa_control(self,
                      iteration,
                      Q=None,
                      gamma=1,
                      alpha=0.5,
                      epsilon=0.1,
                      G_log=None):

        if Q is None:
            Q = np.random.rand(self.length, self.width,
                               self.arrow_possession_states, self.num_action)
        if G_log is None:
            G_log = np.zeros(0)

        count = 0
        # 2. Policy Evaluation  & Policy Improvement
        while count < iteration:
            self.agent_arrow = 0
            self.episode_map = self.map.copy()
            self.debug(count / np.float(iteration))
            count = count + 1

            # Initialize S
            s = (0, 0, 0)
            # Choosing A from S using policy derived from Q
            a = self.epsilon_greedy(Q, s, self.num_action, epsilon)

            r = -1
            G = 0
            while r not in (REWARD_DEATH, REWARD_WIN):
                # Taking action A, observe R, S'
                s_n, r = self.deterministic_move(s[:-1], a)

                G = G + r

                # Choosing A' from S' using policy derived from Q
                a_n = self.epsilon_greedy(Q, s_n, self.num_action, epsilon)

                Q[s + (a,)] = Q[s + (a,)] + alpha * \
                    (r + gamma * Q[s_n + (a_n,)] - Q[s + (a,)])
                s = s_n
                a = a_n

            G_log = np.append(G_log, G)

        return Q, G_log

    def Q_learning_control(self,
                           iteration,
                           Q=None,
                           gamma=1,
                           alpha=0.5,
                           epsilon=0.1,
                           G_log=None):

        if Q is None:
            Q = np.random.rand(self.length, self.width,
                               self.arrow_possession_states, self.num_action)
        if G_log is None:
            G_log = np.zeros(0)

        count = 0
        # 2. Policy Evaluation  & Policy Improvement
        while count < iteration:
            self.agent_arrow = 0
            self.episode_map = self.map.copy()
            self.debug(count / np.float(iteration))
            count = count + 1

            # Initialize S
            s = (0, 0, 0)
            r = -1
            G = 0
            while r not in (REWARD_DEATH, REWARD_WIN):
                # Choosing A from S using policy derived from Q
                a = self.epsilon_greedy(Q, s, self.num_action, epsilon)
                # Taking action A, observe R, S'
                s_n, r = self.deterministic_move(s[:-1], a)
                G = G + r
                Q[s + (a,)] = Q[s + (a,)] + alpha * \
                    (r + gamma * Q[s_n].max() - Q[s + (a,)])

                s = s_n

            G_log = np.append(G_log, G)

        return Q, G_log
