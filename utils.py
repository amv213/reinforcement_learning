"""

Core function definitions for supervised learning

"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec


class Maze:
    """
    A maze environment for unsupervised learning agents.
    """

    def __init__(self, width=81, height=51, complexity=.75, density=.75):
        """Initializes a random maze environment.

        Args:
            width (int):        desired width of the maze
            height (int):       desired height of the maze
            complexity (float): maze complexity parameter
            density (float):    maze wall density parameter

        Attributes:
            z (array):              2D matrix representing the maze
            shape (tuple):          utility attribute for the actual shape of the maze created
            target_ico (string):  filename of icon to use for the target reward
            target_i (int):         row index of the target position
            target_j (int):         column index of the target position

        Returns:
            _ (Maze):    initialized instance of the Maze class
        """

        # make a maze, convert array from True/False to integer:
        self.z = self.build_maze(width, height, complexity, density)
        self.shape = self.z.shape

        # Place Target/Reward
        self.target_ico = "imgs/bread.png"
        self.target_i, self.target_j = self.random_available_ij()

    def __repr__(self):
        """
        A text representation of the maze.
        """
        return f"Maze {self.shape} with target at {(self.target_i, self.target_j)}\n\n{self.z}"

    @staticmethod
    def build_maze(width, height, complexity, density):
        """Creates a matrix representation of a maze.

        Ref:
            Wikipedia's maze building algorithm

        Args:
            width (int):        desired width of the maze
            height (int):       desired height of the maze
            complexity (float): maze complexity parameter
            density (float):    maze wall density parameter

        Returns:
            z (array):          2D matrix representing the maze
        """

        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)

        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # number of components
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # size of components

        # Build actual maze
        Z = np.zeros(shape, dtype=bool)

        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1

        # Make aisles
        for i in range(density):

            # pick a random position
            x = np.random.randint(0, shape[1] // 2) * 2
            y = np.random.randint(0, shape[0] // 2) * 2
            Z[y, x] = 1

            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.randint(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        return np.array(Z, dtype="int")

    def random_available_ij(self):
        """Selects random matrix indices not occupied by a wall.

        Returns:
            i (int):    row matrix index of random available position
            j (int):    column matrix index of random available position
        """

        i, j = np.random.choice(self.shape[0]), np.random.choice(self.shape[1])

        while self.z[i][j] != 0:  # while not on an empty space ...
            i, j = self.random_available_ij()  # ... try again

        return i, j

    def render_target(self, ax):
        """Plots an icon representing the target at its position in the maze.

        Args:
            ax (axes):  axes onto which to plot

        Returns:
            ax (axes):  updated plot axes
        """

        target_img = OffsetImage(plt.imread(self.target_ico), zoom=5 / ax.figure.dpi, dpi_cor=False)
        target_ab = AnnotationBbox(target_img, (self.target_j, self.target_i), frameon=False)  # express (x,y) as (j,i)
        ax.add_artist(target_ab)
        return ax

    def render(self, ax=None):
        """Plots the maze and displays the position of the target.

        Args:
            ax (axes, optional): axes onto which to plot

        Returns:
            ax (axes):  updated plot axes
        """

        if ax is None:
            ax = plt.gca()

        # Plot maze
        ax.imshow(self.z, origin='upper', vmin=0, vmax=3)  # scaling chosen for aesthetic reasons

        # Plot target
        ax = self.render_target(ax)

        ax.axis('off')

        return ax


class Agent:
    """
    An unsupervised leaning agent.
    """

    def __init__(self, maze, na=4, q=None):
        """Initializes a randomized agent evolving in a given environment.

        Args:
            maze (Maze):            maze in which the agent is trained
            na (int):               size of the agent's action space
            q (numpy.ndarray):      array of Q-values of shape (num_states, num_actions)

        Attributes:
            maze (Maze):            maze in which the agent is trained
            i (int):                row matrix index of the agent's current position
            j (int):                column matrix index of the agent's current position
            i_prev (int):           row matrix index of the agent's previous position
            j_prev (int):           column matrix index of the agent's previous position
            q (numpy.ndarray):      array of current Q-values, of shape (num_states, num_actions)
            a (int):                action taken on previous step to get to current state
            r (int):                reward awarded to the agent for reaching current state
            on_target (bool):       True if agent is on target. False otherwise.
            agent_ico (string):     filename of icon to use for the agent's avatar

        Returns:
            _ (Agent):              initialized instance of the Agent class

        """

        self.agent_ico = 'imgs/nerd.png'

        self.i_prev, self.j_prev = (None, None)

        self.maze = maze
        self.num_actions = na

        self.i, self.j = self.maze.random_available_ij()
        while (self.i, self.j) == (self.maze.target_i, self.maze.target_j):
            self.i, self.j = self.maze.random_available_ij()

        if q is None:
            self.q = np.zeros([self.maze.shape[0], self.maze.shape[1], self.num_actions])
        else:
            self.q = q

        self.a = None

        self.on_target = self.check_is_on_target()

        self.r = self.calc_reward()


    def __repr__(self):
        """Provides useful informations about the agent
        """

        return f"Agent now at {(self.i, self.j)} took action {self.a} from {(self.i_prev, self.j_prev)} and got " \
               f"rewarded {self.r} | Policy table {self.q.shape}"

    def check_is_on_target(self):
        """Checks if the agent is on target.

        Returns:
            bool:   True if agent is on target. False otherwise.
        """
        if (self.i, self.j) == (self.maze.target_i, self.maze.target_j):
            return True

        else:
            return False

    def available_actions(self):
        """Returns the actions available to the agent from its current position.

        Returns:
            available_a (list): a list of available actions, of size <= num_actions
        """

        d = {(-1, 0): 0, (1, 0): 1, (0, 1): 2, (0, -1): 3}  # U, D, R, L

        # All actions available unless lead to wall
        available_a = [d[k] for k in d if self.maze.z[self.i + k[0], self.j + k[1]] == 0]

        return available_a

    def pick_action(self, eps):
        """Makes agent pick an action given its current state.

        Args:
            eps (float):    exploration threshold. The agent will take a stochastic action with probability eps. Else
                            it will choose the action with highest Q-value.

        Sets:
            a (int):        the agent's action picked given its current state
        """

        actions = self.available_actions()  # Actions available to the agent in this state

        p = np.random.random()  # stochastic exploration probability

        # Stochastic exploration
        if p < eps:

            self.a = np.random.choice(actions)

        # Q-table exploit
        else:

            avaq = [self.q[self.i, self.j, ava] for ava in actions]  # find Q of available actions
            self.a = actions[np.argmax(avaq)]  # select available action with maximum Q

    def execute_action(self):
        """Makes the agent take an action in its environment.

        Sets:
            agent's attributes linked to actions.
        """

        # Mapping between action indices and actual actions in the environment
        mov = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # U, D, R, L

        # Flush agent's current position as previous position
        self.i_prev = self.i
        self.j_prev = self.j

        # Update agent's current position
        self.i += mov[self.a][0]
        self.j += mov[self.a][1]

    def act(self, eps):
        """ Makes the agent pick and execute an action in its environment. The agent transitions to a new state.

        Args:
            eps (float):    exploration threshold. The agent will take a stochastic action with probability eps. Else
                            it will choose the action with highest Q-value.

        Updates:
            Agent's state.

        """

        self.pick_action(eps)  # Pick an action based on the exploration threshold

        self.execute_action()  # Execute the action in the environment. State s --> State s+1

        self.on_target = self.check_is_on_target()  # Check if agent on target

    def calc_reward(self):
        """ Calculates the reward for a given state of the agent.

        Returns:
            r (int):    value of the reward
        """

        # Reward if agent is on target
        if self.on_target:
            r = 10

        # Else don't reward
        else:
            r = 0

        return r

    def reward(self):
        """Rewards the agent for its current state.

        Sets:
            r (int):    reward awarded to the agent for reaching current state
        """

        self.r = self.calc_reward()

    def update_q(self, alpha, gamma):
        """Updates the Q-table for agent in current state according to the Bellman equation.

        Args:
            alpha (float):  learning rate, keep low if evolving in stochastic environments (0.1)
            gamma (float):  discount rate, keep high to make reward glow more (0.9)

        Updates:
            q (numpy.ndarray):  array of current Q-values, of shape (num_states, num_actions)
        """
        q_max = np.max(self.q[self.i, self.j, :])  # maximum Q that agent can see from his current position

        # Update Q-value of agent's previous state according to the Bellman equation
        self.q[self.i_prev, self.j_prev, self.a] = (1 - alpha) * self.q[self.i_prev, self.j_prev, self.a] + alpha * (
                    self.r + gamma * q_max)

    def respawn(self):
        """Respawns the agent in the environment.

        Returns:
            new_agent (Agent):  a new Agent instance initialized with the old agent's experience.
        """

        new_agent = Agent(maze =self.maze, na=self.num_actions, q=self.q)

        return new_agent

    # -----------------
    # RENDERING METHODS
    # -----------------

    def render_q(self, axs=None):
        """Visualization of the policy table for each action. And the optimal policy table for all actions.

        Args:
            axs (list of axes):  list of (num_actions + 1) axes onto which to plot

        Returns:
            axs (axes):  updated plot axes
        """

        vmax = np.max(self.q)

        if axs is None:
            fig, axs = plt.subplots(1, 5, figsize=(12, 3))

        # Individual actions Q-values
        for i, x in enumerate(axs[:-1]):
            dirs = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}  # U, D, R, L
            x.imshow(self.q[:, :, i] - self.maze.z, origin='upper', vmin=-1, vmax=vmax)  # also see maze walls
            x.set_title(f"{dirs[i]}")
            x.axis('off')

        # Best Q-Value for any given state
        axs[-1].imshow(np.max(self.q, axis=-1) - self.maze.z, origin='upper', vmin=-1, vmax=vmax)
        axs[-1].axis('off')

        return axs

    def render_q_here(self, ax=None):
        """Visualization of the policy for all actions available to the agent in the current state.

        Args:
            ax (axes):  axes onto which to plot

        Returns:
            ax (axes):  updated plot axes
        """
        if ax is None:
            ax = plt.gca()

        vmax = np.max(self.q)  # normalize color scale by maximum policy in table

        qs_here = [np.array(self.q[self.i, self.j, :])]  # policies for current state
        ax.imshow(np.roll(qs_here, 1), vmin=-1, vmax=vmax)  # roll array so to have actions in human-friendly order L,U,D,R

        # Set plot ticks
        ax.set_yticks([])
        ax.set_xticks(range(self.num_actions))
        ax.set_xticklabels(["L", "U", "D", "R"], fontsize=16)  # note we rolled values

        # Remove axes frames
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        return ax

    def render(self, ax=None):
        """Plots an icon representing the agent at its position in the environment.

        Args:
            ax (axes):  axes onto which to plot

        Returns:
            ax (axes):  updated plot axes
        """
        if ax is None:
            ax = plt.gca()

        agent_ico = OffsetImage(plt.imread(self.agent_ico), zoom=4 / ax.figure.dpi, dpi_cor=False)
        agent_ab = AnnotationBbox(agent_ico, (self.j, self.i), frameon=False)  # flipped x/y
        ax.add_artist(agent_ab)

        return ax


def plot_dashboard(agent, maze, episode=0):
    """Plots a dashboard to visualize the supervised learning training in a maze for a given episode.

    Args:
        agent (Agent): the agent object at given episode
        maze (Maze): the environment in which the agent has been trained
        episode (int): training episode number

    Returns:
        *ax (axes): a multi-axes plot
    """

    fig = plt.figure(figsize=(15, 9))

    fig.suptitle(f"Epoch {episode}")  # Print episode number

    # Blueprint, setup subplots layout
    gs1 = GridSpec(3, 5)  # num_rows, num_columns

    ax1 = fig.add_subplot(gs1[0:2, 0:2])    # axes for environment display
    ax2 = fig.add_subplot(gs1[0:2, 2:4])    # axes for optimal policy table
    ax3 = fig.add_subplot(gs1[-1, :2])      # axes for current state policy across all actions
    ax4 = fig.add_subplot(gs1[-1, 2])       # axes for 'LEFT' policy table
    ax5 = fig.add_subplot(gs1[-1, 3])       # axes for 'RIGHT' policy table
    ax6 = fig.add_subplot(gs1[0:1, 4])      # axes for 'UP' policy table
    ax7 = fig.add_subplot(gs1[1:2, 4])      # axes for 'DOWN' policy table
    ax8 = fig.add_subplot(gs1[-1, 4])       # axes for agent's avatar

    # Maze and agent
    ax1 = maze.render(ax1)
    agent.render(ax1)

    # Policy tables
    axs = [ax6, ax7, ax5, ax4, ax2]  # U, D, R, L, TOT
    agent.render_q(axs)

    # Current policy across all actions
    agent.render_q_here(ax3)

    # Avatar icon
    ax8.imshow(plt.imread(agent.agent_ico))
    ax8.axis("off")

    return ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8


def update_eps(i, min_eps, max_eps, eps_tau):
    """Calculate the optimal value for the exploration threshold at given episode, following a decaying exponential
    model.

    Args:
        i (int):    episode number
        min_eps:    minimum exploration threshold allowed
        max_eps:    maximum exploration threshold allowed
        eps_tau:    desired exploration threshold lifetime (in number of episodes)

    Returns:
        eps (float):    exploration threshold. The agent will take a stochastic action with probability eps. Else
                        it will choose the action with highest Q-value.
    """

    return min_eps + (max_eps - min_eps) * np.exp(-i / eps_tau)


if __name__ == "__main__":

    world = Maze(width=16, height=16)

    print(world)

    world.render()
    plt.show()
