import random
#import gridworld
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_minigrid.wrappers import *

GAMMA = 0.95
LEARNING_RATE = 0.00025
EPSILON = 0.1
BATCH_SIZE = 32
NUM_EPISODES = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class FlatFullyObsWrapper(gym.core.ObservationWrapper):
    """Fully observable gridworld returning a flat grid encoding."""

    def __init__(self, env):
        super().__init__(env)

        # Since the outer walls are always present, we remove left, right, top, bottom walls
        # from the observation space of the agent. There are 3 channels, but for simplicity,
        # we will deal with flattened version of state.
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=((self.env.width) * (self.env.height) * 3,),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        # this method is called in the step() function to get the observation
        # we provide code that gets the grid state and places the agent in it
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        flattened_grid = full_grid.flatten()

        return flattened_grid

class DQNet(nn.Module):
  def __init__(self, input_size, output_size):
    super(DQNet, self).__init__()

    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, output_size)

  def forward(self, x):
    x = self.fc1(x)
    x = torch.tanh(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    x = self.fc4(x)
    return x

class Transition:
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class DQN:
    def __init__(self, env, MEMORY_SIZE):
        self.epsilon = EPSILON
        self.env = env
        self.num_states = self.env.height * self.env.width * 3
        # agent and target network with copied weights
        self.q_net = DQNet(self.num_states, self.env.action_space.n)
        self.qhat_net = DQNet(self.num_states, self.env.action_space.n)
        self.qhat_net.load_state_dict(self.q_net.state_dict())
        # model agent learns
        self.model = Model(self.num_states + 1, self.num_states)
        # adam optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        # memory storage
        self.memory = []
        self.capacity = MEMORY_SIZE
        self.position = 0

    def remember(self, state, action, next_state, reward):
        # if memory is at capacity, replace instead of append
        if len(self.memory) >= self.capacity:
            self.memory[self.position] = Transition(state, action, next_state, reward)
        else:
            self.memory.append(Transition(state, action, next_state, reward))
        self.position = (self.position + 1) % self.capacity

    def act(self, state):
        # get random generator
        rg = np.random.default_rng()
        # choose epsilon greedy action
        if (rg.random() < self.epsilon):
            a = self.explore(state)
        else:
            with torch.no_grad():
                q = self.q_net(torch.Tensor(state))
            a = np.argmax(q.detach().numpy())
        return a

    def explore(self, state):
        N = len(self.memory)
        num_samples = min(N, 50)
        samples = []
        for i in range(N - num_samples, N):
            # states
           samples.append(self.memory[i].state)

        least_p = np.inf
        best_a = -1
        for action in range(self.env.action_space.n):
            next_state = self.model(torch.Tensor(np.append(state, [[action]])))
            p = self.get_probability(next_state, samples)
            if p < least_p:
                best_a = action
                least_p = p
        return best_a
    
    def get_probability(self, state, samples):
        # samples = np.stack(samples).T
        # cov = np.cov(samples)
        # mean = np.mean(samples)
        # p = stats.multivariate_normal.pdf(state, mean, cov)
        p = random.randrange(0, 1)
        return p
    
    def update_model(self):
        # get sample transitions from memory once it is large enough
        if len(self.memory) < BATCH_SIZE:
            return
        samples = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor([t.state for t in samples])
        actions = torch.LongTensor([t.action for t in samples])
        next_states = torch.FloatTensor([t.next_state for t in samples])
        batched_inputs = states.flatten()
        batched_targets = next_states.flatten()
        #self.model.fit(batched_inputs, batched_targets, epochs=1, verbose=0)

    def experience_replay(self):
        # get sample transitions from memory once it is large enough
        if len(self.memory) < BATCH_SIZE:
            return
        batches = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor([t.state for t in batches])
        actions = torch.LongTensor([t.action for t in batches])
        next_states = torch.FloatTensor([t.next_state for t in batches])
        rewards = torch.FloatTensor([t.reward for t in batches])

        q = self.q_net(states).gather(1, actions.unsqueeze(1)).view(-1)
        with torch.no_grad():
            q_hat = self.qhat_net(next_states).max(1)[0]
        y = rewards + GAMMA * q_hat

        # optimize
        loss = nn.MSELoss()(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(TARGET_UPDATE=1000, MEMORY_SIZE=100000):
    TARGET_UPDATE = TARGET_UPDATE
    MEMORY_SIZE = MEMORY_SIZE

    # create environment
    # env = gridworld.GridWorld(hard_version=False)
    env = gym.make('MiniGrid-Empty-8x8-v0')
    # env = FullyObsWrapper(env)
    env = FlatFullyObsWrapper(env)
    # env = ImgObsWrapper(env)

    dqn = DQN(env, MEMORY_SIZE)

    num_steps = 0
    rewards = []
    print('Starting training...')
    for i in range(NUM_EPISODES):
        tot_reward = 0
        s = env.reset()
        # simulate until episode is done
        done = False
        while not done:
            num_steps += 1

            # determine and take action
            a = dqn.act(s)
            s1, r, done, info = env.step(a)

            # add trajectory to memory
            dqn.remember(s, a, s1, r)

            # update variables
            s = s1
            tot_reward += np.power(GAMMA, num_steps) * r

            dqn.experience_replay()

            # update target network and model
            if num_steps % TARGET_UPDATE == 0:
                dqn.qhat_net.load_state_dict(dqn.q_net.state_dict())
                dqn.update_model()
        
        rewards.append(tot_reward)
        if i % 10 == 0:
            print('Episode: {}, Reward: {}'.format(i, tot_reward))

if __name__ == "__main__":
    train()