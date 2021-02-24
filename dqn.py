import random
import gridworld
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.95
LEARNING_RATE = 0.00025
EPSILON = 0.1
BATCH_SIZE = 32
NUM_EPISODES = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

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

class Model:
    def __init__(self, input_size, output_size):
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)

class DQN:
    def __init__(self, env, MEMORY_SIZE):
        self.epsilon = EPSILON
        self.env = env
        # agent and target network with copied weights
        self.q_net = DQNet(self.env.num_states, self.env.num_actions)
        self.qhat_net = DQNet(self.env.num_states, self.env.num_actions)
        self.qhat_net.load_state_dict(self.q_net.state_dict())
        # model agent learns
        self.model = Model(self.env.num_states, self.env.num_states)
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
        # choose epsilon greedy action
        if (rg.random() < self.epsilon):
            a = self.explore()
        else:
            with torch.no_grad():
                q = self.q_net(torch.Tensor(state))
            a = np.argmax(q.detach().numpy())
        return a

    def explore(self, state):
        N = len(self.memory)
        num_samples = 50
        samples = []
        for i in range(N - num_samples, N):
            # states
           samples.append(self.memory[i][0])

        least_p = np.inf
        best_a = -1
        for action in range(self.env.num_actions):
            next_state = self.model.predict(np.append(state, [[action]], axis=1))
            p = self.get_probability(next_state, samples)
            if p < least_p:
                best_a = action
                least_p = p
        return best_a
    
    def get_probability(self, state, samples):
        design = []
        for s in samples:
            design.append(s[0])
        design = np.stack(design).T
        cov = np.cov(design)
        mean = np.mean(design, axis = 1)
        p = stats.multivariate_normal.pdf(state[0], mean, cov)
        return p
    
    def update_model(self):
        # get sample transitions from memory once it is large enough
        if len(self.memory) < BATCH_SIZE:
            return
        samples = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor([t.state for t in samples])
        actions = torch.LongTensor([t.action for t in samples])
        next_states = torch.FloatTensor([t.next_state for t in samples])

        batched_inputs = np.concatenate(states, axis=0)
        batched_targets = np.concatenate(next_states, axis=0)
        self.model.fit(batched_inputs, batched_targets, epochs=1, verbose=0)

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
    env = gridworld.GridWorld(hard_version=False)

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
            (s1, r, done) = env.step(a)

            # add trajectory to memory
            dqn.remember(s, a, s1, r)

            # update variables
            s = s1
            tot_reward += np.power(GAMMA, env.num_steps) * r

            dqn.experience_replay()

            # update target network and model
            if num_steps % TARGET_UPDATE == 0:
                dqn.qhat_net.load_state_dict(dqn.q_net.state_dict())
                dqn.update_model()
        
        rewards.append(tot_reward)
        print('Episode: {}, Reward: {}'.format(i, tot_reward))

if __name__ == "__main__":
    train()