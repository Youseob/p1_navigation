from model import Qnetwork, Duel_Qnetwork
from experience_buffer import Replay_buffer

import torch
import numpy as np
import random


from torch import optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
LOCAL_UPDATE = 4
TAU = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():

    def __init__(self, state_size, action_size, double = False, duel=False):

        self.state_size = state_size
        self.action_size = action_size
        self.discounted_factor = 0.99
        self.learning_rate = 0.001

        self.double = double

        # Define Model
        if duel:
            self.local_model = Duel_Qnetwork(state_size, action_size).to(device)
            self.target_model = Duel_Qnetwork(state_size, action_size).to(device)
        else:
            self.local_model = Qnetwork(state_size, action_size).to(device)
            self.target_model = Qnetwork(state_size, action_size).to(device)

        # Define optimizer
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.learning_rate)

        # Define Buffer
        self.buffer = Replay_buffer(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

        # time_step, local_model update, target_model update
        self.t_step = 0
        self.target_update_t = 0

    def get_action(self, state, eps=0.0):
        """state (numpy.ndarray)"""
        state = torch.from_numpy(state.reshape(1, self.state_size)).float().to(device)

        self.local_model.eval()
        with torch.no_grad():
            action_values = self.local_model(state)  # .detach().cpu()
        self.local_model.train()

        # epsilon greedy policy
        if random.random() < eps:
            action = np.random.randint(4)
            return action
        else:
            action = np.argmax(action_values.cpu().data.numpy())

            return int(action)

    def append_sample(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step += 1
        if self.t_step % LOCAL_UPDATE == 0:
            """If there are enough experiences"""
            if self.buffer.__len__() > BATCH_SIZE:
                experiences = self.buffer.sample()
                self.learn(experiences)

                # self.target_update_t += 1
                # if self.target_update_t % TARGET_UPDATE == 0:
                self.soft_target_model_update(TAU)

    def learn(self, experiences):
        """experiences ;tensor  """
        states, actions, rewards, next_states, dones = experiences

        pred_q = self.local_model(states).gather(1, actions)

        if self.double:
            _, argmax_actions =torch.max(self.local_model.forward(next_states).detach(), 1, keepdim=True)
            pred_next_q = self.target_model.forward(next_states).gather(1, argmax_actions)
        else :
            pred_next_q, _ = torch.max(self.target_model.forward(next_states).detach(), 1, keepdim=True)

        target_q = rewards + ((1 - dones) * self.discounted_factor * pred_next_q)
        loss = F.mse_loss(target_q, pred_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_target_model_update(self, tau):
        for target_param, local_param in zip(self.target_model.parameters(), self.local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)