import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """
        Build New Q-network
        :param state_size:
        :param action_size:
        :param fc1_units:
        :param fc2_units:
        """

        super(Qnetwork, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        return self.model(state)



class Duel_Qnetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """

        :param state_size:
        :param action_size:
        :param fc1_units:
        :param fc2_units:
        """
        super(Duel_Qnetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.fc_adv1 = nn.Linear(fc2_units, 32)
        self.fc_adv2 = nn.Linear(32, action_size)

        self.fc_val1 = nn.Linear(fc2_units, 32)
        self.fc_val2 = nn.Linear(32, 1)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)

        val = F.relu(self.fc_val1(x))
        val = self.fc_val2(val)


        q_value = val - adv
        return q_value

