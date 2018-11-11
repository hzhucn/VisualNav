import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(520, num_actions)

    def forward(self, frames, goals):
        frames = F.relu(self.conv1(frames))
        frames = F.relu(self.conv2(frames))
        frames = F.relu(self.conv3(frames))
        frames = F.relu(self.fc4(frames.view(frames.size(0), -1)))
        features = torch.cat([frames, goals.view(goals.size(0), -1)], dim=1)
        return self.fc5(features)


class DAQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18, with_region_index=False):
        """
        DQN with goal-dependent attention
        """
        super(DAQN, self).__init__()
        self.with_region_index = with_region_index
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if with_region_index:
            self.attention = nn.Sequential(nn.Linear(64 + 9, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            self.attention = nn.Sequential(nn.Linear(64 + 8, 128), nn.ReLU(), nn.Linear(128, 1))
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(520, num_actions)

        # for visualization
        self.attention_weights = None

    def forward(self, frames, goals):
        device = frames.device
        batch_size = goals.size(0)
        frames = F.relu(self.conv1(frames))
        frames = F.relu(self.conv2(frames))
        frames = F.relu(self.conv3(frames))
        # (b, 64, 7, 7) -> (b, 49, 64)
        frames = frames.view(frames.size(0), frames.size(1), -1).permute(0, 2, 1)
        # (b * 49, 64)
        frames = frames.contiguous().view(-1, frames.size(2))

        # (b, 8) -> (b, 49, 8) -> (b, 49, 9) -> (b * 49, 9)
        goals_expanded = goals.view(batch_size, -1).unsqueeze(1).expand((batch_size, 49, 8))
        if self.with_region_index:
            block_indices = torch.Tensor([i for i in range(49)]).unsqueeze(0).unsqueeze(2).to(device)
            block_indices = block_indices.expand((batch_size, 49, 1))
            queries = torch.cat([goals_expanded, block_indices], dim=2).view(-1, 9)
        else:
            queries = goals_expanded.contiguous().view(-1, 8)

        # compute attention scores (b, 49, 1)
        attention_input = torch.cat([frames, queries], dim=1)
        attention_scores = self.attention(attention_input).view(batch_size, 49, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights.data

        # compute aggregated feature
        frames = frames.view(batch_size, 49, 64)
        agg_features = torch.sum(torch.mul(frames, attention_weights), dim=1)
        frames = F.relu(self.fc4(agg_features))
        features = torch.cat([frames, goals.view(goals.size(0), -1)], dim=1)
        return self.fc5(features)


class DA1QN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        DQN with goal-dependent attention
        """
        super(DA1QN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.theta2 = nn.Conv2d(64, 32, 1)
        self.alpha = nn.Linear(8, 32)
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(520, num_actions)

        # for visualization
        self.attention_weights = None

    def forward(self, frames, goals):
        frames = F.relu(self.conv1(frames))
        frames = F.relu(self.conv2(frames))
        frames = F.relu(self.conv3(frames))

        # -> b, 32, 7, 7
        theta2_x = self.theta2(frames)
        theta2_x = theta2_x.view(-1, 32, 49)
        # -> b, 1, 32
        alpha_g = self.alpha(goals.view(-1, 8)).unsqueeze(1)
        # -> b, 49, 1, batched matrix multiplication
        attention_scores = torch.matmul(alpha_g, theta2_x).view(-1, 49, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights.data

        # compute aggregated feature
        frames = frames.view(-1, 64, 49).permute(0, 2, 1)
        agg_features = torch.sum(torch.mul(frames, attention_weights), dim=1)
        frames = F.relu(self.fc4(agg_features))
        features = torch.cat([frames, goals.view(goals.size(0), -1)], dim=1)
        return self.fc5(features)


class DA2QN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        DQN with goal-dependent attention
        """
        super(DA2QN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # first layer attention
        self.theta1 = nn.Conv2d(64, 32, 1)
        self.phi = nn.Conv2d(64, 32, 1)
        self.g = nn.Conv2d(64, 32, 1)
        self.p = nn.Conv2d(32, 64, 1)

        # second layer attention
        self.theta2 = nn.Conv2d(64, 32, 1)
        self.alpha = nn.Linear(8, 32)
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(520, num_actions)

        # for visualization
        self.attention_weights = None

    def forward(self, frames, goals):
        batch_size = goals.size(0)
        # compute feature map
        frames = F.relu(self.conv1(frames))
        frames = F.relu(self.conv2(frames))
        feature_maps = F.relu(self.conv3(frames))

        # self-attention
        theta1_x = self.theta1(feature_maps).view(-1, 32, 49).permute(0, 2, 1)
        phi_x = self.phi(feature_maps).view(-1, 32, 49)
        # -> 49*49 similarity matrix
        similarity_matrix = torch.matmul(theta1_x, phi_x)
        self_attention_scores = F.softmax(similarity_matrix, dim=2)
        # -> b, 32, 7, 7
        g_x = self.g(feature_maps).view(-1, 32, 49).permute(0, 2, 1)
        # -> b, 49, 32
        sa_feature_maps = torch.matmul(self_attention_scores, g_x)
        # -> b, 64, 7, 7
        sa_feature_maps = self.p(sa_feature_maps.permute(0, 2, 1).contiguous().view(-1, 32, 7, 7))

        # goal-oriented navigation
        # -> b, 32, 7, 7
        theta2_x = self.theta2(sa_feature_maps)
        theta2_x = theta2_x.view(-1, 32, 49)
        # -> b, 1, 32
        alpha_g = self.alpha(goals.view(-1, 8)).unsqueeze(1)
        # -> b, 49, 1, batched matrix multiplication
        attention_scores = torch.matmul(alpha_g, theta2_x).view(-1, 49, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights.data

        # compute aggregated feature
        sa_feature_maps = sa_feature_maps.view(-1, 64, 49).permute(0, 2, 1)
        agg_features = torch.sum(torch.mul(sa_feature_maps, attention_weights), dim=1)
        sa_feature_maps = F.relu(self.fc4(agg_features))
        features = torch.cat([sa_feature_maps, goals.view(goals.size(0), -1)], dim=1)
        return self.fc5(features)