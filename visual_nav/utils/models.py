from visual_nav.utils.model_archive import *


class GDDA(nn.Module):
    def __init__(self, in_channels=4, num_actions=18, with_sa=True, with_ga=True, share_embedding=True):
        """
        DQN with goal-dependent attention
        """
        super(GDDA, self).__init__()
        self.with_sa = with_sa
        self.with_ga = with_ga
        self.share_embedding = share_embedding
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        if with_sa:
            if not with_ga:
                raise ValueError('Self-attention is only applicable together with goal-driven attention')
            # self-attention
            self.theta = nn.Conv2d(64, 32, 1)
            self.phi = nn.Conv2d(64, 32, 1)
            self.g = nn.Conv2d(64, 32, 1)
            self.recover_dim = nn.Conv2d(32, 64, 1)

        if with_ga:
            # goal-driven attention
            if not share_embedding:
                self.theta2 = nn.Conv2d(64, 32, 1)
            self.alpha = nn.Linear(8, 32)
            conv_feature_dim = 64 * 1 * 1
        else:
            # TODO: compare with mean pooling layer
            conv_feature_dim = 64 * 7 * 7

        # action classification layers
        self.fc4 = nn.Linear(conv_feature_dim, 512)
        self.fc5 = nn.Linear(520, num_actions)

        # for visualization
        self.attention_weights = None

    def forward(self, frames, goals):
        batch_size = goals.size(0)

        # compute feature map
        frames = F.relu(self.conv1(frames))
        frames = F.relu(self.conv2(frames))
        feature_maps = F.relu(self.conv3(frames))

        if self.with_sa:
            theta_x = self.theta(feature_maps).view(-1, 32, 49).permute(0, 2, 1)
            phi_x = self.phi(feature_maps).view(-1, 32, 49)
            # -> 49*49 similarity matrix
            similarity_matrix = torch.matmul(theta_x, phi_x)
            self_attention_scores = F.softmax(similarity_matrix, dim=2)
            # -> b, 32, 7, 7
            g_x = self.g(feature_maps).view(-1, 32, 49).permute(0, 2, 1)
            # -> b, 49, 32
            sa_feature_maps = torch.matmul(self_attention_scores, g_x)
            # -> b, 64, 7, 7
            sa_feature_maps = self.recover_dim(sa_feature_maps.permute(0, 2, 1).contiguous().view(-1, 32, 7, 7))
            last_feature_maps = sa_feature_maps
        else:
            last_feature_maps = feature_maps

        if self.with_ga:
            # -> b, 1, 32
            alpha_g = self.alpha(goals.view(-1, 8)).unsqueeze(2)
            # -> b, 49, 1, batched matrix multiplication
            if self.share_embedding:
                attention_scores = torch.matmul(theta_x, alpha_g)
            else:
                theta2_x = self.theta2(feature_maps).view(-1, 32, 49).permute(0, 2, 1)
                attention_scores = torch.matmul(theta2_x, alpha_g)
            attention_weights = F.softmax(attention_scores, dim=1)
            self.attention_weights = attention_weights.data

            # compute aggregated feature
            # -> b, 49, 64
            last_feature_maps = last_feature_maps.view(batch_size, 64, 49).permute(0, 2, 1)
            conv_features = torch.sum(torch.mul(last_feature_maps, attention_weights), dim=1)
        else:
            conv_features = feature_maps.view(frames.size(0), -1)

        # action classification
        fc4_outputs = F.relu(self.fc4(conv_features))
        fc5_outputs = self.fc5(torch.cat([fc4_outputs, goals.view(goals.size(0), -1)], dim=1))
        return fc5_outputs


class GDA(GDDA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, with_sa=False, with_ga=True)


class PlainCNN(GDDA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, with_sa=False, with_ga=False)


model_factory = {'dqn': DQN, 'daqn': DAQN, 'da1qn': DA1QN, 'da2qn': DA2QN,
                 'gdda': GDDA, 'gda': GDA, 'plain_cnn': PlainCNN}
