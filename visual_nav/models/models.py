from visual_nav.models.model_archive import *


class GDNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=18, with_sa=True, with_ga=True, goal_embedding_as_feature=False,
                 share_image_embedding=False, mean_pool_feature_map=False, residual_connection=False,
                 attention_as_regressor=False, sa_input_to_ga=False, regression=False):
        """
        A base network architecture for goal-driven tasks
        """
        super(GDNet, self).__init__()
        self.with_sa = with_sa
        self.with_ga = with_ga
        self.goal_embedding_as_feature = goal_embedding_as_feature
        self.share_image_embedding = share_image_embedding
        self.mean_pool_feature_map = mean_pool_feature_map
        self.residual_connection = residual_connection
        self.attention_as_regressor = attention_as_regressor
        self.sa_input_to_ga = sa_input_to_ga

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv_block = None

        # input feature channel size
        self.C = 64
        self.H = 7
        self.W = 7
        self.E = 32
        # input goal dim
        self.D = 2

        if with_sa:
            if not with_ga:
                raise ValueError('Self-attention is only applicable together with goal-driven attention')
            # self-attention
            self.theta = nn.Conv2d(self.C, self.E, 1)
            self.phi = nn.Conv2d(self.C, self.E, 1)
            self.g = nn.Conv2d(self.C, self.E, 1)
            self.recover_dim = nn.Conv2d(self.E, self.C, 1)

        if with_ga:
            # goal-driven attention
            if attention_as_regressor:
                self.attention = nn.Sequential(nn.Linear(self.C + self.D, 128), nn.ReLU(), nn.Linear(128, 1))
            else:
                if not self.share_image_embedding:
                    self.theta2 = nn.Conv2d(self.C, self.E, 1)
                self.alpha = nn.Linear(self.D, self.E)
            image_feature_dim = self.C * 1 * 1
        else:
            if mean_pool_feature_map:
                image_feature_dim = self.C * 1 * 1
            else:
                image_feature_dim = self.C * self.H * self.W

        if goal_embedding_as_feature:
            goal_feature_dim = self.E
        else:
            goal_feature_dim = self.D

        # action classification layers
        self.fc4 = nn.Linear(image_feature_dim + goal_feature_dim, 256)
        self.fc5 = nn.Linear(256, 520)
        if regression:
            self.fc6 = nn.Linear(520, 2)
        else:
            self.fc6 = nn.Linear(520, num_actions)
        self.fc_block = None

        # for visualization
        self.attention_weights = None
        self.max_response_index = None
        self.mean_block_response = None
        self.guided_backprop_initialized = False
        self.gradients = None

    def forward(self, frames, goals):
        B = goals.size(0)

        if self.guided_backprop_initialized:
            feature_maps = self.conv_block(frames)
        else:
            # compute feature map -> B, C, H, W
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
            if self.residual_connection:
                sa_feature_maps = sa_feature_maps + feature_maps
            last_feature_maps = sa_feature_maps
        else:
            last_feature_maps = feature_maps

        if self.with_ga:
            if self.sa_input_to_ga:
                input_feature_maps = last_feature_maps
            else:
                input_feature_maps = feature_maps
            if self.attention_as_regressor:
                # (b, 64, 7, 7) -> (b, 49, 64)
                image_blocks = input_feature_maps.view(input_feature_maps.size(0), input_feature_maps.size(1), -1).\
                    permute(0, 2, 1)
                # (b * 49, 64)
                image_blocks = image_blocks.contiguous().view(-1, image_blocks.size(2))
                goals_expanded = goals.view(B, -1).unsqueeze(1).expand((B, 49, self.D))
                queries = goals_expanded.contiguous().view(-1, self.D)
                attention_input = torch.cat([image_blocks, queries], dim=1)
                attention_scores = self.attention(attention_input).view(B, 49, 1)
                attention_weights = F.softmax(attention_scores, dim=1)
                self.attention_weights = attention_weights.data
            else:
                # -> b, 1, 32
                alpha_g = self.alpha(goals.view(-1, self.D)).unsqueeze(2)
                # -> b, 49, 1, batched matrix multiplication
                if self.share_image_embedding:
                    attention_scores = torch.matmul(theta_x, alpha_g)
                else:
                    theta2_x = self.theta2(input_feature_maps).view(-1, 32, 49).permute(0, 2, 1)
                    attention_scores = torch.matmul(theta2_x, alpha_g)
                attention_weights = F.softmax(attention_scores, dim=1)
                self.attention_weights = attention_weights.data

            # compute aggregated feature
            # -> b, 49, 64
            last_feature_maps = last_feature_maps.view(B, 64, 49).permute(0, 2, 1)
            image_features = torch.sum(torch.mul(last_feature_maps, attention_weights), dim=1)

            if self.goal_embedding_as_feature:
                goal_features = alpha_g.squeeze(2)
            else:
                goal_features = goals.view(goals.size(0), -1)
        else:
            self.mean_block_response = F.softmax(torch.mean(feature_maps.view(B, self.C, -1), 1), 1).data
            _, self.max_response_index = torch.max(self.mean_block_response, 1)
            if self.mean_pool_feature_map:
                # B, C, H, W -> B, C
                image_features = torch.mean(feature_maps.view(B, self.C, -1), 2)
            else:
                image_features = feature_maps.view(B, -1)
            goal_features = goals.view(goals.size(0), -1)

        # action classification
        fc_inputs = torch.cat([image_features, goal_features], dim=1)
        if self.guided_backprop_initialized:
            outputs = self.fc_block(fc_inputs)
        else:
            outputs = F.relu(self.fc4(fc_inputs))
            outputs = F.relu(self.fc5(outputs))
            outputs = self.fc6(outputs)
        return outputs

    def init_guided_backprop(self):
        self.conv_block = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3)
        self.fc_block = nn.Sequential(self.fc4, nn.ReLU(), self.fc5, nn.ReLU(), self.fc6)

        self.guided_backprop_initialized = True
        self.gradients = None
        self.eval()

        # update relus
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.conv_block._modules.items():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_hook_function)
        for pos, module in self.fc_block._modules.items():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_hook_function)

        # Register hook to the first layer
        # hook layers
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = list(self.conv_block._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, frames, goals, target_class):
        frames_input = torch.autograd.Variable(frames.cuda(), requires_grad=True)

        model_output = self.forward(frames_input, goals)
        # Zero gradients
        self.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(frames.device)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        return frames_input.grad.data[0, 0, :, :]


class PlainCNN(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=False,
                         with_ga=False,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=False
                         )


class PlainCNNMean(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=False,
                         with_ga=False,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=True
                         )


class GDANoGEF(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=False,
                         with_ga=True,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=False
                         )


class GDA(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=False,
                         with_ga=True,
                         goal_embedding_as_feature=True,
                         share_image_embedding=False,
                         mean_pool_feature_map=False
                         )


class GDARegressor(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=False,
                         with_ga=True,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=False,
                         attention_as_regressor=True
                         )


class GDDANoSIE(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=True,
                         share_image_embedding=False,
                         mean_pool_feature_map=False
                         )


class GDDANoGEF(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=False,
                         share_image_embedding=True,
                         mean_pool_feature_map=False
                         )


class GDDA(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=True,
                         share_image_embedding=True,
                         mean_pool_feature_map=False
                         )


class GDDAResidual(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=True,
                         share_image_embedding=True,
                         mean_pool_feature_map=False,
                         residual_connection=True
                         )


class GDDAResidualNoSIENoGEF(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=False,
                         residual_connection=True
                         )


class GDDAResidualRegressor(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=False,
                         residual_connection=True,
                         attention_as_regressor=True
                         )


class GDDASai(GDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_sa=True,
                         with_ga=True,
                         goal_embedding_as_feature=False,
                         share_image_embedding=False,
                         mean_pool_feature_map=False,
                         residual_connection=True,
                         attention_as_regressor=True,
                         sa_input_to_ga=True
                         )


model_factory = {'dqn': DQN, 'daqn': DAQN, 'da1qn': DA1QN, 'da2qn': DA2QN,
                 'gdda': GDDA, 'gda': GDA, 'plain_cnn': PlainCNN,
                 'plain_cnn_mean': PlainCNNMean, 'gda_no_gef': GDANoGEF, 'gdda_no_sie': GDDANoSIE,
                 'gdda_residual': GDDAResidual, 'gdda_no_gef': GDDANoGEF, 'gda_regressor': GDARegressor,
                 'gdda_residual_no_sie_no_gef': GDDAResidualNoSIENoGEF, 'gdda_residual_regressor': GDDAResidualRegressor,
                 'gdda_sai': GDDASai}

