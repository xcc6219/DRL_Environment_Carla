import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
from tensorboardX import SummaryWriter


class Replay_buffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        n_batches = len(self.storage) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            s, a, r, ns, d = [], [], [], [], []
            Sstates, Sbirdseye, Ssemantics, NSstates, NSbirdseye, NSsemantics = [], [], [], [], [], []
            for i in range(start_idx, end_idx):
                S, A, R, NS, D = self.storage[i]
                Sstates.append(S['state'])
                Sbirdseye.append(S['birdseye'])
                Ssemantics.append(S['semantic'])
                a.append(np.array(A, copy=False))
                r.append(np.array(R, copy=False))
                NSstates.append(NS['state'])
                NSbirdseye.append(NS['birdseye'])
                NSsemantics.append(NS['semantic'])
                d.append(np.array(D, copy=False))
            Sstates = torch.stack(Sstates)
            Sbirdseye = torch.stack(Sbirdseye)
            Ssemantics = torch.stack(Ssemantics)
            NSstates = torch.stack(NSstates)
            NSbirdseye = torch.stack(NSbirdseye)
            NSsemantics = torch.stack(NSsemantics)

            # 将批量化后的数据重新组织成字典
            s = {'state': Sstates, 'birdseye': Sbirdseye, 'semantic': Ssemantics}
            ns = {'state': NSstates, 'birdseye': NSbirdseye, 'semantic': NSsemantics}

            yield s, np.array(a), np.array(r).reshape(-1, 1), ns, np.array(d).reshape(-1, 1)


def dicts_to_batched_dict(dict_list):
    batched_dict = {}
    if not dict_list:
        return batched_dict
    for key in dict_list[0].keys():
        batched_dict[key] = []
    for d in dict_list:
        for key, value in d.items():
            batched_dict[key].append(value)
    return batched_dict


def create_directory(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        exp_number = 1
    else:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        exp_numbers = [int(d.replace('exp', '')) for d in subdirs if d.startswith('exp') and d[3:].isdigit()]
        exp_number = max(exp_numbers) + 1 if exp_numbers else 1
    new_exp_dir = os.path.join(base_dir, f'exp{exp_number}')
    os.makedirs(new_exp_dir, exist_ok=True)
    return new_exp_dir


def create_conv_sequence(in_channels, out_features):
    return nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(32),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),
                         nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(32),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),
                         nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(32),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),
                         nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(32),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),
                         nn.Flatten(),  # 将多维的输出展平
                         nn.Linear(32 * 15 * 15, out_features))


class ConvFeatureExtractor(nn.Module):
    def __init__(self, img_in_channels, img_out_features, state_in, state_out):
        super(ConvFeatureExtractor, self).__init__()
        self.conv_bev = create_conv_sequence(img_in_channels, img_out_features)
        self.conv_seg = create_conv_sequence(img_in_channels, img_out_features)
        self.state_fc = nn.Sequential(nn.Linear(state_in, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, state_out))

    def forward(self, state):
        x = state['state']
        y = state['birdseye']
        z = state['semantic']
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(y.shape) == 3:
            y = y.unsqueeze(0)
        if len(z.shape) == 3:
            z = z.unsqueeze(0)
        x = self.state_fc(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        y = self.conv_bev(y)
        z = self.conv_seg(z)
        features = torch.cat((x, y, z), -1)
        return features


class Actor(nn.Module):
    def __init__(self, input_features, min_log_std=-20, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_features, 648)
        self.fc2 = nn.Linear(648, 256)
        self.fc3 = nn.Linear(256, 64)
        self.mu_head = nn.Linear(64, 2)  # 注意：这里应匹配动作维度
        self.log_std_head = nn.Linear(64, 2)  # 注意：这里也应匹配动作维度
        self.max_action = 1.0

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, features):
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        features = F.relu(self.fc3(features))
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        mu = torch.tanh(mu) * self.max_action
        return mu, std


class Critic(nn.Module):
    def __init__(self, input_features):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_features, 648)
        self.fc2 = nn.Linear(648, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, features):
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        features = self.fc3(features)
        return features


class Q(nn.Module):
    def __init__(self, input_features):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(input_features + 2, 648)
        self.fc2 = nn.Linear(648, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        a = a.reshape(-1, 2)  # 动作维度
        features = torch.cat((s, a), -1)  # combination s and a
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        features = self.fc3(features)
        return features


class SAC():
    def __init__(self, params):
        super(SAC, self).__init__()
        self.opt = params
        self.feature_extractor = ConvFeatureExtractor(img_in_channels=3, img_out_features=128, state_in=257,
                                                      state_out=392).to(self.opt.device)
        self.policy_net = Actor(input_features=128 * 2 + 392).to(self.opt.device)
        self.value_net = Critic(input_features=128 * 2 + 392).to(self.opt.device)
        self.Q_net = Q(input_features=128 * 2 + 392).to(self.opt.device)
        self.Target_value_net = Critic(input_features=128 * 2 + 392).to(self.opt.device)
        self.replay_buffer = Replay_buffer(self.opt.capacity)
        self.policy_optimizer = optim.Adam(list(self.feature_extractor.parameters()) +
                                           list(self.policy_net.parameters()), lr=self.opt.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.opt.learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=self.opt.learning_rate)
        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=self.opt.learning_rate)
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        base_dir = './exp-SAC'
        self.writer = SummaryWriter(create_directory(base_dir))

        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.save_dir = create_directory(base_dir='./SAC_model')

    def select_action(self, state):
        state = self.feature_extractor(state)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action

    def get_action_log_prob(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        min_Val = torch.tensor(1e-7).float()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + min_Val)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):
        for _ in range(self.opt.gradient_steps):
            if self.num_training % 500 == 0:
                print("Training ... {} ".format(self.num_training))
            for bn_s, bn_a, bn_r, bn_s_, bn_d in self.replay_buffer.sample(self.opt.batch_size):
                # bn_s = torch.FloatTensor(bn_s).to(self.opt.device)
                bn_a = torch.FloatTensor(bn_a).to(self.opt.device)
                bn_r = torch.FloatTensor(bn_r).to(self.opt.device)
                # bn_s_ = torch.FloatTensor(bn_s_).to(self.opt.device)
                bn_d = torch.FloatTensor(1 - bn_d).to(self.opt.device)

                f_bn_s_ = self.feature_extractor(bn_s_)
                f_bn_s = self.feature_extractor(bn_s)

                target_value = self.Target_value_net(f_bn_s_)
                next_q_value = bn_r + (1 - bn_d) * self.opt.gamma * target_value
                excepted_value = self.value_net(f_bn_s)
                excepted_Q = self.Q_net(f_bn_s, bn_a)
                sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(f_bn_s)
                excepted_new_Q = self.Q_net(f_bn_s, sample_action)
                log_prob_average = log_prob.mean(dim=-1, keepdim=True)
                next_value = excepted_new_Q - log_prob_average

                # !!!Note that the actions are sampled according to the current policy,
                # instead of replay buffer. (From original paper)

                V_loss = self.value_criterion(excepted_value, next_value.detach())  # J_V
                V_loss = V_loss.mean()

                # Single Q_net this is different from original paper!!!
                Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach())  # J_Q
                Q_loss = Q_loss.mean()

                log_policy_target = excepted_new_Q - excepted_value

                pi_loss = log_prob_average * (log_prob_average - log_policy_target).detach()
                pi_loss = pi_loss.mean()

                feature_loss = log_prob_average * (log_prob_average - log_policy_target).detach()
                feature_loss = feature_loss.mean()

                self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
                self.writer.add_scalar('Loss/Q_loss', Q_loss, global_step=self.num_training)
                self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)
                self.writer.add_scalar('Loss/feature_loss', feature_loss, global_step=self.num_training)
                # mini batch gradient descent

                self.value_optimizer.zero_grad()
                V_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                self.Q_optimizer.zero_grad()
                Q_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
                self.Q_optimizer.step()
                self.policy_optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(list(self.feature_extractor.parameters()) +
                                         list(self.policy_net.parameters()), 0.5)
                self.policy_optimizer.step()
                # soft update
                for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(target_param.data * (1 - self.opt.tau) + param.data * self.opt.tau)

                self.num_training += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), os.path.join(self.save_dir, 'policy_net.pth'))
        torch.save(self.value_net.state_dict(), os.path.join(self.save_dir, 'value_net.pth'))
        torch.save(self.Q_net.state_dict(), os.path.join(self.save_dir, 'Q_net.pth'))

    def load(self):
        exp_dir = self.opt.load_dir
        self.policy_net.load_state_dict(torch.load(os.path.join(exp_dir, 'policy_net.pth')))
        self.value_net.load_state_dict(torch.load(os.path.join(exp_dir, 'value_net.pth')))
        self.Q_net.load_state_dict(torch.load(os.path.join(exp_dir, 'Q_net.pth')))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")
