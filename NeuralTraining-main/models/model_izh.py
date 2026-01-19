import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 2
lens = 0.4
decay = 0.2
device = torch.device("cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class ActFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

cfg_fc = [16, 16, 16, 2]

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class IzhikevichNeuron(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(IzhikevichNeuron, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes)
        self.apply(weights_init_)

        self.oup = out_planes

        # Izhikevich parameters
        self.a = nn.Parameter(0.02 * torch.ones(out_planes), requires_grad=False)  # Recovery rate
        self.b = nn.Parameter(0.2 * torch.ones(out_planes), requires_grad=False)   # Sensitivity
        self.c = nn.Parameter(-65 * torch.ones(out_planes), requires_grad=False)  # Reset potential
        self.d = nn.Parameter(8 * torch.ones(out_planes), requires_grad=False)    # Recovery increment

        self.dt = 1.0  # Time step

    def zeros_state(self, size):
        v = -65 * torch.ones(*size).to(device)  # Membrane potential
        u = self.b * v                         # Recovery variable
        return v, u

    def update_neuron(self, inputs, states):
        v, u = states

        # Update dynamics
        I = inputs  # Input current
        v = v + self.dt * (0.04 * v ** 2 + 5 * v + 140 - u + I)
        u = u + self.dt * self.a * (self.b * v - u)

        # Reset condition
        reset = v >= 30
        v = torch.where(reset, self.c, v)
        u = torch.where(reset, u + self.d, u)

        spike_out = act_fun(v - thresh)
        new_state = (v, u)
        return new_state, spike_out

    def forward(self, input, wins=15):
        batch_size = input.size(0)
        state1 = self.zeros_state([batch_size, self.oup])
        spikes = torch.zeros([batch_size, wins, self.oup]).to(device)

        for step in range(wins):
            state1, spike_out = self.update_neuron(self.fc(input[:, step, :]), state1)
            spikes[:, step, :] = spike_out
        return spikes


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Q1 architecture
        self.fc1 = IzhikevichNeuron(num_inputs + num_actions, hidden_dim)    
        self.fc_output1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc2 = IzhikevichNeuron(num_inputs + num_actions, hidden_dim)    
        self.fc_output2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action, wins=15):
        input = torch.cat([state, action], 2)

        output1 = self.fc1(input, wins)
        out1 = self.fc_output1(torch.mean(output1, dim=1))
        
        output2 = self.fc2(input, wins)
        out2 = self.fc_output2(torch.mean(output2, dim=1))

        return out1, out2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.hh_layer = IzhikevichNeuron(num_inputs, hidden_dim)
        self.linear1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        input_tmp = []
        for i in range(5):
            input_tmp += [state[:, i, ...]] * 3
        state = torch.stack(input_tmp, dim=1) 
        x = self.hh_layer(state)
        x = torch.mean(x, dim=1)
        x1 = self.linear1_1(x)
        x1 = nn.ReLU()(x1)
        x1 = self.linear1_2(x1)
        x1 = nn.ReLU()(x1)
        mean = self.mean_linear(x1)
        x2 = self.linear2_1(x)
        x2 = nn.ReLU()(x2)
        x2 = self.linear2_2(x2)
        x2 = nn.ReLU()(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
