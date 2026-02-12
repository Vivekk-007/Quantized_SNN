import torch
import torch.nn as nn
from models.lif_neuron import FastSigmoidSpike
from models.quantization import QuantizedConv2d, QuantizedLinear


class SpikingCNN(nn.Module):
    def __init__(self, T=10, slope=10):
        super().__init__()
        self.T = T
        self.slope = slope
        self.threshold = 0.5

        self.conv1 = QuantizedConv2d(1, 32, 3, padding=1)
        self.conv2 = QuantizedConv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = QuantizedLinear(64 * 7 * 7, 256)
        self.fc2 = QuantizedLinear(256, 10)

    def forward(self, x, return_spike_stats=False):

        batch_size = x.size(0)

        mem1 = torch.zeros(batch_size, 32, 28, 28, device=x.device)
        mem2 = torch.zeros(batch_size, 64, 7, 7, device=x.device)
        mem3 = torch.zeros(batch_size, 256, device=x.device)
        output_sum = torch.zeros(batch_size, 10, device=x.device)

        # Spike counters
        total_spikes = 0

        for _ in range(self.T):

            # Conv1
            cur1 = self.conv1(x)
            mem1 += cur1
            spike1 = FastSigmoidSpike.apply(mem1 - self.threshold, self.slope)
            mem1 *= (1 - spike1)

            total_spikes += spike1.sum().item()

            # Conv2
            cur2 = self.conv2(spike1)
            cur2 = self.pool(cur2)
            cur2 = self.pool(cur2)

            mem2 += cur2
            spike2 = FastSigmoidSpike.apply(mem2 - self.threshold, self.slope)
            mem2 *= (1 - spike2)

            total_spikes += spike2.sum().item()

            # FC
            flat = spike2.view(batch_size, -1)
            flat = self.dropout(flat)

            mem3 += self.fc1(flat)
            spike3 = FastSigmoidSpike.apply(mem3 - self.threshold, self.slope)
            mem3 *= (1 - spike3)

            total_spikes += spike3.sum().item()

            output_sum += self.fc2(spike3)

        if return_spike_stats:
            return output_sum / self.T, total_spikes
        else:
            return output_sum / self.T
