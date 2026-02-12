import torch


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slope):
        ctx.save_for_backward(input)
        ctx.slope = slope
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        slope = ctx.slope
        return grad_output * (slope / (1 + slope * input.abs())**2), None


