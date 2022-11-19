import torch


def pnsr(output, target):
    output = output.float()
    target = target.float()
    output = output * 255
    target = target * 255
    output = torch.clamp(output, 0, 255)
    target = torch.clamp(target, 0, 255)
    square_max = torch.square(torch.max(target))
    mse = torch.mean(torch.square(target - output))
    return 10 * (torch.log10(square_max) - torch.log10(mse))
