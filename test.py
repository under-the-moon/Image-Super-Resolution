import torch.nn as nn

def lr_func(epoch):
    lr = max(1e-3 * (0.5 ** (epoch // 10)), 1e-6)
    return lr


for i in range(100):
    print(lr_func(epoch=i))



class Upsample(nn.Module):
    def __init__(self, input_resolution, dim=96):
        super(Upsample, self).__init__()
        self.scale = 2
        self.dim = dim

        # self.num_features = input_resolution[0] * input_resolution[1]
        #
        # self.expansion_wh = nn.Linear(self.num_features, self.num_features * 4)
        # self.expansion_dim = nn.Linear(self.dim, self.dim * 4)
        #
        # self.reduction = nn.Sequential(
        #     Conv2d(self.dim * 2, self.dim * 2, 3, 1, 1),
        #     Conv2d(self.dim * 2, self.dim, 3, 1, 1)
        # )
        # self.reduction_atten = Conv2d(self.dim, self.dim, 3, 1, 1)

        # self.combine = nn.Sequential(
        #     Conv2d(self.dim, self.dim, 3, 1, 1),
        #     Conv2d(self.dim, self.dim, 3, 1, 1)
        # )

    # def upsample_wh(self, x):
    #     B, C, H, W = x.shape
    #
    #     x = x.view(B, C, -1)
    #     x = self.expansion_wh(x)
    #     x = x.view(B, C, self.num_features, 4)
    #     x = x.view(B, C, H, W, 2, 2)
    #     x = x.permute(0, 1, 2, 4, 3, 5)
    #     x = x.reshape(B, C, H * 2, W * 2)
    #     return x
    #
    # def upsample_dim(self, x):
    #     B, C, H, W, = x.shape
    #     x = x.permute(0, 2, 3, 1)
    #     x = self.expansion_dim(x)
    #     x = x.view(B, H, W, C, 4)
    #     x = x.permute(0, 3, 1, 2, 4)
    #     x = x.view(B, C, H, W, 2, 2)
    #     x = x.permute(0, 1, 2, 4, 3, 5)
    #     x = x.reshape(B, C, H * 2, W * 2)
    #     return x

    def forward(self, x):
        # x_wh = self.upsample_wh(x)
        # x_dim = self.upsample_dim(x)
        # context = torch.concat([x_wh, x_dim], dim=1)
        # context = self.reduction(context)
        # scores = torch.sigmoid(self.reduction_atten(context))
        # x = scores * x_wh + (1 - scores) * x_dim
        return