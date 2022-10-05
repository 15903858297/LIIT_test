import torch.nn as nn

from models import register

@register('mlp')
class MLP(nn.Module):
    """
        全连接网络
        网络的每个全连接层由nn.Linear()函数和nn.ReLU()函数构成
        nn.ReLU()表示使用激活函数ReLU
    """
    # in_dim：隐藏层的输入，数据的特征数
    # out_dim 隐藏层的输出，表示神经元的数量
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        #定义一个名为layers的空列表
        layers = []
        # 隐藏层的输入，数据的特征数
        lastv = in_dim
        # 使用了全连接神经网络分类器
        # 通过下面的for循环遍历hidden_list列表，可以得到一个由全连接层、激活函数等组成的一个列表layers
        for hidden in hidden_list:
            #nn.Linear用于创建一个多输入、多输出的全连接层
            # 第一个参数in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
            # in_features的数量，决定参数的个数  Y = WX + b, X的维度就是in_features
            # out_features的数量，决定了全连接层中神经元的个数，因为每个神经元只有一个输出。
            # 所以多少个输出，就需要多少个神经元
            layers.append(nn.Linear(lastv, hidden))
            #追加激活函数
            layers.append(nn.ReLU())
            lastv = hidden
            #追加一个全连接层，参数为数据的特征数和神经元的数量
        layers.append(nn.Linear(lastv, out_dim))
        #  通过nn.Sequential函数将列表通过非关键字参数的形式传入
        self.layers = nn.Sequential(*layers)

    # 定义网络的向前传播路径
    def forward(self, x):
        # 取倒数第一个元素
        shape = x.shape[:-1]
        #view()函数就是用来改变tensor的形状的，view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变。
        #shape[-1]代表最后一个维度,如在二维张量里,shape[-1]表示列数
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
