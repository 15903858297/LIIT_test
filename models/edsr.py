# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

# 构建子模块，是在__int__()中实现的。
# 拼接子模块，是在froward()中实现的。



# EDSR模型，全称为enhanced deep super-resolution network
# （增强的深度学习超分辨率重建网络）

#Conv层（卷积层）
# in_channels参数代表输入特征矩阵的深度即channel，比如输入一张RGB彩色图像，那in_channels = 3
# out_channels参数代表卷积核的个数，使用n个卷积核输出的特征矩阵深度即channel就是n
# kernel_size参数代表卷积核的尺寸，输入可以是int类型如3 代表卷积核的height = width = 3，也可以是tuple类型如(3, 5)代表卷积核的height = 3，width = 5
# bias = True 是否要添加偏置参数作为可学习参数的一个，默认为True
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
# padding参数代表在输入特征矩阵四周补零的情况默认为0，同样输入可以为int型如1 代表上下方向各补一行0元素，左右方向各补一列0像素（即补一圈0），
# 如果输入为tuple型如(2, 1) 代表在上方补两行下方补两行，左边补一列，右边补一列。可见下图，padding[0]是在H高度方向两侧填充的，padding[1]是在W宽度方向两侧填充的；
        padding=(kernel_size//2), bias=bias)
#//是向下取整的意思 a//b,应该是对除以b的结果向负无穷方向取整后的数 举例: 5//2=2

#MeanShift均值漂移算法
# 以二维来说明：图中有很多红点，红点就是样本特征点，meanshift就是在这些点中的任意一个点为圆心，然后以半径R画一个圆，然后落在这个圆中的所有点和圆心都会对应的一个向量，
# 把所有这些向量相加（注意是向量相加），最终我们只得到一个向量，就是下图中用黄色箭头表示的向量，这个向量就是meanshift向量
# 然后再以这个meanshift向量的终点为圆心，继续上述过程，又可以得到一个meanshift向量。然后不断地继续这样的过程
# 可以得到很多连续的meanshift向量，这些向量首尾相连，最终得到会在一个地方停下来（即我们说的meanshift算法会收敛），最后的那个meanshift向量的终点就是最终得到的结果
#meanshift算法的过程最终的效果就是：从起点开始，最终会一步一步到达样本特征点最密集的点那里
#nn.Conv2d是二维卷积方法
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        #使用Pytorch进行预处理时,mean和std分别表示图像集每个通道的均值和方差
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        #调用父类的构造函数
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        #返回rgb_std对用的张量
        std = torch.Tensor(rgb_std)
        #初始化weight和bias
        # torch.nn.Conv2d.weight.data
        #torch.eye(3)是为了生成对角线全1，其余部分全0的三维数组  view（）重新定义矩阵的形状
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        #torch.nn.Conv2d.bias.data
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        # parameters()存储模型训练中所有的参数信息
        for p in self.parameters():
            #requires_grad = False屏蔽预训练模型的权重，只训练最后一层的全连接的权重
            p.requires_grad = False

#ResBlock（残差块）
# 残差思想：去掉与主体内容相同的部分，从而突出微小的变化。只学习扰动，这样可以用来训练特别深的网络，在反向求梯度拟合的过程就更容易算
# 残差网络结构的思想特别适合用来解决超分辨率问题，因为：
# 低分辨率图像和输出的高分辨率图像在很大程度上是相似的，也就是指低分辨率图像携带的低频信息与高分辨率图像的低频信息相近，训练时带上这部分会多花费大量的时间，
# 实际上只需要学习高分辨率图像和低分辨率图像之间的高频部分残差即可。所以残差网络结构的思想特别适合用来解决超分辨率问题
# 所有的网络层都是继承于nn.Module这个类的
class ResBlock(nn.Module):
    def __init__(
        #kernel_size代表卷积核的尺寸
        self, conv, n_feats, kernel_size,
        #使用relu激活
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        #调用父类的构造函数
        super(ResBlock, self).__init__()
        m = []#定义一个空列表
        #i的范围在0到1
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
                m.append(nn.BatchNorm2d(n_feats))#
            if i == 0:
                m.append(act)#m.append（）是往里面填加一个卷积层
        #一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。
        #*作用在实参上，是将输入迭代器拆成一个个元素。从nn.Sequential的定义来看，遇到list，必须用*号进行转化
        self.body = nn.Sequential(*m)#self.body网络型模型的实体
        # nn.Sequential 把它线性化了
        self.res_scale = res_scale

    #forward（）表示在建立模型后，进行神经元网络的前向传播。也就是说forward就是专门用来计算给定输入，得到神经元网络输出的方法。
    def forward(self, x):
        #mul（）逐个对 input 和 other 中对应的元素相乘，nput 和 other 均可以是张量或者数字。
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
#UpSample（上采样，用于扩大图像像素）
# Upsample可用的算法是最近邻和线性，双线性，双三次和三线性插值算法。
# 可以给出scale_factor或目标输出大小来计算输出大小（不能同时给出两者）
# 此模块是真正令图像扩大分辨率的模块，其功能是将输入图片按到一定规则rescale到一个想要的尺寸
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
    #UpSample的实现过程不是直接通过插值等方式产生这个高分辨率图像，而是通过卷积先得到 r2个通道的特征图，也就是说——通过卷积将通道数扩大一倍；
    # 然后再用PixelShuffle，将两个通道的特征图相互插入使得尺寸扩大一倍，来使得图像最终的超像素重建。
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            #log(x，base)参数：x - 数值表达式   base - 可选，底数，默认为e
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                #nn.PixleShuffle(upscale_factor) upscale_factor就是放大的倍数，数据类型为int 放大倍数为2倍
                m.append(nn.PixelShuffle(2))

                if bn:
                    #对n_feats进行数据的归一化处理
                    # class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
                    # num_features： 来自期望输入的特征数，该期望输入的大小为'batch_size x num_features x height x width'
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':#激活函数
                    m.append(nn.ReLU(True))
                    #PReLU 也是 ReLU 的改进版本
                    # 在负值域，PReLU的斜率较小，这也可以避免Dead ReLU问题。
                    # PReLU 在负值域是线性运算。尽管斜率很小，但不会趋于0
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            #放大倍数为3倍
            m.append(nn.PixelShuffle(3))
            if bn:
                #对数据进行归一化处理
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        #调用父类的构造函数
        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}
# 利用PyTorch定义深度网络层
# 自定义EDSR类，该类继承自nn.Module类，实现两个基本的函数：
# 构造函数__init__()、层的逻辑运算函数forward()
class EDSR(nn.Module):
    #在构造函数__init__()中实现层的参数定义
    # 构建子模块，是在__int__()中实现的。
    def __init__(self, args, conv=default_conv):
        # 函数的第一个参数是实例对象本身 self
        #继承父类构造函数中的内容，并且子类需要有所补充的时候使用。
        super(EDSR, self).__init__()
        self.args = args#把属性绑定到self 也就是实例本身上
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        #kernel_size代表卷积核的尺寸为3
        kernel_size = 3
        #[0]为 scale中第1个元素
        scale = args.scale[0]
        # inplace = True,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
        act = nn.ReLU(True)
        #字符串前加r去掉反斜杠的转移机制 例：r"\n\n”　表示一个普通字符串 \n\n，而不表示换行了
        # 以 f开头表示在字符串内支持大括号内的python 表达式
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        #如果url_name在url中
        if url_name in url:
            self.url = url[url_name]
        else:
            # 否则的话url的值为空
            self.url = None
            # 将参数的rgb_range得值传递到均值漂移算法中
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        # nn.Sequential() 可以允许将整个容器视为单个模块（即相当于把多个模块封装成一个模块），
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        # 如果args.no_upsampling的值为True
        if args.no_upsampling:
            #将n_feats的值赋值到out_dim中
            self.out_dim = n_feats
        else:
            #否则将args.n_colors的值赋值给self.out_dim
            self.out_dim = args.n_colors
            # define tail module  定义尾部的模型
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            # nn.Sequential() 可以允许将整个容器视为单个模块（即相当于把多个模块封装成一个模块），
            # forward()方法接收输入之后，nn.Sequential()按照内部模块的顺序自动依次计算并输出结果
            self.tail = nn.Sequential(*m_tail)
    #forward（）表示在建立模型后，进行神经元网络的前向传播。
    # 也就是说forward就是专门用来计算给定输入，得到神经元网络输出的方法。
    # 在前向传播forward函数中实现批数据的前向传播逻辑，
    # 只要在nn.Module的子类中定义了forward()函数，backward()函数就会被自动实现。
    # 拼接子模块，是在froward()中实现的。
    def forward(self, x):
        #输入x经过head模块、body模块、tail模块 后输出结果
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x
    # 加载预训练模型文件的参数
    # 使用load_state_dict提供的参数strict=False，网络结构名字一致的会被导入，不一致的会被舍弃
    def load_state_dict(self, state_dict, strict=True):
        #state_dict()返回一个包含模块整个状态的字典
        own_state = self.state_dict()
        # .items()以列表返回可遍历的(键, 值) 元组数组,用于 for来循环遍历
        for name, param in state_dict.items():
            if name in own_state:
                # isinstance(object, classinfo)
                # object – 实例对象  classinfo – 可以是直接或间接类名、基本类型或者由它们组成的元组。
                # 如果对象的类型与参数二的类型相同，则返回True,否则返回False
                if isinstance(param, nn.Parameter):
                    param = param.data
                # 异常处理 每当在运行时检测到程序错误时，python就会引发异常
                # 把可能发生错误的语句放在try模块里，用except来处理异常
                try:
                    own_state[name].copy_(param)
                except Exception:
                    # find() 方法检测字符串中是否包含子字符串
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
                                        # “在复制名为{name}的参数时，”
                                        # '模型中的维数为{ own_state[name].size()}'
                                        # ，它在检查点中的维度是{param.size()}
            #如果strict的值为True
            elif strict:
                #name中有'tail'的值为-1的话
                if name.find('tail') == -1:
                    #报错 状态字典中出现意外的键名
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
#注册器模块
# 一个深度学习项目可能支持多种模型；具体使用哪种模型可能是用户在配置文件中指定的。最简单的实现方式，就是维护一个模型名称->模型类的字典。
# 但每当你增加一个模型时，这个字典就需要手动维护，比较繁琐。使用注册器的模块，需要维护的是需要注册的模块的代码路径。
@register('edsr-baseline')
# 残差块的数量是16   期望输入的特征数为64
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1):
    # 命名空间(Namespace)是从名称到对象的映射。各个命名空间是独立的，没有任何关系，所以一个命名空间不能有重名，
    # 但不同的命名空间是可以重名而没有任何影响
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]#为什么加[]?
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3 #表示3通道
    return EDSR(args)

#'edsr'注册器模块
@register('edsr')
# 残差块的数量是32   期望输入的特征数为64
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)
