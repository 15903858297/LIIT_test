""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import numpy as np
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr


#加载数据集
# Dataset：是被封装进DataLoader里，实现该方法封装自己的数据和标签。
# DataLoader：被封装入DataLoaderIter里，实现该方法达到数据的划分。
def make_data_loader(spec, tag=''): #具体的；特定的
    if spec is None:
        return None
    #dataset在程序中起到的作用是告诉程序数据在哪，每个索引所对应的数据是什么。
    # 相当于一系列的存储单元，每个单元都存储了数据。
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    #items() 函数以列表返回可遍历的(键, 值) 元组数组
    #把字典中每对 key 和 value 组成一个元组,并把这些元组放在列表中返回
    #提取标量值
    for k, v in dataset[0].items():
        # tuple() 函数将列表转换为元组
        log('  {}: shape={}'.format(k, tuple(v.shape)))#对shape的理解
    #dataloader是一个装载数据集的一个工具，从dataset中取数据
    #dataset: Dataset类， 决定数据从哪读取以及如何读取、bathsize: 批大小、num_works: 是否多进程读取机制、
    # shuffle: 每个epoch是否乱序、drop_last: 当样本数不能被batchsize整除时， 是否舍弃最后一批数据
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    #yaml文件里面的'resume'值不是空
    if config.get('resume') is not None:
        #torch.load()用来加载torch.save() 保存的模型文件
        sv_file = torch.load(config['resume'])
        #更新模型的参数并将模型加载到GPU上去
        model = models.make(sv_file['model'], load_sd=True).cuda()
        #更新优化器参数
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        #设置训练开始的轮次，从保存的模型文件中的'epoch'值加1
        epoch_start = sv_file['epoch'] + 1
        #如果yaml文件里面的'multi_step_lr'的值为空
        if config.get('multi_step_lr') is None:
            """在实际训练中学习率最好是能够动态变化的。一般情况下，
            希望学习率一开始比较高，因为一开始训练的时候我们需要更快的学习速度，
            梯度下降法能够更快的帮助我们的模型参数到达一个比较好的值。
            但是当模型训练到一定程度后，学习速度能够降下来，相当于那个时候的训练会是一个微调训练的过程。
            pytorch中的lr_sheduler就满足了这样的需求"""
            lr_scheduler = None
        else:
            #使用MultiStepLR动态调整学习率
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        # 其中“-”只是一个占位符，可以把它理解为i或者j等等任意的字母
        for _ in range(epoch_start - 1):
            #step()生成lr_scheduler阶梯图
            lr_scheduler.step()
    #如果yaml文件里面的'resume'值不是空
    else:
        #更新模型的参数并将模型加载到GPU上去
        model = models.make(config['model']).cuda()
        #更新优化器的参数
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        #定义训练开始轮次为1
        epoch_start = 1
        #如果yaml文件中的'multi_step_lr'值是空
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        #否则
        else:
            #使用MultiStepLR动态调整学习率，传入优化器参数，传入多个'multi_step_lr'的值
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    #将模型设置为训练状态，启用batch normalization和Dropout
    #batch normalization批标准化, 和普通的数据标准化类似, 是将分散的数据统一的一种做法
    # Dropout能够有效缓解模型的过拟合问题，从而使得训练更深更宽的网络成为可能
    model.train()
    #损失函数nn.L1Loss()
    # 作用是计算网络输出与标签之差的绝对值，返回的数据类型可以是张量，也可以是标量。
    loss_fn = nn.L1Loss()
    #输出训练的损失值的平均值
    train_loss = utils.Averager()
    #从yaml文件中获得‘data_norm’的值
    data_norm = config['data_norm']
    #将yaml文件中的'inp'值赋值给t
    t = data_norm['inp']
    #torch.FloatTensor() 类型转换, 将list ,numpy转化为tensor
    # view()的作用相当于numpy中的reshape，重新定义矩阵的形状
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()#将操作对象放在GPU内存中
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    #tqdm是python中的一个用来供我们创建进度条的库。在进行深度学习的研究时，使用这个库为我们直观地展示当前的训练进度
    #将进度条的前缀信息设置为：train
    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        #计算pred与gt之差的绝对值
        loss = loss_fn(pred, gt)

        #调用utils.Averager()中的add函数
        train_loss.add(loss.item()) #.item()把字典中每对key和value组成一个元组

        #.zero_grad()梯度初始化为零
        optimizer.zero_grad()
        #反向传播求梯度
        loss.backward()
        #更新所有参数
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer #global关键字表明被其修饰的变量是全局变量
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:    #  'w'表示以只写的方式打开文件夹
        #os.path.join用于将多个路径拼接为一个完整路径
        yaml.dump(config, f, sort_keys=False)
        #sort_keys：是否按照字典排序（a-z）输出，True代表是，False代表否。
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None: #获取data_norm的值
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    #字典用花括号编写，拥有键和值。
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    #使用os.environ['CUDA_VISIBLE_DEVICES'] 指定了GPU
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)#使用nn.DataParallel函数来用多个GPU来加速训练

    epoch_max = config['epoch_max'] #epoch_max=1000
    epoch_val = config.get('epoch_val') #epoch_val=1
    epoch_save = config.get('epoch_save') #epoch_save=100
    max_val_v = -1e18 #科学记数法 表示1乘10的18次方 就是1后面18个零

    timer = utils.Timer()
    #参数epoch用于指定模型训练的迭代轮次
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        #利用add_scalar函数来生成一个“lr”名称的二维函数，横轴X值为“optimizer.param_groups[0][‘lr’]”，纵轴Y值为global_step。
        #第一个参数：生成图像的名称 第二个参数：X轴的值 第三个参数：Y轴的值
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()#阶梯图step()
        #在log文件中追加训练损失的值
        log_info.append('train: loss={:.4f}'.format(train_loss))#.4f保留小数点后四位
        #生成损失率的图
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module  #多个GPU加速训练
        else:
            model_ = model

        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()#仅保存学习到的参数
        optimizer_spec = config['optimizer']    #optimizer参数用于指定优化器实例
        #state_dict()是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        #torch.save()保存整个网络模型
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        #后面的参数代表保存文件的绝对路径+保存文件名
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):# %代表取模，返回除法的余数
            #如果GPU的数量大于一个，并且'eval_bsize'的值不是空
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                #调用多个GPU加速训练
                model_ = model.module#多个GPU加速训练
            else:
                #如果GPU的数量只有一个，就不调用GPU训练
                model_ = model
            #eval_psnr峰值信噪比
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))
            #将val的值追加到log里面，保留四位小数
            log_info.append('val: psnr={:.4f}'.format(val_res))
            #生成一个名为“psnr峰值信噪比”的二维函数，X轴的值为val_res，Y轴的值为epoch
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            #如果val_res的值大于max_val_v
            if val_res > max_val_v:

                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        #现在完成的进度 现在进行的轮次/总的轮次
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        #调用utils里面定义的time_text函数计算程序运行时间
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        # 往日志中追加时间信息
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))   #elapsed过去的、经过的  log日志
        # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
        log(', '.join(log_info))
        #flush方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区。
        writer.flush()


if __name__ == '__main__':
    """
    argparse是一个Python模块：命令行选项、参数和子命令解析器。
    主要有三个步骤：
    创建 ArgumentParser() 对象
    调用 add_argument() 方法添加参数
    使用 parse_args() 解析添加的参数
    """
    #使用 argparse 的第一步是创建一个 ArgumentParser 对象。
    #ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。

    parser = argparse.ArgumentParser()
    #添加参数
    """
    ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
    default - 当参数未在命令行中出现时使用的值。
    action - 当参数在命令行中出现时使用的动作基本类型。
    nargs - 命令行参数应当消耗的数目。
    const - 被一些 action 和 nargs 选择所需求的常数。
    type - 命令行参数应当被转换成的类型。
    choices - 可用的参数的容器。
    required - 此命令行选项是否可省略 （仅选项可用）。
    help - 一个此选项作用的简单描述。
    metavar - 在使用方法消息中使用的参数值示例。
    dest - 被添加到 parse_args() 所返回对象上的属性名。
    给属性名之前加上“- -”，就能将之变为可选参数
    """
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    #GPU0 是集显=主板自带的显卡
    # GPU1就独显，是单独的一张显卡性能一般是会比集显要高

#解析参数
    args = parser.parse_args()
    #设置当前使用的GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#可以在语句结束后，关闭文件流。不用with关键字，文件会被python垃圾回收关闭
    #mode的参数是 ‘r’，表示文件只能读取
    with open(args.config, 'r') as f:
        """使用python的load()方法读取yaml文件内容
        在 yaml.load 方法中， loader 参数有四种：
        ①BaseLoader：载入大部分的基础YAML
        ②SafeLoader：载入YAML的子集，推荐在不可信的输入时使用
        ③FullLoader：这是默认的载入方式，载入全部YAML
        ④UnsafeLoader：老版本的载入方式
        """
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
        # ('/')表示以/作为分隔符   [-1]表示获取最后一个参数
        #solit() 对字符串进行分割,分隔后的字符串以列表方式返回
        #[:]使用方括号的形式截取字符
        #[:-len('.yaml')]???
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
