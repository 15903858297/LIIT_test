import copy


datasets = {}#定义datasets为一个空字典


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    # 如果args的值不是空：
    if args is not None:
        #copy.deepcopy()是深复制函数。
        #从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
        #深复制dataset_spec['args']这个参数，使dataset_args与dataset_spec['args']不在同一个内存地址，变量的值不受影响
        dataset_args = copy.deepcopy(dataset_spec['args'])
        #update()函数用于更新字典中的键值对。将两个字典合并操作，有相同的就覆盖
        #update()方法语法：
        #dict.update(dict2)
        #更新args的值，如果是不同的键，则合并，如果是相同的键，则更新键对应的值
        dataset_args.update(args)
    #如果args的值为空：
    else:
        #将dataset_spec['args']的值赋值给dataset_args
        dataset_args = dataset_spec['args']
    #不确定将来要往函数中传入多少个参数，即可使用可变参数（即不定长参数），用*args,**kwargs表示
    # *args称之为Non-keyword Variable Arguments，无关键字参数；
    # 当函数中以列表或者元组的形式传参时，就要使用*args；
    # **kwargs称之为keyword Variable Arguments，有关键字参数；
    # 当传入字典形式的参数时，就要使用**kwargs。
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
