import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        # 深复制model_spec['args']这个参数
        model_args = copy.deepcopy(model_spec['args'])
        #更新args的值
        model_args.update(args)
    else:
        #将args的值赋值给model_args
        model_args = model_spec['args']
        #传入多个字典形式的model_args
    model = models[model_spec['name']](**model_args)
    #如果load_sd的值为True
    if load_sd:
        #load_state_dict(state_dict, strict=True)
        #从 state_dict 中复制参数和缓冲区到 Module 及其子类中
        #state_dict：包含参数和缓冲区的 Module 状态字典
        # strict：默认 True，是否严格匹配 state_dict 的键值和 Module.state_dict()的键值
        model.load_state_dict(model_spec['sd'])
        #返回model
    return model
