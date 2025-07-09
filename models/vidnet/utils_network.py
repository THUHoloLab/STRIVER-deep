import torch
import os

# ----------------------------------------
# network name and number of parameters
# ----------------------------------------
def info_network(model):
    msg = '\n'
    msg += 'Networks name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# ----------------------------------------
# parameters description
# ----------------------------------------
def info_params(model):

    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
    return msg


# ----------------------------------------
# save the state_dict of the network
# ----------------------------------------
def save_network(save_dir, network, epoch):
    save_filename = 'network_{}.pth'.format(epoch)
    save_path = os.path.join(save_dir, save_filename)
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


# ----------------------------------------
# save the state_dict of the optimizer
# ----------------------------------------
def save_optimizer(save_dir, optimizer, epoch):
    save_filename = 'optimizer_{}.pth'.format(epoch)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(optimizer.state_dict(), save_path)
    