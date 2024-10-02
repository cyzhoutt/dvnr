
import json
import pdb
import os

import numpy as np
import torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass


# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# elif hasattr(torch, "xpu"):
#     DEVICE = torch.device("xpu")
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# else:
    DEVICE = torch.device("cpu")


def Tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs, device=DEVICE)


def coherent_prime_hash(coords, log2_hashmap_size=0):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429,
              2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    if log2_hashmap_size == 0:
        return xor_result
    else:
        return torch.tensor((1 << log2_hashmap_size)-1).to(xor_result.device) & xor_result


# def batchify(fn, chunk):
#     '''
#     Constructs a version of 'fn' that applies to smaller batches.
#     '''
#     if chunk is None:
#         return fn
#     def ret(inputs):
#         return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
#     return ret


# def run_network(inputs, fn, netchunk=1024*64):
#     '''
#     Prepares inputs and applies network 'fn'.
#     '''
#     inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
#     outputs_flat = batchify(fn, netchunk)(inputs_flat)
#     outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
#     return outputs.squeeze().float()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def find_next_run_number(folder):
    if not os.path.exists(folder): return 0
    files = os.listdir(folder)
    files = sorted([f for f in files if f.startswith('run')])
    if len(files) == 0: return 0
    # import pdb; pdb.set_trace();
    return int(files[-1][3:]) + 1


def create_logger(basedir, expname, resume=False, purge_step=None):
    from torch.utils.tensorboard import SummaryWriter
    import shutil

    exp_dir = os.path.join(basedir, expname)
    if not resume:
        if '/' in expname:
            if os.path.exists(exp_dir):
                val = input("The experiment directory %s exists. Overwrite? (y/n) " % exp_dir)
                if val == 'y': shutil.rmtree(exp_dir)
            os.makedirs(exp_dir)
            logger = SummaryWriter(exp_dir, purge_step=purge_step)
            logger.model_dir = exp_dir
        else:
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            model_dir = os.path.join(exp_dir, 'run%05d' % find_next_run_number(exp_dir))
            logger = SummaryWriter(model_dir, purge_step=purge_step)
            logger.model_dir = model_dir
    else:
        if not os.path.exists(exp_dir):
            raise Exception(f"Cannot resume from an non-existing directory: {exp_dir}.")
        logger = SummaryWriter(exp_dir, purge_step=purge_step)
        logger.model_dir = exp_dir

    return logger


def l1_loss(x, y):
    return torch.nn.L1Loss()(x, y) if torch.is_tensor(x) else np.absolute(x - y).mean()


def mse_loss(x, y):
    return torch.nn.MSELoss()(x, y) if torch.is_tensor(x) else np.mean((x - y) ** 2)


def mse2psnr(x, data_range=1.0):
    x = x / data_range * data_range
    return (-10. * torch.log(x) / torch.log(Tensor(10.))) if torch.is_tensor(x) else (-10. * np.log(x) / np.log(10.))


if __name__ == "__main__":
    pass
