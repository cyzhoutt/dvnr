import os, sys, json, time, random, pdb

import numpy as np

from tqdm import tqdm, trange

import torch
import torch.optim as optim

from .networks import *
from .samplers import *
from .utils import *


def create_model(tcnn=False, n_output_dims=1, n_hidden_layers=2, n_neurons=64,
                 n_levels=8, n_features_per_level=4, 
                 base_resolution=4, log2_hashmap_size=19,
                 per_level_scale=2.0,
                 **kwargs):
    '''
    Instantiate the INR model.
    '''
    if torch.cuda.is_available():
        NETWORK = INR_TCNN if tcnn else INR_Base
    else:
        NETWORK = INR_Base

    model = NETWORK(n_output_dims=n_output_dims,
                    n_levels=n_levels,
                    n_features_per_level=n_features_per_level,
                    log2_hashmap_size=log2_hashmap_size, 
                    base_resolution=base_resolution,
                    n_hidden_layers=n_hidden_layers,
                    n_neurons=n_neurons,
                    per_level_scale=per_level_scale)
    model.to(DEVICE)

    # print(model)

    return model


def get_boundary_slice_gauss(self, res, face, device=DEVICE):
    num_points = res * res * 2
    # Generate random face choices, coordinates, and 0/1 choices
    coord1 = torch.rand(num_points, device=device)
    coord2 = torch.rand(num_points, device=device)
    # Generate a mirrored gaussian    
    cvalue = torch.clamp(torch.randn(num_points, device=device) * 0.001, -1, 1)
    cvalue[cvalue < 0] += 1
    # Set the coordinates based on the face choices
    size = (self.bounds_hi - self.bounds_lo)
    if face == 0:
        coords = torch.stack((cvalue, coord1, coord2), axis=1)
    elif face == 1:
        coords = torch.stack((coord1, cvalue, coord2), axis=1)
    else:
        coords = torch.stack((coord1, coord2, cvalue), axis=1)
    return coords * size + self.bounds_lo


# def get_boundary_slice_fixed(self, axis, value, dims):
#     spacing = 0.5 / self.dims
#     s = spacing[axis].item()
#     lo = self.bounds_lo + spacing
#     hi = self.bounds_hi - spacing
#     def linspace(r, d): 
#         return torch.linspace(r[0], r[1], d, dtype=torch.float32, device=DEVICE)
#     if axis == 0:
#         x, y, z = torch.meshgrid(
#             linspace([value - s, value + s], 5),
#             linspace([lo[1].item(), hi[1].item()], dims[1]), 
#             linspace([lo[2].item(), hi[2].item()], dims[2]), 
#             indexing='xy')
#     elif axis == 1:
#         x, y, z = torch.meshgrid(
#             linspace([lo[0].item(), hi[0].item()], dims[0]),
#             linspace([value - s, value + s], 5),
#             linspace([lo[2].item(), hi[2].item()], dims[2]),
#             indexing='xy')
#     else:
#         x, y, z = torch.meshgrid(
#             linspace([lo[0].item(), hi[0].item()], dims[0]),
#             linspace([lo[1].item(), hi[1].item()], dims[1]),
#             linspace([value - s, value + s], 5),
#             indexing='xy')
#     coordinates = torch.stack((x, y, z), axis=3).permute(2, 0, 1, 3)
#     return coordinates.reshape([-1, 3])


def gen_randn_distribution(nsamples, std):
    coords = torch.randn(nsamples, device=DEVICE, dtype=torch.float32) * std
    # where coords are negative, we flip them to the positive side
    coords[coords < 0] += 1
    return torch.clamp(coords, 0, 1)


def train(sampler, model, verbose=True, max_steps=50, psnr_target=-1, lrate=1e-2, lrate_decay=250, batchsize=1024*64, weight=0.5, weight_std=0.01, **kwargs):
    '''
    Train the INR model.
    '''

    # query = lambda inputs : run_network(inputs, fn=model)
    query = model

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lrate, 
        betas=(0.9, 0.999), eps=1e-09, # weight_decay=1e-15, 
        # amsgrad=True, foreach=True #, fused=True,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lrate_decay, gamma=0.5)

    # s = math.isqrt(batchsize) // 4 # --> a hand-tuned magic number here

    # if weight > 0.0:
    #     s = math.isqrt(batchsize) // 4 # --> a hand-tuned magic number here
    #     boundary_coords = torch.cat([
    #         get_decoding_inputs_slice(sampler, 0, sampler.bounds_lo[0].item(), sampler._dims),
    #         get_decoding_inputs_slice(sampler, 0, sampler.bounds_hi[0].item(), sampler._dims),
    #         get_decoding_inputs_slice(sampler, 1, sampler.bounds_lo[1].item(), sampler._dims),
    #         get_decoding_inputs_slice(sampler, 1, sampler.bounds_hi[1].item(), sampler._dims),
    #         get_decoding_inputs_slice(sampler, 2, sampler.bounds_lo[2].item(), sampler._dims),
    #         get_decoding_inputs_slice(sampler, 2, sampler.bounds_hi[2].item(), sampler._dims)
    #     ])
    #     # print(boundary_coords.shape)

    # Training
    step = 0
    def task():
        nonlocal step

        optimizer.zero_grad()

        # Reconstruction Loss
        with torch.no_grad():
            if weight > 0.0:
                N = (int(batchsize*weight)//3 + 127) // 128 * 128
                coords = torch.rand(batchsize, 3, device=DEVICE, dtype=torch.float32)
                coords[0*N:1*N, 0] = gen_randn_distribution(N, weight_std)
                coords[1*N:2*N, 1] = gen_randn_distribution(N, weight_std)
                coords[2*N:3*N, 2] = gen_randn_distribution(N, weight_std)
                coords = coords * sampler.sampling_scale + sampler.bounds_lo
                targets = sampler.get_samples(coords)
            else:
                coords, targets = sampler.get_random_samples(batchsize)

        values = query(coords)
        loss = torch.nn.L1Loss()(values, targets)

        # # Boundary Loss
        # if weight > 0.0:
        #     boundary_coords = torch.cat([
        #         get_boundary_slice_gauss(sampler, s, 0),
        #         get_boundary_slice_gauss(sampler, s, 1),
        #         get_boundary_slice_gauss(sampler, s, 2)
        #     ])
        #     boundary_targets = sampler.get_samples(boundary_coords)
        #     boundary_values  = query(boundary_coords)
        #     loss = (1 - weight) * loss + weight * torch.nn.L1Loss()(boundary_values, boundary_targets)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation
        with torch.no_grad():
            psnr = mse2psnr(mse_loss(values, targets))
            # if weight > 0.0:
            #     psnr = min(psnr, mse2psnr(mse_loss(boundary_values, boundary_targets)))

        step += 1
        return loss, psnr

    # Training Loop
    time0 = time.time()

    if psnr_target > 0:
        psnr_ema = 0.0
        alpha = 0.8
        while True:
            loss, psnr = task()
            psnr_ema = alpha * (psnr - psnr_ema) + psnr_ema
            if (psnr_ema > psnr_target):
                break
    else:
        progress = trange(1, max_steps+1) if verbose else range(1, max_steps+1)
        for i in progress:
            loss, psnr = task()
            if verbose: progress.set_postfix_str(
                f'Loss: {loss:7.6f}, PSNR: {psnr:5.3f}dB, lrate: {scheduler.get_last_lr()[0]:5.3f}', refresh=True
            ) 

    total_time = time.time()-time0
    if verbose: print(f'[info] total training time: {total_time:5.3f}s, steps: {step}')
    return total_time


def _compress(sampler, numfields=1, verbose=True, evaluate=False, **kwargs):

    # Create a neural repr
    model = create_model(n_output_dims=numfields, **kwargs)

    # ...
    time = train(sampler, model, verbose=verbose, **kwargs)

    if evaluate:
        with torch.no_grad():
            mse = sampler.compute_mse(model, verbose=verbose)
        if verbose: print(f"[info] mse: {mse}, psnr {mse2psnr(mse)}")
    else:
        mse = float('inf')

    return model.state_dict(), mse, mse2psnr(mse), time


def _decode(state, dims, dtype, minmax, numfields, **kwargs):
    with torch.no_grad():
        model = create_model(n_output_dims=numfields, **kwargs)
        model.load_state_dict(state)
        sampler = VolumeSampler(dims=dims, dtype=dtype, numfields=numfields)
        dmin, dmax = minmax if minmax else [0.0, 1.0]
    return sampler, model, dmin, dmax


# -------------------------------
# PUBLIC API
# -------------------------------

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def compute_macrocell(vdims, model, mcsize_mip=4):

    mcsize = 1 << mcsize_mip

    mcdims = [
        (vdims[0] + mcsize - 1) // mcsize,
        (vdims[1] + mcsize - 1) // mcsize,
        (vdims[2] + mcsize - 1) // mcsize
    ]

    vscale = Tensor([
        1 / vdims[0],
        1 / vdims[1],
        1 / vdims[2]
    ], dtype=torch.float32)
 
    mcscale = Tensor([
        mcsize / vdims[0],
        mcsize / vdims[1],
        mcsize / vdims[2]
    ], dtype=torch.float32)
 
    mcrange = torch.zeros(mcdims[2], mcdims[1], mcdims[0], 2)

    def generate_grid(dimx, dimy, dimz):
        x, y, z = torch.meshgrid(
            torch.linspace(0, (dimx - 1), dimx, dtype=torch.int32, device=DEVICE),
            torch.linspace(0, (dimy - 1), dimy, dtype=torch.int32, device=DEVICE),
            torch.linspace(0, (dimz - 1), dimz, dtype=torch.int32, device=DEVICE),
            indexing='ij'
        )
        coordinates = torch.stack((x, y, z), axis=3).permute(2, 1, 0, 3)
        return coordinates.reshape([-1, 3])

    grid = generate_grid(mcsize+1, mcsize+1, mcsize+1).float() - Tensor(0.5, dtype=torch.float32)
    n = grid.shape[0]

    for iz in range(mcdims[2]):
        m = mcdims[1] * mcdims[0]
        o = generate_grid(mcdims[0], mcdims[1], 1) + Tensor([0, 0, iz], dtype=torch.float32)
        x = grid.repeat(m, 1).reshape([m, n, 3]) + o.unsqueeze(dim=1) * mcsize
        x = x.reshape([-1, 3])
        x = (x * vscale).clamp(min=0, max=1)
        with torch.no_grad():
            y = batchify(model, 64*1024)(x)
            y = y.reshape([m, n])
        vmin, _ = y.min(dim=1)
        vmax, _ = y.max(dim=1)
        mcrange[iz, :, :, 0] = vmin.reshape([mcdims[1], mcdims[0]])
        mcrange[iz, :, :, 1] = vmax.reshape([mcdims[1], mcdims[0]])

    # finally, we remap the value range minmax
    mcrange[:, :, :, 0] = mcrange[:, :, :, 0] - 1
    mcrange[:, :, :, 1] = mcrange[:, :, :, 1] + 1

    return {
        "volumedims": vdims,
        "dims": mcdims,
        "spacings": [ mcsize / vdims[0], mcsize / vdims[1], mcsize / vdims[2] ],
        "value_ranges": mcrange.cpu().numpy()
    }


def create_inr_scene(fn, dims, model):
    import bson

    encoding_config = model.encoding_config
    network_config  = model.network_config

    mc = compute_macrocell(dims, model)

    root = bson.dumps({
        "macrocell": {
            "data": mc["value_ranges"].tobytes(),
            "dims": {
                "x": mc["dims"][0],
                "y": mc["dims"][1],
                "z": mc["dims"][2]
            },
            "groundtruth": False,
            "spacings": {
                "x": mc["spacings"][0],
                "y": mc["spacings"][1],
                "z": mc["spacings"][2]
            }
        },
        "model": {
            "encoding": encoding_config,
            "loss": { "otype": "L1" },
            "network": network_config
        },
        "parameters": {
            "params_binary": model.state_dict()["model.params"].half().cpu().numpy().tobytes()
        },
        "volume": {
            "dims": {
                "x": dims[0],
                "y": dims[1],
                "z": dims[2]
            }
        }
    })

    with open(fn, 'wb') as f:
        f.write(root)


def compress_file(filename, dims:List[int], dtype:str, numfields:int=1, minmax=None, dbounds=None, evaluate=False, **kwargs):
    sampler = VolumeSampler(dims, dtype, numfields, dbounds=dbounds)
    sampler.load_from_file(filename, minmax=minmax)
    return _compress(sampler, numfields=numfields, evaluate=evaluate, **kwargs)


def compress_ndarray(data, dims:List[int], dtype:str, numfields:int=1, minmax=None, dbounds=None, evaluate=False, **kwargs):
    sampler = VolumeSampler(dims, dtype, numfields, dbounds=dbounds)
    sampler.load_from_ndarray(data, minmax=minmax)
    return _compress(sampler, numfields=numfields, evaluate=evaluate, **kwargs)


def compress_callback(callback, dims:List[int], dtype:str, numfields:int=1, minmax=None, dbounds=None, evaluate=False, **kwargs):
    callback.print()
    sampler = VolumeSampler(dims, dtype, numfields, dbounds=dbounds)
    sampler.load_from_callback(callback)
    return _compress(sampler, numfields=numfields, evaluate=evaluate, **kwargs)


def decode_to_file(filename, state, dims, dtype, minmax=None, numfields=1, **kwargs):
    with torch.no_grad():
        sampler, query, dmin, dmax = _decode(state=state, dims=dims, dtype=dtype, minmax=minmax, numfields=numfields, **kwargs)
        with open(filename, 'wb') as f:
            def callback(values, _):
                (values * (dmax - dmin) + dmin).tofile(f)
            sampler.decode_volume(callback, query)


def decode_to_ndarray(memory, state, dims, dtype, minmax=None, numfields=1, **kwargs):
    with torch.no_grad():
        sampler, query, dmin, dmax = _decode(state=state, dims=dims, dtype=dtype, minmax=minmax, numfields=numfields, **kwargs)
        ndarray = np.asarray(memory).reshape((dims[2], -1))
        def callback(values, z):
            ndarray[z,:] = (values * (dmax - dmin) + dmin).flatten()
        sampler.decode_volume(callback, query)
