"""
Source:
https://github.com/krrish94/nerf-pytorch
https://github.com/krrish94/nerf-pytorch/blob/master/train_nerf.py
"""

import argparse
import glob
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import matplotlib.image

import imageio

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)

class UncertaintyLoss(torch.nn.Module):
    """
    Calculate uncertainty loss.
    Uncertainty loss is the weigthed sum of mean squared loss and regularization loss
    """

    def __init__(self):
        super(UncertaintyLoss, self).__init__()
    
    def forward(
        self,
        rgb_coarse_predicted: torch.Tensor,
        rgb_fine_predicted: torch.Tensor,
        target_ray_values: torch.Tensor,
        delta_coarse: torch.Tensor,
        delta_fine: torch.Tensor,
        num_sample_coarse: int, # number of sample points on a ray for coarse model
        num_sample_fine: int, # number of sample points on a ray for fine model
        alpha1: float = 0.1,
        alpha2: float = 0.01,
    ) -> torch.Tensor:
        """
        Forward pass
        """

        """
        Coarse model 
        """
        diff_pred_gt_coarse = torch.subtract(rgb_coarse_predicted, target_ray_values[..., :3])

        # Calculate the E(r) which is square of L2-norm (square of Euclidean distance) for each ray
        diff_pred_gt_coarse_sq = torch.square(diff_pred_gt_coarse)
        Er_coarse = torch.sum(diff_pred_gt_coarse_sq, dim=-1)

        # Calculate the square error loss (Lse)
        Er_coarse_unsq = Er_coarse.unsqueeze(1)
        Er_coarse_expand = Er_coarse_unsq.expand(-1, num_sample_coarse)
        Er_coarse_uncertainty = Er_coarse_expand - delta_coarse
        Lse_max_coarse = torch.max(Er_coarse_uncertainty, torch.zeros_like(Er_coarse_uncertainty))
        Lse_coarse = torch.sum(Lse_max_coarse)

        # Calculate the regularization loss (L0)
        L0_max_coarse = torch.max(delta_coarse, torch.zeros_like(delta_coarse))
        L0_coarse = torch.sum(L0_max_coarse)

        # Uncertainty loss (L_unct) with weighted sum of square error loss (Lse) and regularization loss (L0)
        L_unct_coarse = alpha1 * Lse_coarse + alpha2 * L0_coarse

        """
        Fine model
        """
        L_unct_fine = None
        if rgb_fine_predicted is not None:
            diff_pred_gt_fine = torch.subtract(rgb_fine_predicted, target_ray_values[..., :3])

            # Calculate the E(r) which is square of L2-norm (square of Euclidean distance) for each ray
            diff_pred_gt_fine_sq = torch.square(diff_pred_gt_fine)
            Er_fine = torch.sum(diff_pred_gt_fine_sq, dim=-1)

            # Calculate the square error loss (Lse)
            Er_fine_unsq = Er_fine.unsqueeze(1)
            Er_fine_expand = Er_fine_unsq.expand(-1, num_sample_coarse+num_sample_fine)
            Er_fine_uncertainty = Er_fine_expand - delta_fine
            Lse_max_fine = torch.max(Er_fine_uncertainty, torch.zeros_like(Er_fine_uncertainty))
            Lse_fine = torch.sum(Lse_max_fine)

            # Calculate the regularization loss (L0)
            L0_max_fine = torch.max(delta_fine, torch.zeros_like(delta_fine))
            L0_fine = torch.sum(L0_max_fine)

            # Uncertainty loss (L_unct) with weighted sum of square error loss (Lse) and regularization loss (L0)
            L_unct_fine = alpha1 * Lse_fine + alpha2 * L0_fine
        
        return [L_unct_coarse, L_unct_fine]

def transferFunction(delta: torch.Tensor) -> torch.Tensor:

    # r = 1.0*torch.exp( -(delta - 0.9)**2/1.0 ) +  0.1*torch.exp( -(delta - 0.2)**2/0.1 ) +  0.1*torch.exp( -(delta - 0.001)**2/0.01 )
    # g = 1.0*torch.exp( -(delta - 0.9)**2/1.0 ) +  1.0*torch.exp( -(delta - 0.2)**2/0.1 ) +  0.1*torch.exp( -(delta - 0.001)**2/0.01 )
    # b = 0.1*torch.exp( -(delta - 0.9)**2/1.0 ) +  0.1*torch.exp( -(delta - 0.2)**2/0.1 ) +  1.0*torch.exp( -(delta - 0.001)**2/0.01 )
    # a = 1.0*torch.exp( -(delta - 0.9)**2/1.0 ) +  0.01*torch.exp( -(delta - 0.2)**2/0.1 ) + 0.001*torch.exp( -(delta - 0.001)**2/0.01 )

    # a = 0.95*delta + 0.0

    max = torch.max(delta)
    min = torch.min(delta)
    a = (delta - min)/(max - min)

    return a

def visualize_uncertainty(rgb_fine: torch.Tensor, delta_coarse: torch.Tensor, delta_fine: torch.Tensor, radiance_field: torch.Tensor, device):
    """
    Generate the uncertainty visualization images
    """
    #print("rgb_fine.shape: ", rgb_fine.shape)   # torch.Size([400, 400, 3])
    #print("delta_coarse.shape: ", delta_coarse.shape) # torch.Size([160000, 128])
    #print("delta_fine.shape: ", delta_fine.shape)   # torch.Size([160000, 128])
    
    #n_delta = np.sqrt(delta_fine.shape[0]).astype(int)
    x_delta = rgb_fine.shape[0]
    y_delta = rgb_fine.shape[1]
    d_delta = delta_fine.shape[-1]
    delta_fine = delta_fine.view(x_delta, y_delta, d_delta)
    #print("delta_fine.shape: ", delta_fine.shape)   # torch.Size([400, 400, 128])
    #print("type(delta_fine): ", type(delta_fine))

    #print("radiance_field.shape: ", radiance_field.shape)
    #n_radianceField = np.sqrt(radiance_field.shape[0]).astype(int)
    x_radianceField = rgb_fine.shape[0]
    y_radianceField = rgb_fine.shape[1]
    d_radianceField = radiance_field.shape[1]
    rgb_radianceField = radiance_field.view(x_radianceField, y_radianceField, d_radianceField, -1)
    #print("rgb_radianceField.shape: ", rgb_radianceField.shape)

    #print("delta_fine (max): {:.4f}".format(torch.max(delta_fine).item()))
    #print("delta_fine (min): {:.4f}".format(torch.min(delta_fine).item()))

    # delta_max = torch.max(delta_fine)
    # delta_min = torch.min(delta_fine)
    # delta_fine_normal = (delta_fine - delta_min) / (delta_max - delta_min)

    # https://medium.com/swlh/create-your-own-volume-rendering-with-python-655ca839b097
    # https://github.com/pmocz/volumerender-python
    # https://github.com/pmocz/volumerender-python/blob/main/volumerender.py

    a = transferFunction(delta_fine)
    # print("torch.max(a): ", torch.max(a))
    # print("torch.min(a): ", torch.min(a))

    #a = torch.clip(a, 0.0, 1.0)

    # print("torch.max(a) After clipping: ", torch.max(a))   # tensor(1., device='cuda:0')
    # print("torch.min(a) After clipping: ", torch.min(a))   # tensor(0.4609, device='cuda:0')
    #print("a.shape: ", a.shape) # torch.Size([400, 400, 128])

    img_alpha = torch.Tensor(np.zeros((rgb_radianceField.shape[0], rgb_radianceField.shape[1],3))).to(device)
    # img_alpha = torch.Tensor(np.zeros((a.shape[0], a.shape[1]))).to(device)
    depth_alpha = a.shape[-1]

    for i in range(0, depth_alpha, 1):
        #img_alpha = img_alpha + (1 - img_alpha) * a[:,:,i]
        img_alpha[:,:,0] = a[:,:,i]*rgb_radianceField[:,:,i,0] + (1-a[:,:,i])*img_alpha[:,:,0]
        img_alpha[:,:,1] = a[:,:,i]*rgb_radianceField[:,:,i,1] + (1-a[:,:,i])*img_alpha[:,:,1]
        img_alpha[:,:,2] = a[:,:,i]*rgb_radianceField[:,:,i,2] + (1-a[:,:,i])*img_alpha[:,:,2]
    
    #print("img_alpha.shape: ", img_alpha.shape)
    #print("torch.max(img_alpha): ", torch.max(img_alpha))
    #print("torch.min(img_alpha): ", torch.min(img_alpha))

    img_alpha = torch.clip(img_alpha, 0.20, 1.00)

    #print("torch.max(img_alpha) After clipping: ", torch.max(img_alpha))
    #print("torch.min(img_alpha) After clipping: ", torch.min(img_alpha))

    return img_alpha

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # clear memory in GPU CUDA
    torch.cuda.empty_cache()

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
            )
            i_train, i_val, i_test = i_split
            
            H, W, focal = hwf

            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

            # print("images.shape: ", images.shape)   # torch.Size([20, 378, 504, 3])

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    #print("image: ", images[i_test[0]].size())
    # testimg_ = images[i_test[168]].numpy()
    #testimg_ = images[i_test[16]].numpy()
    testimg_ = images[i_test[cfg.nerf.validation.img]].numpy()
    testimg = testimg_[:,:,:3]
    # print("type(testimg): ", type(testimg))
    # print("testimg.ndim: ", testimg.ndim)
    # print("testimg.shape: ", testimg.shape)
    # print("testimg.dtype: ", testimg.dtype)
    matplotlib.image.imsave('testimg_{}.png'.format(str(cfg.experiment.id)), testimg)

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    print("model_coarse: \n", model_coarse)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)
    print("model_fine: \n", model_fine)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    print("optimizer: \n", optimizer)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)

    # remove output figures in the log folder
    # for file in os.listdir(logdir):
    #     os.remove(os.path.join(logdir, file))

    # writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]
    print("start_iter: ", start_iter)

    # TODO: Prepare raybatch tensor if batching random rays

    losses = []
    psnrs = []
    iternums = []

    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        delta_color_coarse, delta_density_coarse = None, None
        delta_color_fine, delta_density_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            # NOTE: run_one_iter_of_nerf() in nerf/train_utils.py
            rgb_coarse, _, _, rgb_fine, _, _, delta_coarse, delta_fine, radiance_field \
            = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        else:
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            # print("X min: {:.3f}; X max: {:.3f} | Y min: {:.3f}; Y max: {:.3f} | Z min: {:.3f}; Z max: {:.3f}"
            #       .format(torch.min(ray_origins[:,:,0]), torch.max(ray_origins[:,:,0]),
            #               torch.min(ray_origins[:,:,1]), torch.max(ray_origins[:,:,1]),
            #               torch.min(ray_origins[:,:,2]), torch.max(ray_origins[:,:,2])))
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _, delta_coarse, delta_fine, radiance_field \
            = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )

            target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0

        # Uncertainty loss
        UncertaintyLossFunc = UncertaintyLoss()
        uncertainty_losses = UncertaintyLossFunc(rgb_coarse, rgb_fine, target_ray_values,
                                               delta_coarse, delta_fine,
                                               num_sample_coarse=cfg.nerf.train.num_coarse,
                                               num_sample_fine=cfg.nerf.train.num_fine,
                                               alpha1=cfg.nerf.alpha1,
                                               alpha2=cfg.nerf.alpha2,
                                               )
        loss_coarse, loss_fine = uncertainty_losses
        loss_uncertainty = loss_coarse + (loss_fine if loss_fine is not None else 0.0)

        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss

        beta1 = cfg.nerf.beta1
        beta2 = cfg.nerf.beta2

        # loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0) + loss_uncertainty
        loss = beta1 * (coarse_loss + (fine_loss if fine_loss is not None else 0.0)) + beta2 * loss_uncertainty
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        # if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
        #     tqdm.write(
        #         "[TRAIN] Iter: "
        #         + str(i)
        #         + " Loss: "
        #         + str(loss.item())
        #         + " PSNR: "
        #         + str(psnr)
        #     )
        # writer.add_scalar("train/loss", loss.item(), i)
        # writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        # if rgb_fine is not None:
        #     writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        # writer.add_scalar("train/psnr", psnr, i)

        loss = 0.0
        psnr = 0.0

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            # tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _, delta_coarse, delta_fine, radiance_field \
                        = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _, delta_coarse, delta_fine, radiance_field \
                    = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss

                rgb_coarse = rgb_coarse.view(-1, 3)
                rgb_fine = rgb_fine.view(-1, 3)
                if cfg.dataset.type.lower() == "blender":
                    target_ray_values = target_ray_values.view(-1, 4)
                elif cfg.dataset.type.lower() == "llff":
                    target_ray_values = target_ray_values.view(-1, 3)
                uncertainty_losses = UncertaintyLossFunc(rgb_coarse, rgb_fine, target_ray_values,
                                               delta_coarse, delta_fine,
                                               num_sample_coarse=cfg.nerf.train.num_coarse,
                                               num_sample_fine=cfg.nerf.train.num_fine,
                                               alpha1=cfg.nerf.alpha1,
                                               alpha2=cfg.nerf.alpha2,
                                               )
                loss_coarse, loss_fine = uncertainty_losses
                loss_uncertainty = loss_coarse + (loss_fine if loss_fine is not None else 0.0)

                beta1 = cfg.nerf.beta1
                beta2 = cfg.nerf.beta2

                # loss = coarse_loss + fine_loss + loss_uncertainty
                loss = beta1 * (coarse_loss + fine_loss) + beta2 * loss_uncertainty
                #loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                # writer.add_scalar("validation/loss", loss.item(), i)
                # writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                # writer.add_scalar("validataion/psnr", psnr, i)
                # writer.add_image(
                #     "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                # )
                # if rgb_fine is not None:
                #     writer.add_image(
                #         "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                #     )
                #     writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                # writer.add_image(
                #     "validation/img_target",
                #     cast_to_image(target_ray_values[..., :3]),
                #     i,
                # )
                # tqdm.write(
                #     "Validation loss: "
                #     + str(loss.item())
                #     + " Validation PSNR: "
                #     + str(psnr)
                #     + " Time: "
                #     + str(time.time() - start)
                # )
                
                losses.append(loss.item())
                psnrs.append(psnr)
                iternums.append(i)

                tqdm.write("[validation] Loss: {:.4f} | PSNR: {:.4f}".format(loss.item(), psnr))

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            #tqdm.write("================== Saved Checkpoint =================")
        
        # Test
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                #test_ray_values = None
                #img_idx = i_test[168]
                #img_idx = i_test[16]
                img_idx = i_test[cfg.nerf.validation.img]
                img_test = images[img_idx].to(device)
                pose_test = poses[img_idx, :3, :4].to(device)
                ray_origins, ray_directions = get_ray_bundle(
                    H, W, focal, pose_test
                )
                rgb_coarse, _, _, rgb_fine, _, _, delta_coarse, delta_fine, radiance_field \
                = run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    model_coarse,
                    model_fine,
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                )

                #print("radiance_field.shape: ", radiance_field.shape)
                #print("rgb_fine.shape: ", rgb_fine.shape)   # torch.Size([400, 400, 3])
                #print("delta_fine.shape: ", delta_fine.shape)   # torch.Size([160000, 128])
                #print("testimg.shape: ", testimg.shape) # (400, 400, 3)
                #print("torch.min(delta_fine): ", torch.min(delta_fine))
                #print("torch.max(delta_fine): ", torch.max(delta_fine))

                #plt.imshow(rgb_coarse.detach().cpu().numpy())
                #plt.savefig(os.path.join(logdir, "coarse_" + str(i).zfill(6) + ".png"))
                #plt.close("all")

                #print("torch.min(rgb_fine): ", torch.min(rgb_fine))
                #print("torch.max(rgb_fine): ", torch.max(rgb_fine))

                # plt.imshow(rgb_fine.detach().cpu().numpy())
                # plt.savefig(os.path.join(logdir, "fine_" + str(i).zfill(6) + ".png"))
                # plt.close("all")

                # savefile = os.path.join(logdir, f"coarse_{i:06d}.png")
                # imageio.imwrite(
                #     savefile, cast_to_image(rgb_coarse[..., :3])
                # )

                savefile = os.path.join(logdir, f"fine_{i:06d}.png")
                imageio.imwrite(
                    savefile, cast_to_image(rgb_fine[..., :3])
                )

                # n = rgb_fine.shape[-1]
                # rgb_fine_std = torch.std(rgb_fine, -1) * n / (n-1)
                # print("rgb_fine_std.shape: ", rgb_fine_std.shape)
                # plt.imshow(rgb_fine_std.detach().cpu().numpy())
                # plt.savefig(os.path.join(logdir, "fine_std_" + str(i).zfill(6) + ".png"))
                # plt.close("all")

                delta_flatten = delta_fine.view(-1)
                #print("delta_color_flatten.shape: ", delta_color_flatten.shape)
                delta_flatten_arr = delta_flatten.detach().cpu().numpy()
                n, bins, patches = plt.hist(x=delta_flatten_arr, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.xlim(0, 0.5)
                plt.ylim(0, 15000000)
                plt.savefig(os.path.join(logdir, "histogram_delta_" + str(i).zfill(6) + ".png"))
                plt.close("all")

                alpha = visualize_uncertainty(rgb_fine, delta_coarse, delta_fine, radiance_field, device)
                # alpha_np = alpha.detach().cpu().numpy()
                # df = pd.DataFrame(alpha_np)
                # df.to_csv(os.path.join(logdir, "alpha_fine_" + str(i).zfill(6) + ".csv"), index=False)

                # alpha = torch.unsqueeze(alpha, dim=2)
                # print("alpha.shape: ", alpha.shape)
                # rgb_fine_alpha = torch.concatenate((rgb_fine, alpha), axis=-1)
                # print("rgb_fine_alpha.shape: ", rgb_fine_alpha.shape)

                # plt.imshow(alpha.detach().cpu().numpy())
                # plt.savefig(os.path.join(logdir, "alpha_fine_" + str(i).zfill(6) + ".png"))
                # plt.close("all")

                savefile = os.path.join(logdir, f"alpha_fine_{i:06d}.png")
                imageio.imwrite(
                    savefile, cast_to_image(alpha[..., :3])
                )

                # delta_normalize_flatten = delta_normalize.view(-1)
                # delta_normalize_flatten_arr = delta_normalize_flatten.detach().cpu().numpy()
                # n, bins, patches = plt.hist(x=delta_normalize_flatten_arr, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
                # plt.xlim(0, 1.0)
                # plt.ylim(0, 15000000)
                # plt.savefig(os.path.join(logdir, "histogram_delta_normalize_" + str(i).zfill(6) + ".png"))
                # plt.close("all")
        
        if i == cfg.experiment.train_iters - 1:
            plt.plot(iternums, psnrs)
            plt.savefig(os.path.join(logdir, "psnr.png"))
            plt.close("all")

            plt.plot(iternums, losses)
            plt.savefig(os.path.join(logdir, "loss.png"))
            plt.close("all")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    #img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    start_time = datetime.now()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start time: ", dt_string)
    
    main()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End time: ", dt_string)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))