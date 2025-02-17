#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
import readline
from fused_ssim import fused_ssim
import cv2

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def fast_ssim(img1, img2):
    ssim_map = fused_ssim(img1, img2)
    return ssim_map.mean()

def loss_fn(
    output_image: torch.Tensor, image: torch.Tensor, lambda_l1: float
) -> torch.Tensor:
    l1_loss = torch.nn.L1Loss()(output_image, image)
    # l1_loss = torch.Tensor([0]).to("cuda")
    ssim_loss = 1 - fused_ssim(output_image.unsqueeze(0), image.unsqueeze(0))
    # ssim_loss = torch.Tensor([0]).to("cuda")
    total_loss = lambda_l1 * l1_loss + (1 - lambda_l1) * (ssim_loss)
    return {"l1_loss": l1_loss, "ssim_loss": ssim_loss, "total_loss": total_loss}

def save_image(image, path):
    image = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # import pdb; pdb.set_trace()
    optimizer = torch.optim.Adam(gaussians.get_trainable_parameters(), lr=0.01)

    for i in range(10):
        for idx, view in enumerate(tqdm(views[:1], desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3, :, :]

            if args.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]
            save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            if i == 0:
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            loss = loss_fn(rendering, gt, lambda_l1=0.8)
            loss["total_loss"].backward()
            rand_idx_start = 3
            rand_idx_end = 4
            print("Opacity grad: ", gaussians._opacity.grad[rand_idx_start:rand_idx_end])
            print("Quaternion grad: ", gaussians._rotation.grad[rand_idx_start:rand_idx_end])
            print("Scaling grad: ", gaussians._scaling.grad[rand_idx_start:rand_idx_end])
            print("Points grad: ", gaussians._xyz.grad[rand_idx_start:rand_idx_end])

            # import pdb; pdb.set_trace()
            # gaussians._opacity.grad
            print(f"loss: {loss['total_loss']}")
            print(f"l1_loss: {loss['l1_loss']}")
            print(f"ssim_loss: {loss['ssim_loss']} \n\n\n\n")
            optimizer.step()
            optimizer.zero_grad()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    print(f"bg_color: {bg_color}")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
    # if not skip_test:
    #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
