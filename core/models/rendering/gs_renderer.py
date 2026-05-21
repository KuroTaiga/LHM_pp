# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-06-04 20:43:18
# @Function      : 3DGSRender Class
import os

import numpy as np
import torch

from core.models.rendering.base_gs_render import BaseGSRender
from core.models.rendering.gaussian_decoder.mlp_decoder import GSMLPDecoder
from core.models.rendering.utils.typing import *
from core.outputs.output import GaussianAppOutput


class GS3DRenderer(BaseGSRender):
    def __init__(
        self,
        human_model_path,
        subdivide_num,
        smpl_type,
        feat_dim,
        query_dim,
        use_rgb,
        sh_degree,
        xyz_offset_max_step,
        mlp_network_config,
        expr_param_dim,
        shape_param_dim,
        clip_scaling=0.2,
        cano_pose_type=0,
        decoder_mlp=False,
        skip_decoder=False,
        fix_opacity=False,
        fix_rotation=False,
        decode_with_extra_info=None,
        gradient_checkpointing=False,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,  # only use for dense_smaple_smplx
        gs_deform_scale=0.005,
        render_features=False,
        flat_hand_mean=False,
    ):
        """
        Initializes the GS3DRenderer, a subclass of BaseGSRender for 3D Gaussian Splatting rendering.

        Args:
            human_model_path (str): Path to human model files.
            subdivide_num (int): Subdivision number for base mesh.
            smpl_type (str): Type of SMPL/SMPL-X/other model to use.
            feat_dim (int): Dimension of feature embeddings.
            query_dim (int): Dimension of query points/features.
            use_rgb (bool): Whether to use RGB channels.
            sh_degree (int): Spherical harmonics degree for appearance.
            xyz_offset_max_step (float): Max offset per step for position.
            mlp_network_config (dict or None): MLP configuration for feature mapping.
            expr_param_dim (int): Expression parameter dimension.
            shape_param_dim (int): Shape parameter dimension.
            clip_scaling (float, optional): Output scaling for decoder. Default 0.2.
            cano_pose_type (int, optional): Canonical pose type. Default 0.
            decoder_mlp (bool, optional): Use MLP in decoder cross-attention. Default False.
            skip_decoder (bool, optional): Whether to skip decoder and cross-attn layers. Default False.
            fix_opacity (bool, optional): Fix opacity during training. Default False.
            fix_rotation (bool, optional): Fix rotation during training. Default False.
            decode_with_extra_info (dict or None, optional): Provide extra info to decoder. Default None.
            gradient_checkpointing (bool, optional): Enable gradient checkpointing. Default False.
            apply_pose_blendshape (bool, optional): Apply pose blendshape. Default False.
            dense_sample_pts (int, optional): Dense sample points for mesh/voxel. Default 40000.
            gs_deform_scale (float, optional): Deformation scale for Gaussian Splatting. Default 0.005.
            render_features (bool, optional): Output additional features in renderer. Default False.
        """

        super(GS3DRenderer, self).__init__(
            human_model_path,
            subdivide_num,
            smpl_type,
            feat_dim,
            query_dim,
            use_rgb,
            sh_degree,
            xyz_offset_max_step,
            mlp_network_config,
            expr_param_dim,
            shape_param_dim,
            clip_scaling,
            cano_pose_type,
            decoder_mlp,
            skip_decoder,
            fix_opacity,
            fix_rotation,
            decode_with_extra_info,
            gradient_checkpointing,
            apply_pose_blendshape,
            dense_sample_pts,  # only use for dense_smaple_smplx
            gs_deform_scale,
            render_features,
            flat_hand_mean,
        )

        self.gs_net = GSMLPDecoder(
            in_channels=query_dim,
            use_rgb=use_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            init_scaling=-6.0,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            use_fine_feat=(
                True
                if decode_with_extra_info is not None
                and decode_with_extra_info["type"] is not None
                else False
            ),
        )

        self.gs_deform_scale = gs_deform_scale  # deform mask scale.

    def hyper_step(self, step):
        """using to adjust the constrain scale."""

        self.gs_net.hyper_step(step)

    def forward_gs_attr(
        self, x, query_points, smplx_data, debug=False, x_fine=None, mesh_meta=None
    ):
        """
        x: [N, C] Float[Tensor, "Np Cp"],
        query_points: [N, 3] Float[Tensor, "Np 3"]
        """
        device = x.device
        if self.mlp_network_config is not None:
            # x is processed by LayerNorm
            x = self.mlp_net(x)
            if x_fine is not None:
                x_fine = self.mlp_net(x_fine)

        # NOTE that gs_attr contains offset xyz

        is_constrain_body = mesh_meta["is_constrain_body"].to(
            self.smplx_model.is_constrain_body
        )
        is_hands = (mesh_meta["is_rhand"] + mesh_meta["is_lhand"]).to(
            self.smplx_model.is_rhand
        )
        is_upper_body = mesh_meta["is_upper_body"].to(self.smplx_model.is_upper_body)
        # is_constrain_body = self.smplx_model.is_constrain_body
        # is_hands =  self.smplx_model.is_rhand + self.smplx_model.is_lhand
        # is_upper_body = self.smplx_model.is_upper_body

        constrain_dict = dict(
            is_constrain_body=is_constrain_body,
            is_hands=is_hands,
            is_upper_body=is_upper_body,
        )

        gs_attr: GaussianAppOutput = self.gs_net(
            x, query_points, x_fine, constrain_dict
        )

        return gs_attr


def test():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    smplx_data_root = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/smplx_params_smoothed"
    shape_param_file = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/shape_param.json"

    from core.models.rendering.smpl_x import read_smplx_param

    batch_size = 1
    device = "cuda"
    smplx_data, cam_param_list, ori_image_list = read_smplx_param(
        smplx_data_root=smplx_data_root, shape_param_file=shape_param_file, batch_size=2
    )
    smplx_data_tmp = smplx_data
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.unsqueeze(0)
        if (k == "betas") or (k == "face_offset") or (k == "joint_offset"):
            smplx_data_tmp[k] = v[0].unsqueeze(0)
    smplx_data = smplx_data_tmp

    gs_render = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=64,
        query_dim=64,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
    )

    gs_render.to(device)
    # print(cam_param_list[0])

    c2w_list = []
    intr_list = []
    for cam_param in cam_param_list:
        c2w = torch.eye(4).to(device)
        c2w[:3, :3] = cam_param["R"]
        c2w[:3, 3] = cam_param["t"]
        c2w_list.append(c2w)
        intr = torch.eye(4).to(device)
        intr[0, 0] = cam_param["focal"][0]
        intr[1, 1] = cam_param["focal"][1]
        intr[0, 2] = cam_param["princpt"][0]
        intr[1, 2] = cam_param["princpt"][1]
        intr_list.append(intr)

    c2w = torch.stack(c2w_list).unsqueeze(0)
    intrinsic = torch.stack(intr_list).unsqueeze(0)

    out = gs_render.forward(
        gs_hidden_features=torch.zeros((batch_size, 2048, 64)).float().to(device),
        query_points=None,
        smplx_data=smplx_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=int(cam_param_list[0]["princpt"][1]) * 2,
        width=int(cam_param_list[0]["princpt"][0]) * 2,
        background_color=torch.tensor([1.0, 1.0, 1.0])
        .float()
        .view(1, 1, 3)
        .repeat(batch_size, 2, 1)
        .to(device),
        debug=False,
    )

    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(f"./debug_vis/{b_idx}_{v_idx}.ply")
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join("./debug_vis", f"{b_idx}_{v_idx}_{k}.jpg")
                cv2.imwrite(
                    save_path,
                    (v[b_idx, v_idx].detach().cpu().numpy() * 255).astype(np.uint8),
                )
