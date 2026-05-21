# -*- coding: utf-8 -*-#
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-06-08 21:42:24, Version 0.0, SMPLX + FLAME2019 + Voxel-Based Queries.
# @Function      : voxel-based-skinning class
# @Description   : 1.canonical query, 2.offset, 3.blendshape -> 4.posed-view

import os
import sys

sys.path.append("./")

import torch
from pytorch3d.io import save_ply

from core.models.rendering.skinnings.smplx_voxel_skinning import SMPLXVoxelSkinning
from core.structures.voxel_structure import CanoBlendWeightVolume


class SMPLXDiffusedVoxelSkinning(SMPLXVoxelSkinning):
    """
    SMPLXDiffusedVoxelSkinning

    This class extends SMPLXVoxelSkinning to utilize diffused (smoothed) voxel-based skinning weights,
    as proposed by the FITE method. It replaces the base skinning weight query with a volumetric
    lookup (via CanoBlendWeightVolume) over a precomputed diffused skinning field for more robust
    and artifact-free LBS weight assignment.

    Key Features:
        - Loads a diffused voxel-based skinning weight grid from disk.
        - Overrides skinning weight queries to use the diffused voxel weights.
        - Maintains compatibility with all SMPL-X/FLAME mesh and feature querying utilities.

    """

    def __init__(
        self,
        human_model_path,
        gender,
        subdivide_num,
        expr_param_dim=50,
        shape_param_dim=100,
        cano_pose_type=0,
        body_face_ratio=3,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
        flat_hand_mean=False,
        voxel_weights_path="./pretrained_models/voxel_grid/cano_1_volume.npz",
    ):
        """
        Initialize the SMPLXDiffusedVoxelSkinning module.

        This class inherits from SMPLXVoxelSkinning and overrides the skinning weight query
        mechanism with a diffused volumetric field as proposed in FITE. It loads the precomputed
        diffused voxel blend weights and provides a query function compatible with the base interface.

        Args:
            human_model_path (str): Path to the directory containing SMPL-X and FLAME model files.
            gender (str): Gender specification, one of ['male', 'female', 'neutral'].
            subdivide_num (int): Number of mesh subdivision iterations for upsampling mesh detail.
            expr_param_dim (int, optional): Number of facial expression parameters. Defaults to 50.
            shape_param_dim (int, optional): Number of shape (beta) parameters. Defaults to 100.
            cano_pose_type (int, optional): Canonical pose configuration type. Defaults to 0.
            body_face_ratio (int, optional): Ratio of body versus face samples in dense surface sampling. Defaults to 3.
            dense_sample_points (int, optional): Total number of points to sample from body/face surfaces. Defaults to 40000.
            apply_pose_blendshape (bool, optional): Whether to apply pose blendshape corrections. Defaults to False.
            voxel_weights_path (str, optional): Path to the precomputed diffused voxel blendweight field (NPZ file).
                Defaults to "./pretrained_models/voxel_grid/cano_1_volume.npz".
        """

        self.voxel_weights_path = voxel_weights_path

        super(SMPLXDiffusedVoxelSkinning, self).__init__(
            human_model_path,
            gender,
            subdivide_num,
            expr_param_dim,
            shape_param_dim,
            cano_pose_type,
            body_face_ratio,
            dense_sample_points,
            apply_pose_blendshape,
            flat_hand_mean,
        )

    def _voxel_skinning_init(self, *args, **kwargs):
        """register diffused skinning wights proposed by FITE"""

        assert os.path.exists(
            "./pretrained_models/voxel_grid/cano_1_volume.npz"
        ), print("diffused skinning do not exists! download first")
        self.voxel_weights = CanoBlendWeightVolume(self.voxel_weights_path)

        return None, None

    def query_voxel_skinning_weights(self, vs):
        """
        Query diffused voxel-based skinning weights for a given set of 3D points.

        This function overrides the base class implementation and uses a pretrained
        diffused voxel blend weight field (as in FITE) to provide smooth and robust
        skinning weights for arbitrary query points in canonical (neutral) space.

        Args:
            vs (torch.Tensor): Query points of shape [B, N, 3], representing the 3D coordinates
                in canonical (neutral) space.

        Returns:
            torch.Tensor: Queried skinning weights at the provided points,
                suitable for downstream LBS or skinning.
        """

        query_skinning = self.voxel_weights(vs)
        return query_skinning


def test():

    def generate_smplx_point():

        def get_smplx_params(data):
            smplx_params = {}
            smplx_keys = [
                "root_pose",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "lhand_pose",
                "rhand_pose",
                "expr",
                "trans",
                "betas",
            ]
            for k, v in data.items():
                if k in smplx_keys:
                    # print(k, v.shape)
                    smplx_params[k] = data[k].unsqueeze(0).cuda()
            return smplx_params

        def sample_one(data):
            smplx_keys = [
                "root_pose",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "lhand_pose",
                "rhand_pose",
                "trans",
            ]
            for k, v in data.items():
                if k in smplx_keys:
                    # print(k, v.shape)
                    data[k] = data[k][:, 0]
            return data

        human_model_path = "./pretrained_models/human_model_files"
        gender = "neutral"
        subdivide_num = 1

        smplx_model = SMPLXDiffusedVoxelSkinning(
            human_model_path,
            gender,
            shape_param_dim=10,
            expr_param_dim=100,
            subdivide_num=subdivide_num,
            dense_sample_points=160000,
            cano_pose_type=1,
        )

        smplx_model.to("cuda")

        # save_file = f"pretrained_models/human_model_files/smplx_points/smplx_subdivide{subdivide_num}.npy"
        save_file = f"debug/smplx_points/smplx_subdivide{subdivide_num}.npy"
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        smplx_data = {}
        smplx_data["betas"] = torch.zeros((1, 10)).to(device="cuda")
        # mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
        #     smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")
        # )

        # neutral_coords = mesh_neutral_pose,
        # ori_neutral_coords = mesh_neutral_pose_wo_upsample,
        # neutral_normals = mesh_neutral_pose_normal,
        # transform_mat_to_neutral_pose = transform_mat_neutral_pose,

        points = smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")

        debug_pose = torch.load("./debug/pose_example.pth")
        debug_pose["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_data = get_smplx_params(debug_pose)
        smplx_data = sample_one(smplx_data)
        smplx_data["betas"] = torch.ones_like(smplx_data["betas"])

        points = smplx_model.transform_to_posed_verts_from_neutral_pose(
            points,
            smplx_data,
            "cuda",
        )

        posed_coords = points["posed_coords"]
        save_ply("diffused_posed_verts.ply", posed_coords[0])

        return points

    generate_smplx_point()


if __name__ == "__main__":
    test()
