# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Point cloud sampling and EasySMPLX wrapper

import sys

sys.path.append(".")

import copy
import math
import os
import os.path as osp
import pdb
import pickle
import time
import traceback

import ipdb
import numpy as np
import torch
import trimesh
from core.models.rendering.mesh_utils import Mesh
from core.models.rendering.skinnings.constant import SMPLX_JOINTS
from core.models.rendering.smplx import smplx
from pytorch3d.io import save_ply
from tqdm import tqdm


class EasySMPLX:
    """Easy forward smpl-x model"""

    def __init__(
        self,
        human_model_path,
        shape_param_dim=10,  # smpl beta
        expr_param_dim=100,  # flame expression
        gender="neutral",
        flat_hand_mean=False,
    ):
        self.human_model_path = human_model_path
        self.shape_param_dim = shape_param_dim
        self.expr_param_dim = expr_param_dim
        self.gender = gender
        self.flat_hand_mean = bool(flat_hand_mean)

        self._init_smplx_layers()
        self._load_vertex_indices()
        self._init_joint_info()
        self._init_neutral_poses()

        self.smplx_layer = copy.deepcopy(self.layer[self.gender])

    def to(self, device):
        self.smplx_layer.to(device)

    def _add_cavity(self):
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        is_cavity = np.zeros((self.vertex_num), dtype=np.float32)
        is_cavity[lip_vertex_idx] = 1.0

        cavity_face = [[0, 1, 7], [1, 2, 7], [2, 3, 5], [3, 4, 5], [2, 5, 6], [2, 6, 7]]
        face_new = list(self.face_orig)
        for face in cavity_face:
            v1, v2, v3 = face
            face_new.append(
                [lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]]
            )
        face_new = np.array(face_new, dtype=np.int64)

        return is_cavity, face_new

    def _get_expr_vertex_idx(self):
        """SMPLX + FLAME2019 Version: Retrieve related vertices ID according to LBS weights."""
        # Load FLAME 2019 model
        with open(
            osp.join(self.human_model_path, "flame", "2019", "generic_model.pkl"), "rb"
        ) as f:
            flame_2019 = pickle.load(f, encoding="latin1")

        # Identify vertices influenced by expression parameters
        vertex_idxs = np.where(
            (flame_2019["shapedirs"][:, :, 300 : 300 + self.expr_param_dim] != 0).sum(
                (1, 2)
            )
            > 0
        )[
            0
        ]  # FLAME.SHAPE_SPACE_DIM == 300
        flame_joints_name = ("Neck", "Head", "Jaw", "L_Eye", "R_Eye")
        expr_vertex_idx = []
        flame_vertex_num = flame_2019["v_template"].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("Neck")
        ] = 1
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("L_Eye")
        ] = 1
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("R_Eye")
        ] = 1
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)

        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]

        return expr_vertex_idx

    def _init_joint_info(self):
        """Initialize joint information and part mappings"""
        self.joint_num = 55  # Body (22) + Face (3) + Hands (30)
        self.joints_name = SMPLX_JOINTS
        self.root_joint_idx = self.joints_name.index("Pelvis")
        self._init_joint_part_mappings()

    def _load_vertex_indices(self):
        """Load vertex indices for different body parts"""

        self.vertex_num = 10475

        self.face_orig = self.layer["neutral"].faces.astype(np.int64)
        self.is_cavity, self.face = self._add_cavity()
        with open(
            osp.join(self.human_model_path, "smplx", "MANO_SMPLX_vertex_ids.pkl"), "rb"
        ) as f:
            hand_vertex_idx = pickle.load(f, encoding="latin1")

        self.rhand_vertex_idx = hand_vertex_idx["right_hand"]
        self.lhand_vertex_idx = hand_vertex_idx["left_hand"]
        self.expr_vertex_idx = self._get_expr_vertex_idx()

    def _init_joint_part_mappings(self):
        """Initialize mappings between joints and body parts"""
        self.joint_part = {
            "body": range(
                self.joints_name.index("Pelvis"), self.joints_name.index("R_Wrist") + 1
            ),
            "face": range(
                self.joints_name.index("Jaw"), self.joints_name.index("R_Eye") + 1
            ),
            "lhand": range(
                self.joints_name.index("L_Index_1"),
                self.joints_name.index("L_Thumb_3") + 1,
            ),
            "rhand": range(
                self.joints_name.index("R_Index_1"),
                self.joints_name.index("R_Thumb_3") + 1,
            ),
            "lower_body": [
                self.joints_name.index("Pelvis"),
                self.joints_name.index("R_Hip"),
                self.joints_name.index("L_Hip"),
                self.joints_name.index("R_Knee"),
                self.joints_name.index("L_Knee"),
                self.joints_name.index("R_Ankle"),
                self.joints_name.index("L_Ankle"),
                self.joints_name.index("R_Foot"),
                self.joints_name.index("L_Foot"),
            ],
            "upper_body": [
                self.joints_name.index("Spine_1"),
                self.joints_name.index("Spine_2"),
                self.joints_name.index("Spine_3"),
                self.joints_name.index("L_Collar"),
                self.joints_name.index("R_Collar"),
                self.joints_name.index("L_Shoulder"),
                self.joints_name.index("R_Shoulder"),
                self.joints_name.index("L_Elbow"),
                self.joints_name.index("R_Elbow"),
                self.joints_name.index("L_Wrist"),
                self.joints_name.index("R_Wrist"),
            ],
        }

    def _init_smplx_layers(self):
        """Initialize SMPLX layers with appropriate parameters"""
        layer_args = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
            "create_jaw_pose": False,
            "create_leye_pose": False,
            "create_reye_pose": False,
            "create_betas": False,
            "create_expression": False,
            "create_transl": False,
        }

        human_model_path = "./pretrained_models/human_model_files"
        use_face_contour = True

        self.layer = {
            gender: smplx.create(
                human_model_path,
                "smplx",
                gender=gender,
                num_betas=10,
                num_expression_coeffs=100,
                use_pca=False,
                use_face_contour=use_face_contour,
                flat_hand_mean=self.flat_hand_mean,
                **layer_args,
            )
            for gender in ["neutral"]
        }

        self.face_vertex_idx = np.load(
            osp.join(self.human_model_path, "smplx", "SMPL-X__FLAME_vertex_ids.npy")
        )

        if use_face_contour:
            print("Using FLAME expression")
            self.layer = {
                gender: self._get_expr_from_flame(self.layer[gender])
                for gender in ["neutral"]
            }
        else:
            print("Using basic SMPLX without FLAME expression")

    def _get_expr_from_flame(self, smplx_layer):
        """Load expression parameters from FLAME model"""
        flame_layer = smplx.create(
            self.human_model_path,
            "flame",
            gender="neutral",
            num_betas=self.shape_param_dim,
            num_expression_coeffs=self.expr_param_dim,
        )
        smplx_layer.expr_dirs[self.face_vertex_idx, :, :] = flame_layer.expr_dirs
        return smplx_layer

    def _init_neutral_poses(self):
        """Initialize neutral pose configurations"""
        body_joints = len(self.joint_part["body"]) - 1
        self.neutral_body_pose = torch.zeros((body_joints, 3))

        angle = math.pi / 6
        self.neutral_body_pose[15] = torch.FloatTensor([0, 0, -angle])
        self.neutral_body_pose[16] = torch.FloatTensor([0, 0, angle])

    @torch.no_grad()
    def __call__(self, smplx_data, device="cpu"):

        shape_param = smplx_data["betas"]

        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            self.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )  # neutral pose
        zero_hand_pose = (
            torch.zeros((batch_size, len(self.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, self.expr_param_dim)).float().to(device)

        jaw_pose = torch.zeros((batch_size, 3)).float().to(device)

        shape_param = shape_param
        face_offset = None
        joint_offset = None

        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=neutral_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        vertices = output.vertices.squeeze(0)
        faces = self.smplx_layer.faces

        print(vertices.shape)

        mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), faces, process=False)

        return mesh


def basename(path):
    pre_name = os.path.basename(path).split(".")[0]

    return pre_name


def multi_process(worker, items, **kwargs):
    """
    worker worker function to process items
    """

    nodes = kwargs["nodes"]
    dirs = kwargs["dirs"]

    bucket = int(
        np.ceil(len(items) / nodes)
    )  # avoid  last node  process too many items.

    print("Total Nodes:", nodes)
    print("Save Path:", dirs)
    rank = int(os.environ.get("RANK", 0))

    print("Current Rank:", rank)

    kwargs["RANK"] = rank

    if rank == nodes - 1:
        output_dir = worker(items[bucket * rank :], **kwargs)
    else:
        output_dir = worker(items[bucket * rank : bucket * (rank + 1)], **kwargs)

    if rank == 0 and nodes > 1:
        timesleep = kwargs.get("timesleep", 3600)
        time.sleep(timesleep)  # one hour


def run_sampling(items, **params):
    """run pc sampling."""

    output_dir = params["dirs"]
    debug = params["debug"]

    smplx_hand_vertex = params["smplx_hand_vertex"]

    sampling_counts = 80000
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        items = items[:100]

    process_valid = []

    for item in tqdm(items, desc="Processing..."):
        print(item)
        if ".ply" not in item:
            continue
        gs_ply = item
        baseid = basename(gs_ply)

        mesh_root = f"./exps/smplx_output/{baseid}.obj"
        mvs_recon_root = f"./exps/mvs_recon/{baseid}.obj"

        smplx_mesh = trimesh.load_mesh(mesh_root, process=False)
        vertices = smplx_mesh.vertices
        hand_vertices = torch.from_numpy(
            np.asarray(vertices[smplx_hand_vertex])
        ).float()

        mesh = Mesh.load_obj(mvs_recon_root, device="cpu")
        pts = mesh.sample_surface(sampling_counts)
        sampling_pts = torch.cat([pts, hand_vertices], dim=0)

        save_ply_path = os.path.join(output_dir, f"{baseid}_{sampling_counts}.ply")
        save_ply(save_ply_path, sampling_pts)

    return output_dir


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True, help="input path")
    parser.add_argument("-o", "--output", required=True, help="output path")
    parser.add_argument("--nodes", default=1, type=int, help="how many workload?")
    parser.add_argument("--debug", action="store_true", help="debug tag")
    parser.add_argument("--txt", default=None, type=str)
    parser.add_argument(
        "--flat-hand-mean",
        action="store_true",
        default=False,
        help="Use SMPL-X flat_hand_mean=True. Default keeps curved MANO mean hands for zero hand poses.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_parse()

    smplx_model = EasySMPLX(
        "./pretrained_models/human_model_files",
        10,
        100,
        flat_hand_mean=opt.flat_hand_mean,
    )
    rhand_vertex_id = smplx_model.rhand_vertex_idx.tolist()
    lhand_vertex_id = smplx_model.lhand_vertex_idx.tolist()
    smplx_hand_vertex = rhand_vertex_id + lhand_vertex_id

    # catch avaliable items
    if opt.txt == None:
        available_items = os.listdir(opt.input)
        available_items = [os.path.join(opt.input, item) for item in available_items]
    else:
        available_items = []
        with open(opt.txt) as reader:
            for line in reader:
                available_items.append(line.strip())

            available_items = [
                os.path.join(opt.input, item) for item in available_items
            ]

    multi_process(
        worker=run_sampling,
        items=available_items,
        dirs=opt.output,
        nodes=opt.nodes,
        debug=opt.debug,
        smplx_hand_vertex=smplx_hand_vertex,
    )
