# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Distributed SMPL-X estimation pipeline

"""SMPL-X Estimation"""

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

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from core.models.rendering.skinnings.constant import SMPLX_JOINTS
from core.models.rendering.smplx import smplx

DATASETS_CONFIG = {
    "dataset5": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v5_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_val_filter460-self-rotated-69.json",
    ),
    "dataset6": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v6_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv6_test_100-self-rotated-25.json",
    ),
    "synthetic": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/synthetic_data_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_LHM_synthetic_dataset_val_17.json",
    ),
    "dataset5_train": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v5_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_train_filter40K.json",
    ),
    "dataset6_train": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v6_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv6_train_5W.json",
    ),
    "eval_train": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/LHM_video_dataset/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_LHM_dataset_train_filter_16W.json",
    ),
    "eval_test": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/LHM_video_dataset/",
        meta_path="./train_data/ClothVideo/label/valid_LHM_dataset_train_val_100.json",
    ),
}


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
        time_sleep = kwargs.get("timesleep", 7200)
        time.sleep(time_sleep)  # one hour


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
    def __call__(self, smplx_data, device="cuda"):

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


def debug():
    smplx_model = EasySMPLX("./pretrained_models/human_model_files", 10, 100)
    smplx_model.to("cuda")

    smplx_data = {}
    smplx_data["betas"] = torch.zeros((1, 10)).to(device="cuda")

    out_mesh = smplx_model(smplx_data)
    out_mesh.export("example.obj")


def run_smplx_estimation(items, **params):
    output_dir = params["dirs"]
    dataset = params["dataset"]
    smplx_model = params["model"]
    debug = params["debug"]
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"

    if debug:
        items = items[:5]

    process_valid = []
    for item in tqdm(items, desc="Processing..."):

        try:
            item = dataset.__getitem__(item, 1)
            baseid, beta = item["uid"], item["betas"]
            save_obj = os.path.join(output_dir, f"{baseid}.obj")

            # if os.path.exists(save_obj):
            #     continue
            print(f"processing img dir {baseid}")
            smplx_data = dict()
            smplx_data["betas"] = beta.to(device)
            out_mesh = smplx_model(smplx_data)
            out_mesh.export(save_obj)
        except:
            traceback.print_exc()

    return output_dir


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        choices=["dataset5_train", "dataset6_train", "eval_train", "eval_test"],
        help="dataset type",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        default="./exps/smplx_output",
        type=str,
        help="output_dir",
    )
    parser.add_argument("--nodes", default=1, type=int, help="number of workloads")
    parser.add_argument("--debug", action="store_true")
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
    exp_name = opt.input

    # dataset
    dataset_config = DATASETS_CONFIG[exp_name]
    kwargs = {}
    if exp_name == "eval" or exp_name == "eval_train":
        from core.datasets.video_human_lhm_dataset_a4o import (
            VideoHumanLHMA4ODatasetEval as VideoDataset,
        )
    elif "in_the_wild" in exp_name:
        from core.datasets.video_in_the_wild_dataset import (
            VideoInTheWildEval as VideoDataset,
        )

        kwargs["heuristic_sampling"] = True
    else:
        import pdb

        from core.datasets.video_human_dataset_a4o import (
            VideoHumanA4ODatasetEval as VideoDataset,
        )

    dataset = VideoDataset(
        root_dirs=dataset_config["root_dirs"],
        meta_path=dataset_config["meta_path"],
        sample_side_views=7,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        source_image_res=384,
        debug=False,
        use_flame=True,
        ref_img_size=1,
        womask=True,
        is_val=True,
        **kwargs,
    )

    smplx_model = EasySMPLX(
        "./pretrained_models/human_model_files",
        10,
        100,
        flat_hand_mean=opt.flat_hand_mean,
    )
    smplx_model.to("cuda")

    available_items = np.arange(len(dataset))

    multi_process(
        worker=run_smplx_estimation,
        items=available_items,
        dirs=opt.output,
        nodes=opt.nodes,
        debug=opt.debug,
        model=smplx_model,
        dataset=dataset,
    )
