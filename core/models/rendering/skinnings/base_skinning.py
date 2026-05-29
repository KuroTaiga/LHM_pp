# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author: Lingteng Qiu
# @Email: 220019047@link.cuhk.edu.cn
# @Time: 2025-06-05 21:42:24, Version 0.0, SMPLX + FLAME2019
# @Function: base skinning functions with pytorch3D

"""
Base Skinning Module for SMPL-X and FLAME Integration.

This module provides the foundational skinning functionality for human body modeling
using SMPL-X (Skinned Multi-Person Linear Model eXtended) and FLAME (Faces Learned
with an Articulated Model and Expressions). It supports comprehensive human body
representation including body pose, facial expressions, hand poses, and mesh
subdivision for high-resolution rendering.

The module integrates both SMPL-X and FLAME2019 models to provide a unified
framework for articulated human body modeling with detailed facial expressions
and hand articulation capabilities.
"""

import math
import os.path as osp
import pickle
import sys

sys.path.append("./")

import numpy as np
import torch
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

from core.models.rendering.numpy_compat import ensure_numpy_legacy_aliases
from core.models.rendering.skinnings.constant import SMPLX_JOINTS
from core.models.rendering.smplx import smplx

ensure_numpy_legacy_aliases()


class BaseSkinning:
    """
    Base Skinning class for SMPL-X and FLAME integration in human body modeling.

    This class provides comprehensive skinning functionality for articulated human body
    modeling using SMPL-X and FLAME models. It supports mesh subdivision, joint
    articulation, facial expressions, and body part segmentation for high-quality
    human avatar reconstruction and rendering.

    The class integrates both SMPL-X (for body and hand articulation) and FLAME2019
    (for detailed facial expressions) to provide a unified framework for human body
    modeling with fine-grained control over pose, shape, and expression parameters.

    Attributes:
        human_model_path (str): Path to the SMPL-X/FLAME model files.
        shape_param_dim (int): Dimension of shape parameters (beta coefficients).
        expr_param_dim (int): Dimension of facial expression parameters.
        subdivide_num (int): Number of mesh subdivision iterations for upsampling.
        cano_pose_type (int): Type of canonical pose configuration.
        layer (dict): Dictionary of SMPL-X layers for different genders.
        joint_num (int): Total number of joints in the model (55: body+face+hands).
        vertex_num (int): Number of vertices in the base mesh (10475).
        vertex_num_upsampled (int): Number of vertices after subdivision.
    """

    def __init__(
        self,
        human_model_path,
        shape_param_dim=100,
        expr_param_dim=50,
        subdivide_num=2,
        cano_pose_type=0,
        flat_hand_mean=False,
    ):
        """
        Initialize the BaseSkinning model with SMPL-X and FLAME integration.

        Args:
            human_model_path (str): Path to the directory containing SMPL-X and FLAME model files.
            shape_param_dim (int, optional): Dimension of shape parameters (beta coefficients).
                Defaults to 100.
            expr_param_dim (int, optional): Dimension of facial expression parameters.
                Defaults to 50.
            subdivide_num (int, optional): Number of mesh subdivision iterations for upsampling.
                Defaults to 2.
            cano_pose_type (int, optional): Type of canonical pose configuration.
                0 for exavatar-cano-pose, 1 for alternative configuration. Defaults to 0.

        Note:
            The initialization process sets up SMPL-X layers for different genders,
            loads vertex indices and joint information, initializes neutral poses,
            sets up mesh subdividers, and registers constraint priors for canonical
            space regularization.
        """
        self.human_model_path = human_model_path
        self.shape_param_dim = shape_param_dim
        self.expr_param_dim = expr_param_dim
        self.subdivide_num = subdivide_num
        self.cano_pose_type = cano_pose_type
        if isinstance(flat_hand_mean, str):
            flat_hand_mean = flat_hand_mean.strip().lower() in {"1", "true", "yes", "on"}
        else:
            flat_hand_mean = bool(flat_hand_mean)
        self.flat_hand_mean = flat_hand_mean

        # Initialize all components
        self._init_smplx_layers()
        self._load_vertex_indices()
        self._init_joint_info()
        self._init_neutral_poses()
        self._init_subdividers()
        self._init_constrain()

    def _init_constrain(self):
        """
        Initialize constraint mechanisms for canonical space regularization.

        Sets up body-head mapping and registers constraint priors to ensure
        stable canonical space representation during video-based training.
        """
        self.body_head_mapping = self._get_body_face_mapping()
        self.register_constrain_prior()

    def register_constrain_prior(self):
        """
        Register human body prior constraints for canonical space regularization.

        Since video data cannot provide sufficient supervision for the canonical space,
        this method adds human body priors to constrain rotations and ensure stable
        canonical representation. This constraint mechanism is particularly effective
        for maintaining realistic body poses during training.

        Note:
            The constraint masks are loaded from pre-computed human prior data
            and applied to specific vertex indices to regularize the canonical space.
        """
        constrain_body = np.load(
            "./pretrained_models/voxel_grid/human_prior_constrain.npz"
        )["masks"]

        self.constrain_body_vertex_idx = np.where(constrain_body > 0)[0]

    def _init_smplx_layers(self):
        """
        Initialize SMPL-X layers with appropriate parameters for different genders.

        Creates SMPL-X model layers for neutral, male, and female genders with
        configurable shape and expression parameters. The method also integrates
        FLAME expression capabilities when appropriate parameter dimensions are used.

        Note:
            The layers are configured to not create default pose parameters, allowing
            for explicit control over pose, expression, and shape parameters during
            runtime. FLAME integration is enabled when using full parameter dimensions.
        """
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

        use_face_contour = not (
            self.shape_param_dim == 10 and self.expr_param_dim == 10
        )

        self.layer = {
            gender: smplx.create(
                self.human_model_path,
                "smplx",
                gender=gender,
                num_betas=self.shape_param_dim,
                num_expression_coeffs=self.expr_param_dim,
                use_pca=False,
                use_face_contour=use_face_contour,
                flat_hand_mean=self.flat_hand_mean,
                **layer_args
            )
            for gender in ["neutral", "male", "female"]
        }

        self.face_vertex_idx = np.load(
            osp.join(self.human_model_path, "smplx", "SMPL-X__FLAME_vertex_ids.npy")
        )

        if use_face_contour:
            print("Using FLAME expression")
            self.layer = {
                gender: self._get_expr_from_flame(self.layer[gender])
                for gender in ["neutral", "male", "female"]
            }
        else:
            print("Using basic SMPLX without FLAME expression")

    def _init_neutral_poses(self):
        """
        Initialize neutral pose configurations for canonical space representation.

        Sets up default neutral poses for body joints and jaw pose based on the
        specified canonical pose type. Different canonical pose configurations
        are used for different reconstruction scenarios.

        Note:
            - cano_pose_type=0: Uses exavatar-cano-pose configuration with specific
              shoulder rotations for better canonical space representation
            - cano_pose_type=1: Uses alternative configuration with smaller rotation angles
        """
        body_joints = len(self.joint_part["body"]) - 1
        self.neutral_body_pose = torch.zeros((body_joints, 3))

        if self.cano_pose_type == 0:  # exavatar-cano-pose
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        else:
            angle = math.pi / 9
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, angle])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -angle])

        self.neutral_jaw_pose = torch.FloatTensor([1 / 3, 0, 0])

    def _mesh_based_neutral_pose(self):
        """
        Initialize mesh-based neutral pose configuration.

        Alternative neutral pose setup that focuses on specific joint rotations
        for mesh-based reconstruction scenarios. This configuration uses different
        joint indices and rotation angles compared to the standard neutral pose.

        Note:
            This method is used for specialized mesh-based reconstruction pipelines
            that require different canonical pose configurations.
        """
        body_joints = len(self.joint_part["body"]) - 1
        self.neutral_body_pose = torch.zeros((body_joints, 3))

        angle = math.pi / 6
        self.neutral_body_pose[15] = torch.FloatTensor([0, 0, -angle])
        self.neutral_body_pose[16] = torch.FloatTensor([0, 0, angle])
        self.neutral_jaw_pose = torch.FloatTensor([0, 0, 0])

    def _init_subdividers(self):
        """
        Initialize mesh subdividers for high-resolution mesh upsampling.

        Sets up PyTorch3D mesh subdivision pipeline to create higher resolution
        meshes from the base SMPL-X mesh. The subdivision process increases the
        number of vertices and faces for better geometric detail representation.

        Note:
            The subdivision process is applied iteratively based on subdivide_num,
            with each iteration approximately quadrupling the number of faces.
        """
        vert = self.layer["neutral"].v_template.float()
        face = torch.LongTensor(self.face)
        mesh = Meshes(vert[None, :, :], face[None, :, :])

        if self.subdivide_num > 0:
            self.subdivider_list = [SubdivideMeshes(mesh)]
            for _ in range(self.subdivide_num - 1):
                mesh = self.subdivider_list[-1](mesh)
                self.subdivider_list.append(SubdivideMeshes(mesh))

            self.face_upsampled = (
                self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
            )
        else:
            self.subdivider_list = [mesh]
            self.face_upsampled = self.face

        self.vertex_num_upsampled = int(np.max(self.face_upsampled) + 1)

    def _init_joint_info(self):
        """
        Initialize joint information and part mappings for SMPL-X model.

        Sets up the joint structure for SMPL-X model including 55 total joints:
        22 body joints, 3 face joints, and 30 hand joints (15 per hand).
        Also initializes the joint part mappings for different body regions.
        """
        self.joint_num = 55  # Body (22) + Face (3) + Hands (30)
        self.joints_name = SMPLX_JOINTS
        self.root_joint_idx = self.joints_name.index("Pelvis")
        self._init_joint_part_mappings()

    def _init_joint_part_mappings(self):
        """
        Initialize mappings between joints and body parts for segmentation.

        Creates comprehensive mappings between joint indices and different body
        parts including body, face, hands, lower body, and upper body regions.
        This enables targeted control and analysis of specific body regions.
        """
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

        self.lower_body_vertex_idx = self._get_lower_body_vertices("lower_body")
        self.upper_body_vertex_idx = self._get_lower_body_vertices("upper_body")

    def _get_lower_body_vertices(self, body_type="lower_body"):
        """
        Identify vertices belonging to specific body parts using skinning weights.

        Uses Linear Blend Skinning (LBS) weights to determine which vertices
        are primarily influenced by joints in the specified body part. This
        enables accurate body part segmentation for targeted control.

        Args:
            body_type (str): Type of body part ('lower_body' or 'upper_body').

        Returns:
            np.ndarray: Array of vertex indices belonging to the specified body part.
        """
        lower_body_skinning_index = set(self.joint_part[body_type])
        skinning_weight = self.layer["neutral"].lbs_weights.float()
        skinning_part = skinning_weight.argmax(1).cpu().numpy()

        return np.array(
            [
                v_id
                for v_id, v_s in enumerate(skinning_part)
                if v_s in lower_body_skinning_index
            ]
        )

    def _get_expr_from_flame(self, smplx_layer):
        """
        Integrate FLAME expression parameters into SMPL-X layer.

        Loads FLAME model and transfers expression directions to the corresponding
        face vertices in the SMPL-X model. This enables detailed facial expression
        control using FLAME's expression parameters.

        Args:
            smplx_layer: SMPL-X model layer to integrate with FLAME expressions.

        Returns:
            SMPL-X layer with integrated FLAME expression capabilities.
        """
        flame_layer = smplx.create(
            self.human_model_path,
            "flame",
            gender="neutral",
            num_betas=self.shape_param_dim,
            num_expression_coeffs=self.expr_param_dim,
        )
        smplx_layer.expr_dirs[self.face_vertex_idx, :, :] = flame_layer.expr_dirs
        return smplx_layer

    def _load_vertex_indices(self):
        """
        Load vertex indices for different body parts and facial regions.

        Loads and processes vertex indices for face, hands, and expression regions
        from the SMPL-X model. This includes adding cavity faces for lip region
        and loading MANO hand vertex mappings.
        """
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

    def _add_cavity(self):
        """
        Add cavity faces for lip region to improve mouth modeling.

        Creates additional triangular faces in the lip region to provide better
        geometric representation of the mouth cavity for improved expression modeling.

        Returns:
            tuple: (is_cavity, face_new) where is_cavity is a binary mask for cavity
                   vertices and face_new includes the original faces plus cavity faces.
        """
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
        """
        Retrieve vertex indices influenced by facial expression parameters.

        SMPL-X + FLAME2019 Version: Identifies vertices that are influenced by
        expression parameters using FLAME2019 model's shape directions and
        LBS weights. Excludes vertices primarily influenced by neck and eye joints
        to focus on facial expression regions.

        Returns:
            np.ndarray: Array of vertex indices that are influenced by expression parameters.
        """
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

    def _get_body_face_mapping(self):
        """
        Create mapping between body and face regions for segmentation.

        Analyzes the mesh structure to separate head and body components based on
        face vertex indices. Faces are classified as head faces (all three vertices
        are face vertices) or body faces (no face vertices).

        Returns:
            dict: Dictionary containing head and body mappings with:
                - 'head': Dictionary with head faces and face vertex indices
                - 'body': Dictionary with body faces and non-face vertex indices
        """
        face_vertex_set = set(self.face_vertex_idx)
        all_vertices = set(range(self.vertex_num))

        # Classify face labels
        face_labels = np.isin(self.face.reshape(-1), self.face_vertex_idx)
        face_labels = face_labels.reshape(-1, 3).sum(axis=1)

        # Get head and body components
        head_mask = face_labels == 3
        body_mask = face_labels == 0

        ret_dict = {
            "head": {
                "face": self.face[head_mask],
                "vert": np.array(self.face_vertex_idx),
            },
            "body": {
                "face": self.face[body_mask],
                "vert": np.array(list(all_vertices - face_vertex_set)),
            },
        }

        return ret_dict


if __name__ == "__main__":
    human_model_path = "./pretrained_models/human_model_files"
    gender = "neutral"
    subdivide_num = 1
    smplx_model = BaseSkinning(
        human_model_path,
        shape_param_dim=10,
        expr_param_dim=10,
        subdivide_num=subdivide_num,
        cano_pose_type=1,
    )
