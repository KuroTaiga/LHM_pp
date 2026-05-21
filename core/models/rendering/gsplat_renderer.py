# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-14 19:43:20
# @Function      : GSPlat-based Renderer.

try:
    from gsplat.rendering import rasterization

    gsplat_enable = True
except:
    gsplat_enable = False

from collections import defaultdict

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_quaternion

from core.models.rendering.gs_renderer import GS3DRenderer
from core.models.rendering.utils.typing import *
from core.outputs.output import GaussianAppOutput
from core.structures.camera import Camera
from core.structures.gaussian_model import GaussianModel


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y
    return intrs


def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values


class GSPlatRenderer(GS3DRenderer):

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
        Initializes the GSPlatRenderer, an extension of GS3DRenderer for Gaussian Splatting rendering.

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

        if gsplat_enable is False:
            raise ImportError("GSPlat is not installed, please install it first.")
        else:
            super(GSPlatRenderer, self).__init__(
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

    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):
        """
        Extracts and returns the 3D Gaussian properties for rendering from a GaussianModel instance in the context
        of the provided viewpoint camera.

        Args:
            viewpoint_camera (Camera): The viewpoint camera for rendering. (Unused in this stub, but kept for interface compatibility.)
            gaussian_model (GaussianModel): The GaussianModel object containing the properties to extract.

        Returns:
            Tuple:
                xyz (Tensor): The 3D coordinates of the Gaussians.
                shs (Tensor or None): The spherical harmonics coefficients or None.
                colors_precomp (Tensor): The precomputed RGB colors (if use_rgb).
                opacity (Tensor): The opacities of the Gaussians.
                scales (Tensor): The scaling factors per Gaussian.
                rotations (Tensor): Quaternion rotations per Gaussian.
                cov3D_precomp (None): Reserved for covariance data (not used here).
        """

        xyz = gaussian_model.xyz
        opacity = gaussian_model.opacity
        scales = gaussian_model.scaling
        rotations = gaussian_model.rotation
        cov3D_precomp = None
        shs = None
        if gaussian_model.use_rgb:
            colors_precomp = gaussian_model.shs
        else:
            raise NotImplementedError
        return xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp

    def forward_single_view(
        self,
        gaussian_model: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        features=None,
    ):
        """
        Renders a single view using the provided GaussianModel and camera parameters.

        Args:
            gaussian_model (GaussianModel): The 3D Gaussian model to be rendered.
            viewpoint_camera (Camera): The viewpoint camera used for rendering (contains intrinsics, extrinsics, size).
            background_color (Optional[Float[Tensor, "3"]]): Optional background color for the rendered image.
            ret_mask (bool, optional): Whether to return the alpha mask. Default: True.
            features (Optional[Tensor], optional): Optional feature tensor to concatenate with the Gaussian appearance.

        Returns:
            dict:
                {
                    "comp_rgb": Rendered RGB image (H, W, 3),
                    "comp_mask": Rendered alpha mask (H, W),
                    "comp_features": (optional) Rendered additional features (if features is not None)
                }
        """

        xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = (
            self.get_gaussians_properties(viewpoint_camera, gaussian_model)
        )

        intrinsics = viewpoint_camera.intrinsic
        extrinsics = viewpoint_camera.world_view_transform.transpose(
            0, 1
        ).contiguous()  # c2w -> w2c

        img_height = int(viewpoint_camera.height)
        img_width = int(viewpoint_camera.width)

        colors_precomp = colors_precomp.squeeze(1)
        opacity = opacity.squeeze(1)

        if features is not None:
            colors_precomp = torch.cat([colors_precomp, features], dim=1)
            channel = colors_precomp.shape[1]
            background_color = background_color[0].repeat(channel)

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            render_rgbd, render_alphas, meta = rasterization(
                means=xyz.float(),
                quats=rotations.float(),
                scales=scales.float(),
                opacities=opacity.float(),
                colors=colors_precomp.float(),
                viewmats=extrinsics.unsqueeze(0).float(),
                Ks=intrinsics.float().unsqueeze(0)[:, :3, :3],
                width=img_width,
                height=img_height,
                near_plane=viewpoint_camera.znear,
                far_plane=viewpoint_camera.zfar,
                # radius_clip=3.0,
                eps2d=0.3,  # 3 pixel
                render_mode="RGB",
                # render_mode="RGB+D",
                backgrounds=background_color.unsqueeze(0).float(),
                camera_model="pinhole",
            )

        render_rgbd = render_rgbd.squeeze(0)
        render_alphas = render_alphas.squeeze(0)

        rendered_image = render_rgbd[:, :, :3]
        # rendered_depth = render_rgbd[:, :, -1:]

        if features is not None:
            ret = {
                "comp_rgb": rendered_image,  # [H, W, 3]
                "comp_features": render_rgbd[:, :, 3:],
                "comp_rgb_bg": background_color,
                "comp_mask": render_alphas,
                # "comp_depth": rendered_depth,
            }
        else:
            ret = {
                "comp_rgb": rendered_image,  # [H, W, 3]
                "comp_rgb_bg": background_color,
                "comp_mask": render_alphas,
                # "comp_depth": rendered_depth,
            }

        return ret


class GSPlatFeatRenderer(GSPlatRenderer):
    """
    GSPlatFeatRenderer extends GSPlatRenderer to support rendering outputs with additional learned feature channels,
    in addition to RGB images, masks, and optional background handling.

    This class modifies the behavior of `_render_views` and related functions to enable rendering per-pixel features,
    which can be used for downstream tasks like representation learning or multimodal perception.

    Typical usage mirrors GSPlatRenderer but expects `features` to be provided as input for rendering and
    produces feature maps in the output dictionary under the key `"comp_features"`, along with the rendered RGB image
    and mask.

    The rest of the pipeline, including Gaussian attribute animation with SMPL-X or other mesh deformation, is inherited.
    """

    def forward_animate_gs(
        self,
        gs_attr_list: List[GaussianAppOutput],
        query_points: Dict[str, Tensor],
        smplx_data: Dict[str, Tensor],
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        df_data: Optional[Dict] = None,
        patch_size=14,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Animate and render Gaussian Splatting (GS) models with feature support for a batch of frames/views.

        Args:
            gs_attr_list (List[GaussianAppOutput]):
                List of Gaussian attribute outputs, one per batch item. Each element contains predicted Gaussian parameters
                such as offset positions, opacity, rotation, scaling, and appearance for canonical points.
            query_points (Dict[str, Tensor]):
                Dictionary containing query information, must include:
                    - 'neutral_coords': Tensor of canonical coordinate positions, shape [B, N, 3].
                    - 'mesh_meta': (Optional) Dictionary with mesh region meta-info as required by the skinning/posing models.
            smplx_data (Dict[str, Tensor]):
                Dictionary containing per-batch SMPL-X (or similar model) data for the current animation/frame. Used for pose and shape transformation.
            c2w (Float[Tensor, "B Nv 4 4"]):
                Camera-to-world matrices for the views to render (B: batch, Nv: number of views).
            intrinsic (Float[Tensor, "B Nv 4 4"]):
                Intrinsic camera matrices, shape matches c2w.
            height (int):
                Height of output render images (in pixels).
            width (int):
                Width of output render images (in pixels).
            background_color (Optional[Float[Tensor, "B Nv 3"]], default=None):
                Optional RGB background color per batch/view.
            debug (bool, optional):
                If True, enables debug behavior (e.g., simplifies opacities, disables poses, saves debug visualizations).
            df_data (Optional[Dict], default=None):
                Optional dictionary of additional deformation/feature data.
            patch_size (int, optional):
                Size of patches to use for rendering (default 14).
            **kwargs:
                Additional keyword arguments. Can optionally contain 'features' key for render feature maps.

        Returns:
            Dict[str, Tensor]:
                Dictionary of rendered outputs, including:
                - Rendered features (under 'comp_features'), images, masks, etc., batched accordingly.
                - '3dgs': List of all canonical-space GaussianModel instances for the batch.
        """

        batch_size = len(gs_attr_list)
        out_list, cano_out_list = [], []
        query_points_pos = query_points["neutral_coords"]
        mesh_meta = query_points["mesh_meta"]

        gs_list = []
        for b in range(batch_size):
            # Animate GS models
            anim_models, cano_models = self.animate_gs_model(
                gs_attr_list[b],
                query_points_pos[b],
                self._get_single_batch_data(smplx_data, b),
                debug=debug,
                mesh_meta=mesh_meta,
            )
            gs_list.extend(cano_models)
            features = (
                kwargs["features"][b] if kwargs.get("features") is not None else None
            )
            # Render animated views
            out_list.append(
                self._render_views(
                    anim_models[: c2w.shape[1]],  # Only keep requested views
                    c2w[b],
                    intrinsic[b],
                    height,
                    width,
                    background_color[b] if background_color is not None else None,
                    debug,
                    patch_size=patch_size,
                    features=features,
                )
            )

        results = self._combine_outputs(out_list, cano_out_list)
        results["3dgs"] = gs_list

        return results

    def animate_gs_model(
        self,
        gs_attr: GaussianAppOutput,
        query_points,
        smplx_data,
        debug=False,
        mesh_meta=None,
    ):
        """
        Animates the Gaussian Splatting (GS) model by transforming canonical (neutral) points and attributes into the posed space using SMPL-X model deformations.

        Args:
            gs_attr (GaussianAppOutput): Gaussian attribute output for canonical points, including offset positions, opacity, rotation, scaling, and appearance.
            query_points (Tensor): Canonical query point coordinates, shape (N, 3).
            smplx_data (dict): SMPL-X input data for the current animation frame, including body pose, shape, etc.
            debug (bool, optional): If True, use debug mode (e.g., force all opacities to 1.0, use identity rotations). Default: False.
            mesh_meta (dict, optional): Additional mesh region meta-information (e.g., for constraints). Default: None.

        Returns:
            Tuple[List[GaussianModel], List[GaussianModel]]:
                - gs_list: List of posed-space GaussianModel instances (one per camera/view except canonical view).
                - cano_gs_list: List of canonical-space GaussianModel instances (last view is canonical).
        """

        device = gs_attr.offset_xyz.device

        if debug:
            N = gs_attr.offset_xyz.shape[0]
            gs_attr.xyz = torch.zeros_like(gs_attr.offset_xyz)
            gs_attr.opacity = torch.ones((N, 1), device=device)
            gs_attr.rotation = matrix_to_quaternion(
                torch.eye(3, device=device).expand(N, 3, 3)
            )

        # build cano_dependent_pose
        merge_smplx_data = self._prepare_smplx_data(smplx_data)

        posed_points = self._transform_points(
            merge_smplx_data, query_points, gs_attr.offset_xyz, device, mesh_meta
        )
        rotation_pose_verts = self._compute_rotations(
            posed_points["transform_mat_posed"],
            gs_attr.rotation,
            device,
            posed_points["mesh_meta"],
        )

        return self._create_gaussian_models(
            posed_points["posed_coords"],
            gs_attr,
            rotation_pose_verts,
            merge_smplx_data["body_pose"].shape[0],
        )

    def _render_views(
        self,
        gs_list: List[GaussianModel],
        c2w: Tensor,
        intrinsic: Tensor,
        height: int,
        width: int,
        bg_color: Optional[Tensor],
        debug: bool,
        patch_size: int,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Renders multiple views for a list of GaussianModel instances.

        Args:
            gs_list (List[GaussianModel]): List of GaussianModel instances to render for each view.
            c2w (Tensor): Camera-to-world matrices for each view, shape [Nv, 4, 4].
            intrinsic (Tensor): Intrinsic camera matrices for each view, shape [Nv, 4, 4].
            height (int): Output image height (in pixels).
            width (int): Output image width (in pixels).
            bg_color (Optional[Tensor]): Optional background color tensor, shape [Nv, 3] or None.
            debug (bool): If True, enable debug visualizations/saving intermediate images.
            patch_size (int): Patch size used in rendering for downsampling/aggregation.
            **kwargs: Additional keyword arguments, can include 'features' per view if rendering extra feature channels.

        Returns:
            Dict[str, Tensor]:
                Dictionary with rendering outputs for all views.
                Main key 'render' is a list of dictionaries per view,
                each with keys such as 'comp_rgb', 'comp_mask', and, if features are provided, 'comp_features'.
        """
        # obtain device
        self.device = gs_list[0].xyz.device
        results = defaultdict(list)

        for v_idx, gs in enumerate(gs_list):

            if self.render_features:
                render_features = kwargs["features"]
            else:
                render_features = None

            camera = Camera.from_c2w(c2w[v_idx], intrinsic[v_idx], height, width)
            results["render"].append(
                self.forward_single_view(
                    gs,
                    camera,
                    bg_color[v_idx],
                    features=render_features,
                    patch_size=patch_size,
                )
            )

            if debug and v_idx == 0:
                self._debug_save_image(results["render"][-1]["comp_rgb"])

        return results

    def forward_single_view(
        self,
        gaussian_model: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        features=None,
        patch_size=14,
    ):
        """
        Renders a single view for a GaussianModel with optional extra features and positional embedding patching.

        Args:
            gaussian_model (GaussianModel): The 3D Gaussian model to be rendered.
            viewpoint_camera (Camera): The viewpoint camera containing intrinsics, extrinsics, and image size.
            background_color (Optional[Float[Tensor, "3"]]): RGB background color for the rendered image.
            ret_mask (bool, optional): Whether to return the alpha mask in the output. Default: True.
            features (Optional[Tensor], optional): Additional features to concatenate with Gaussian appearance. Default: None.
            patch_size (int, optional): Size of patches for rendering downsampling. Default: 14.

        Returns:
            dict:
                {
                    "comp_rgb": Rendered RGB image (patch_height, patch_width, 3),
                    "comp_mask": Rendered alpha mask (patch_height, patch_width),
                    "comp_features": (optional) Rendered feature image (patch_height, patch_width, channels) if features are provided,
                    ... (other keys as required for downstream use)
                }
        """

        xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = (
            self.get_gaussians_properties(viewpoint_camera, gaussian_model)
        )

        extrinsics = viewpoint_camera.world_view_transform.transpose(
            0, 1
        ).contiguous()  # c2w -> w2c

        # scale_ratio for patch size
        img_height = int(viewpoint_camera.height)
        img_width = int(viewpoint_camera.width)

        patch_height = img_height // patch_size
        patch_width = img_width // patch_size

        scale_y = img_height / patch_height
        scale_x = img_width / patch_width
        intrinsics = scale_intrs(
            viewpoint_camera.intrinsic.clone(), 1.0 / scale_x, 1.0 / scale_y
        )

        colors_precomp = colors_precomp.squeeze(1)
        opacity = opacity.squeeze(1)

        if features is not None:
            colors_precomp = torch.cat([colors_precomp, features], dim=1)
            channel = colors_precomp.shape[1]
            background_color = background_color[0].repeat(channel)

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            render_rgbd, render_alphas, meta = rasterization(
                means=xyz.float(),
                quats=rotations.float(),
                scales=scales.float(),
                opacities=opacity.float(),
                colors=colors_precomp.float(),
                viewmats=extrinsics.unsqueeze(0).float(),
                Ks=intrinsics.float().unsqueeze(0)[:, :3, :3],
                width=patch_width,
                height=patch_height,
                near_plane=viewpoint_camera.znear,
                far_plane=viewpoint_camera.zfar,
                # radius_clip=3.0,
                eps2d=0.3,  # 3 pixel
                render_mode="RGB",
                # render_mode="RGB+D",
                backgrounds=background_color.unsqueeze(0).float(),
                camera_model="pinhole",
            )

        render_rgbd = render_rgbd.squeeze(0)
        render_alphas = render_alphas.squeeze(0)

        rendered_image = render_rgbd[:, :, :3]

        if features is not None:
            ret = {
                "comp_features": render_rgbd[:, :, 3:],
                "comp_mask": render_alphas,
                "comp_rgb": rendered_image,  # [H, W, 3]
                # "comp_depth": rendered_depth,
            }
        else:
            ret = {
                "comp_rgb": rendered_image,  # [H, W, 3]
                "comp_rgb_bg": background_color,
                "comp_mask": render_alphas,
                # "comp_depth": rendered_depth,
            }

        return ret

    def forward(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: dict,
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Forward rendering pass for the GSPlatFeatPosEmbedRenderer.

        Args:
            gs_hidden_features (Float[Tensor, "B Np Cp"]):
                Gaussian hidden features for the batch (B: batch size, Np: number of points, Cp: feature channels).
            query_points (dict):
                Dictionary containing query data, typically including canonical coordinates and mesh metadata.
            smplx_data:
                SMPL-X (or similar parametric model) data; for example, may contain pose, shape, and other parameters.
            c2w (Float[Tensor, "B Nv 4 4"]):
                Camera-to-world transformation matrices for each batch and view.
            intrinsic (Float[Tensor, "B Nv 4 4"]):
                Intrinsic camera matrices, matching batch and view dimensions.
            height (int):
                Height of the output images.
            width (int):
                Width of the output images.
            additional_features (Optional[Float[Tensor, "B C H W"]], optional):
                Additional feature maps to include in rendering, if provided.
            background_color (Optional[Float[Tensor, "B Nv 3"]], optional):
                Optional background RGB colors per batch/view.
            debug (bool, optional):
                If True, enables debug information or simplified behavior.
            **kwargs:
                Additional keyword arguments. Typically includes:
                    - df_data: Deformation/feature data for advanced driving fields.
                    - patch_size: Patch size for chunked rendering.

        Returns:
            Dict[str, Tensor]: Output dictionary containing rendered images (e.g., 'comp_rgb'),
            optional features (e.g., 'comp_features'), masks, Gaussian attributes, and mesh metadata.
        """

        # need shape_params of smplx_data to get querty points and get "transform_mat_neutral_pose"
        # only forward gs params

        gs_attr_list, query_points, smplx_data = self.forward_gs(
            gs_hidden_features,
            query_points,
            smplx_data=smplx_data,
            additional_features=additional_features,
            debug=debug,
        )

        out = self.forward_animate_gs(
            gs_attr_list,
            query_points,
            smplx_data,
            c2w,
            intrinsic,
            height,
            width,
            background_color,
            debug,
            df_data=kwargs["df_data"],
            patch_size=kwargs.get("patch_size", 14),
            features=gs_hidden_features,
        )

        out["gs_attr"] = gs_attr_list
        out["mesh_meta"] = query_points["mesh_meta"]

        return out

    def _combine_outputs(
        self, out_list: List[Dict], cano_out_list: List[Dict]
    ) -> Dict[str, Tensor]:
        """
        Combines the outputs from multiple rendered views (out_list) and canonical outputs (cano_out_list) into a single batched output dictionary.

        Args:
            out_list (List[Dict]):
                A list containing the output dictionaries for each batch/item, where each dictionary holds the rendered outputs for each viewpoint.
            cano_out_list (List[Dict]):
                A list of output dictionaries for the canonical (neutral pose) view per batch item.

        Returns:
            Dict[str, Tensor]:
                A combined dictionary of batched output tensors (organized by key), typically containing:
                    - Rendered RGB images, masks, and features with shape [batch_size, num_views, ...]
                    - Canonical view outputs under distinct keys, if required.
                    - Any other output keys from the rendering pipeline.
        """

        batch_size = len(out_list)

        combined = defaultdict(list)
        for out in out_list:
            # Collect render outputs
            for render_item in out["render"]:
                for k, v in render_item.items():
                    combined[k].append(v)

        # Reshape and permute tensors
        result = {
            k: torch.stack(v).view(batch_size, -1, *v[0].shape).permute(0, 1, 4, 2, 3)
            for k, v in combined.items()
            if torch.stack(v).dim() >= 4
        }

        return result


class GSPlatBackFeatRenderer(GSPlatFeatRenderer):
    """Adding an embedding to model background color"""

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
        Initializes the GSPlatBackFeatRenderer, an extension of GSPlatFeatRenderer adding a learnable embedding
            to model background color.

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
            flat_hand_mean (bool, optional): Whether zero hand pose should use SMPL-X flat hand mean. Default False.

        Adds:
            self.background_embedding (nn.Parameter): Learnable background embedding for enhanced modeling.
        """

        super(GSPlatBackFeatRenderer, self).__init__(
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

        # learnable positional embedding
        self.background_embedding = nn.Parameter(torch.zeros(3, 128))
        # xavier init
        nn.init.xavier_uniform_(self.background_embedding)

    def forward_single_view(
        self,
        gaussian_model: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        features=None,
        patch_size=14,
    ):
        """
        Renders a single view using the provided GaussianModel and camera parameters, with support for patch-based rendering
        and learnable background embeddings for enhanced modeling.

        Args:
            gaussian_model (GaussianModel): The 3D Gaussian model to be rendered.
            viewpoint_camera (Camera): The viewpoint camera used for rendering (contains intrinsics, extrinsics, size).
            background_color (Optional[Float[Tensor, "3"]]): Optional background color for the rendered image.
            ret_mask (bool, optional): Whether to return the alpha mask. Default is True.
            features (Optional[Tensor], optional): Optional feature tensor to concatenate with the Gaussian appearance.
            patch_size (int, optional): Size of the rendering patches (default: 14).

        Returns:
            dict:
                {
                    "comp_rgb": Rendered RGB image (H, W, 3),
                    "comp_features": (optional) Rendered additional features (if features is not None),
                    "comp_rgb_bg": Rendered background color,
                    "comp_mask": Rendered alpha mask (H, W),
                    # "comp_depth": Optionally rendered depth map (if enabled)
                }
        """

        xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = (
            self.get_gaussians_properties(viewpoint_camera, gaussian_model)
        )

        extrinsics = viewpoint_camera.world_view_transform.transpose(
            0, 1
        ).contiguous()  # c2w -> w2c

        # scale_ratio for patch size
        img_height = int(viewpoint_camera.height)
        img_width = int(viewpoint_camera.width)

        patch_height = img_height // patch_size
        patch_width = img_width // patch_size

        scale_y = img_height / patch_height
        scale_x = img_width / patch_width
        intrinsics = scale_intrs(
            viewpoint_camera.intrinsic.clone(), 1.0 / scale_x, 1.0 / scale_y
        )

        colors_precomp = colors_precomp.squeeze(1)
        opacity = opacity.squeeze(1)

        if features is not None:
            colors_precomp = torch.cat([colors_precomp, features], dim=1)
            channel = colors_precomp.shape[1]
            bg_idx = (background_color[0] / 0.5).int().item()
            background_embedding = self.background_embedding[bg_idx]
            background_color = torch.cat(
                [background_color, background_embedding], dim=0
            )

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            render_rgbd, render_alphas, meta = rasterization(
                means=xyz.float(),
                quats=rotations.float(),
                scales=scales.float(),
                opacities=opacity.float(),
                colors=colors_precomp.float(),
                viewmats=extrinsics.unsqueeze(0).float(),
                Ks=intrinsics.float().unsqueeze(0)[:, :3, :3],
                width=patch_width,
                height=patch_height,
                near_plane=viewpoint_camera.znear,
                far_plane=viewpoint_camera.zfar,
                # radius_clip=3.0,
                eps2d=0.3,  # 3 pixel
                render_mode="RGB",
                # render_mode="RGB+D",
                backgrounds=background_color.unsqueeze(0).float(),
                camera_model="pinhole",
            )

        render_rgbd = render_rgbd.squeeze(0)
        render_alphas = render_alphas.squeeze(0)

        rendered_image = render_rgbd[:, :, :3]
        rendered_features = render_rgbd[:, :, 3:]

        if features is not None:
            ret = {
                "comp_features": rendered_features,
                "comp_mask": render_alphas,
                "comp_rgb": rendered_image,  # [H, W, 3]
                # "comp_depth": rendered_depth,
            }
        else:
            ret = {
                "comp_rgb": rendered_image,  # [H, W, 3]
                "comp_rgb_bg": background_color,
                "comp_mask": render_alphas,
                # "comp_depth": rendered_depth,
            }

        return ret
