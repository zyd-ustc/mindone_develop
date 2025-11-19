# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the MindSpore SAM2 Video model."""

import numpy as np
import pytest
import torch
from transformers import (
    Sam2VideoConfig,
    Sam2VideoMaskDecoderConfig,
    Sam2VideoPromptEncoderConfig,
    Sam2VisionConfig,
    Sam2HieraDetConfig,
)
from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession as PtSam2VideoInferenceSession

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy
from mindone.transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class Sam2VideoPromptEncoderTester:
    def __init__(
        self,
        hidden_size=32,
        input_image_size=128,
        patch_size=16,
        mask_input_channels=8,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        scale=1,
    ):
        self.hidden_size = hidden_size
        self.input_image_size = input_image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale

    def get_config(self):
        return Sam2VideoPromptEncoderConfig(
            image_size=self.input_image_size,
            patch_size=self.patch_size,
            mask_input_channels=self.mask_input_channels,
            hidden_size=self.hidden_size,
            num_point_embeddings=self.num_point_embeddings,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            scale=self.scale,
        )

    def prepare_config_and_inputs(self):
        dummy_points = floats_numpy([self.batch_size, 3, 2])
        config = self.get_config()

        return config, dummy_points


class Sam2VideoMaskDecoderTester:
    def __init__(
        self,
        hidden_size=32,
        hidden_act="gelu",
        mlp_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=32,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def get_config(self):
        return Sam2VideoMaskDecoderConfig(
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            mlp_dim=self.mlp_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_downsample_rate=self.attention_downsample_rate,
            num_multimask_outputs=self.num_multimask_outputs,
            iou_head_depth=self.iou_head_depth,
            iou_head_hidden_dim=self.iou_head_hidden_dim,
            dynamic_multimask_via_stability=self.dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=self.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=self.dynamic_multimask_stability_thresh,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        dummy_inputs = {
            "image_embedding": floats_numpy([self.batch_size, self.hidden_size]),
        }

        return config, dummy_inputs


class Sam2VideoModelTester:
    def __init__(
        self,
        num_channels=3,
        image_size=128,
        hidden_size=12,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        blocks_per_stage=[1, 2, 7, 2],
        embed_dim_per_stage=[12, 24, 48, 96],
        backbone_channel_list=[96, 48, 24, 12],
        backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]],
        fpn_hidden_size=32,
        memory_encoder_hidden_size=32,
        batch_size=2,
        is_training=False,
    ):
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.blocks_per_stage = blocks_per_stage
        self.embed_dim_per_stage = embed_dim_per_stage
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.memory_encoder_hidden_size = memory_encoder_hidden_size

        self.prompt_encoder_tester = Sam2VideoPromptEncoderTester()
        self.mask_decoder_tester = Sam2VideoMaskDecoderTester()

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        backbone_config = Sam2HieraDetConfig(
            hidden_size=self.hidden_size,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_stride=self.patch_stride,
            patch_kernel_size=self.patch_kernel_size,
            patch_padding=self.patch_padding,
            blocks_per_stage=self.blocks_per_stage,
            embed_dim_per_stage=self.embed_dim_per_stage,
        )
        vision_config = Sam2VisionConfig(
            backbone_config=backbone_config,
            backbone_channel_list=self.backbone_channel_list,
            backbone_feature_sizes=self.backbone_feature_sizes,
            fpn_hidden_size=self.fpn_hidden_size,
        )

        prompt_encoder_config = self.prompt_encoder_tester.get_config()

        mask_decoder_config = self.mask_decoder_tester.get_config()

        return Sam2VideoConfig(
            vision_config=vision_config,
            prompt_encoder_config=prompt_encoder_config,
            mask_decoder_config=mask_decoder_config,
            memory_attention_hidden_size=self.hidden_size,
            memory_encoder_hidden_size=self.memory_encoder_hidden_size,
            image_size=self.image_size,
            mask_downsampler_embed_dim=32,
            memory_fuser_embed_dim=32,
            memory_attention_num_layers=1,
            memory_attention_feed_forward_hidden_size=32,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        
        # Create video tensor: (num_frames, channels, height, width)
        num_frames = 3
        video_np = np.stack([pixel_values[0] for _ in range(num_frames)], axis=0)
        
        # Create point inputs: (batch_size=1, point_batch_size=1, num_points=1, 2)
        # Format should match what _run_single_frame_inference expects
        point_coords = floats_numpy([1, 1, 1, 2]) * self.image_size  # (1, 1, 1, 2)
        point_labels = np.array([[[[1]]]], dtype=np.int32)  # (1, 1, 1) - positive point
        
        frame_idx = 0
        
        # Return raw data, session will be created separately for torch and ms
        inputs_dict = {
            "video_np": video_np,
            "video_height": self.image_size,
            "video_width": self.image_size,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "frame_idx": frame_idx,
        }
        return config, inputs_dict


model_tester = Sam2VideoModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


_CASES = [
    [
        "Sam2VideoModel",
        "transformers.Sam2VideoModel",
        "mindone.transformers.Sam2VideoModel",
        (config,),
        {},
        (),
        inputs_dict,
        {"pred_masks": 0},
    ],
]


def _create_inference_session_for_pt(video_np, video_height, video_width, point_coords, point_labels, frame_idx, dtype):
    """Create PyTorch inference session."""
    # Convert numpy to torch tensor
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    video = torch.from_numpy(video_np).to(torch_dtype)
    
    # Create PyTorch inference session
    inference_session = PtSam2VideoInferenceSession(
        video=video,
        video_height=video_height,
        video_width=video_width,
        dtype=torch_dtype,
    )
    
    # Add object and point inputs for frame 0
    obj_id = 0
    obj_idx = inference_session.obj_id_to_idx(obj_id)
    
    inference_session.add_point_inputs(
        obj_idx=obj_idx,
        frame_idx=frame_idx,
        inputs={
            "point_coords": torch.from_numpy(point_coords).to(torch_dtype),
            "point_labels": torch.from_numpy(point_labels).to(torch.int32),
        }
    )
    inference_session.obj_with_new_inputs.append(obj_id)
    
    return inference_session


def _create_inference_session_for_ms(video_np, video_height, video_width, point_coords, point_labels, frame_idx, dtype):
    """Create MindSpore inference session."""
    # Convert numpy to ms tensor
    if dtype == "fp16":
        ms_dtype = ms.float16
    elif dtype == "bf16":
        ms_dtype = ms.bfloat16
    else:
        ms_dtype = ms.float32
    
    video = ms.Tensor(video_np, dtype=ms_dtype)
    
    # Create MindSpore inference session
    inference_session = Sam2VideoInferenceSession(
        video=video,
        video_height=video_height,
        video_width=video_width,
        dtype=ms_dtype,
    )
    
    # Add object and point inputs for frame 0
    obj_id = 0
    obj_idx = inference_session.obj_id_to_idx(obj_id)
    
    inference_session.add_point_inputs(
        obj_idx=obj_idx,
        frame_idx=frame_idx,
        inputs={
            "point_coords": ms.Tensor(point_coords, dtype=ms_dtype),
            "point_labels": ms.Tensor(point_labels, dtype=ms.int32),
        }
    )
    inference_session.obj_with_new_inputs.append(obj_id)
    
    return inference_session


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in _CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype, mode
):
    ms.set_context(mode=mode)

    (pt_model, ms_model, pt_dtype, ms_dtype) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    
    # Extract session creation data from inputs_kwargs
    session_data = inputs_kwargs.pop("video_np")
    video_height = inputs_kwargs.pop("video_height")
    video_width = inputs_kwargs.pop("video_width")
    point_coords = inputs_kwargs.pop("point_coords")
    point_labels = inputs_kwargs.pop("point_labels")
    frame_idx = inputs_kwargs.pop("frame_idx")
    
    # Create separate inference sessions for torch and ms
    pt_inference_session = _create_inference_session_for_pt(
        session_data, video_height, video_width, point_coords, point_labels, frame_idx, pt_dtype
    )
    ms_inference_session = _create_inference_session_for_ms(
        session_data, video_height, video_width, point_coords, point_labels, frame_idx, ms_dtype
    )
    
    # Parse remaining args
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )
    
    # Add inference sessions to kwargs
    pt_inputs_kwargs["inference_session"] = pt_inference_session
    pt_inputs_kwargs["frame_idx"] = frame_idx
    ms_inputs_kwargs["inference_session"] = ms_inference_session
    ms_inputs_kwargs["frame_idx"] = frame_idx

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )

