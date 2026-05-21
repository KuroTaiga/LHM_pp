# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : HuggingFace hub model wrapper

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def wrap_model_hub(model_cls: nn.Module):
    class HfModel(model_cls, PyTorchModelHubMixin):
        def __init__(self, config: dict | None = None, **kwargs):
            merged_config = dict(config or {})
            merged_config.update(kwargs)
            super().__init__(**merged_config)
            self.config = merged_config

    return HfModel
