# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
# roi_ops only imported to load library
from . import roi_ops # noqa 

nms = torch.ops.roi_ops.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
