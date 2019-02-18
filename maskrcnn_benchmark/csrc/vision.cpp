// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "torch/extension.h"

#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"

static auto registry = torch::jit::RegisterOperators()
  .op("roi_ops::nms", &nms)
  .op("roi_ops::roi_align_forward(Tensor input, Tensor rois, float spatial_scale,"
      "int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
      &ROIAlign_forward);
     
#ifndef NO_PYTHON
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
}
#endif
