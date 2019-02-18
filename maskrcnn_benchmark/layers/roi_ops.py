import torch
if torch.cuda.is_available():
    from . import roi_ops_cuda as _roi_ops
else:
    from . import roi_ops_cpu as _roi_ops

torch.ops.load_library(_roi_ops.__file__)

roi_align_backward = _roi_ops.roi_align_backward

roi_align_forward = torch.ops.roi_ops.roi_align_forward
