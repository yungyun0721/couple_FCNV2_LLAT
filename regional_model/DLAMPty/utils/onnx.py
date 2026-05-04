import numpy as np
import onnxruntime as ort
import torch


def create_session(model_path: str) -> ort.InferenceSession:
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4  # Increase the number for faster inference and more memory consumption
    # options.enable_profiling = True
    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested",
        # "cudnn_conv_algo_search": "EXHAUSTIVE",
        # "cudnn_conv_use_max_workspace": "1",
        # "enable_cuda_graph": "1",
    }
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    return session


def bind_tensors(session: ort.InferenceSession,
                 x_upper: torch.Tensor,
                 x_surface: torch.Tensor,
                 ) -> tuple[ort.IOBinding, torch.Tensor, torch.Tensor]:
    binding = session.io_binding()

    x_upper_tensor = x_upper.contiguous()
    x_surface_tensor = x_surface.contiguous()

    binding.bind_input(
        name="input",
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_upper_tensor.shape),
        buffer_ptr=x_upper_tensor.data_ptr(),
    )
    binding.bind_input(
        name="input_surface",
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_surface_tensor.shape),
        buffer_ptr=x_surface_tensor.data_ptr(),
    )

    y_upper_tensor = torch.empty_like(x_upper, dtype=x_upper.dtype, device="cuda:0").contiguous()
    y_surface_tensor = torch.empty_like(x_surface, dtype=x_surface.dtype, device="cuda:0").contiguous()

    binding.bind_output(
        name="output",
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(y_upper_tensor.shape),
        buffer_ptr=y_upper_tensor.data_ptr(),
    )
    binding.bind_output(
        name="output_surface",
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(y_surface_tensor.shape),
        buffer_ptr=y_surface_tensor.data_ptr(),
    )

    return binding, y_upper_tensor, y_surface_tensor


def rebind_input(binding: ort.IOBinding,
                 x_upper: torch.Tensor,
                 x_surface: torch.Tensor,
                 ) -> None:
    x_upper_tensor = x_upper.contiguous()
    x_surface_tensor = x_surface.contiguous()

    binding.bind_input(
        name="input",
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_upper_tensor.shape),
        buffer_ptr=x_upper_tensor.data_ptr(),
    )
    binding.bind_input(
        name="input_surface",
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_surface_tensor.shape),
        buffer_ptr=x_surface_tensor.data_ptr(),
    )
