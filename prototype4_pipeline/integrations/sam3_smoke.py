import importlib
import importlib.metadata
import importlib.util
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path


SAM3_EXPECTED_FILES = {
    "readme": "README.md",
    "release_notes": "RELEASE_SAM3p1.md",
    "pyproject": "pyproject.toml",
    "model_builder": "sam3/model_builder.py",
    "package_init": "sam3/__init__.py",
    "image_processor": "sam3/model/sam3_image_processor.py",
    "video_predictor": "sam3/model/sam3_video_predictor.py",
    "multiplex_video_predictor": "sam3/model/sam3_multiplex_video_predictor.py",
    "video_inference": "sam3/model/sam3_video_inference.py",
    "tokenizer_asset": "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    "video_example_notebook": "examples/sam3_video_predictor_example.ipynb",
    "multiplex_example_notebook": "examples/sam3.1_video_predictor_example.ipynb",
}

SAM3_DEPENDENCY_MODULES = {
    "setuptools/pkg_resources": "pkg_resources",
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "ftfy": "ftfy",
    "regex": "regex",
    "iopath": "iopath",
    "typing_extensions": "typing_extensions",
    "huggingface_hub": "huggingface_hub",
    "Pillow": "PIL",
    "opencv-python": "cv2",
    "decord": "decord",
    "einops": "einops",
}

IMPORT_TARGETS = [
    {
        "name": "SAM 3 package",
        "module": "sam3",
        "attributes": [],
    },
    {
        "name": "SAM 3 model builder",
        "module": "sam3.model_builder",
        "attributes": ["build_sam3_image_model", "build_sam3_video_predictor", "build_sam3_multiplex_video_predictor"],
    },
    {
        "name": "SAM 3 image processor",
        "module": "sam3.model.sam3_image_processor",
        "attributes": ["Sam3Processor"],
    },
    {
        "name": "SAM 3 video predictor",
        "module": "sam3.model.sam3_video_predictor",
        "attributes": ["Sam3VideoPredictorMultiGPU"],
    },
]


@contextmanager
def temporary_sys_path(path):
    path = str(Path(path).resolve())
    original = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def file_status(repo):
    files = {}
    missing = []
    for name, relative_path in SAM3_EXPECTED_FILES.items():
        path = repo / relative_path
        present = path.exists()
        files[name] = {"path": str(path), "present": present}
        if not present:
            missing.append(relative_path)
    return files, missing


def package_version(package_name, module_name):
    candidates = [package_name, module_name]
    if package_name == "opencv-python":
        candidates.insert(0, "opencv-python")
    if package_name == "Pillow":
        candidates.insert(0, "Pillow")
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def dependency_status():
    modules = {}
    missing = []
    for package_name, module_name in SAM3_DEPENDENCY_MODULES.items():
        available = importlib.util.find_spec(module_name) is not None
        modules[package_name] = {
            "module": module_name,
            "available": available,
            "version": package_version(package_name, module_name) if available else None,
        }
        if not available:
            missing.append(package_name)
    return modules, missing


def torch_runtime_status(device):
    status = {
        "expected_torch": ">=2.7 per SAM 3 README",
        "expected_cuda": ">=12.6 per SAM 3 README",
        "torch_imported": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "requested_device": device,
        "device_ready": False,
        "warning": None,
    }
    try:
        import torch
        status["torch_imported"] = True
        status["torch_version"] = str(torch.__version__)
        status["torch_cuda_version"] = str(getattr(torch.version, "cuda", None))
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["version_warning"] = None
        try:
            major_minor = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
            if major_minor < (2, 7):
                status["version_warning"] = "SAM 3 README recommends PyTorch 2.7 or higher."
        except Exception:
            status["version_warning"] = "Could not parse torch version."
        cuda_version = getattr(torch.version, "cuda", None)
        status["cuda_version_warning"] = None
        if cuda_version is not None:
            try:
                cuda_major_minor = tuple(int(part) for part in str(cuda_version).split(".")[:2])
                if cuda_major_minor < (12, 6):
                    status["cuda_version_warning"] = "SAM 3 README recommends CUDA 12.6 or higher."
            except Exception:
                status["cuda_version_warning"] = "Could not parse torch CUDA version."
        status["device_ready"] = device == "cpu" or status["cuda_available"]
        if device == "cuda" and not status["cuda_available"]:
            status["warning"] = "CUDA requested but torch.cuda.is_available() is false."
    except Exception as exc:
        status["warning"] = "{}: {}".format(exc.__class__.__name__, exc)
    return status


def check_import_target(target):
    record = {
        "name": target["name"],
        "module": target["module"],
        "attributes": target.get("attributes", []),
        "status": "not_checked",
        "error": None,
    }
    try:
        module = importlib.import_module(target["module"])
        missing_attrs = [attr for attr in target.get("attributes", []) if not hasattr(module, attr)]
        if missing_attrs:
            record["status"] = "missing_attributes"
            record["error"] = "Missing attributes: {}".format(", ".join(missing_attrs))
        else:
            record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = "{}: {}".format(exc.__class__.__name__, exc)
    return record


def run_import_check(repo_path, device="cuda"):
    repo = Path(repo_path)
    expected_files, missing_files = file_status(repo)
    dependencies, missing_dependencies = dependency_status()
    result = {
        "repo_path": str(repo),
        "repo_present": repo.exists(),
        "expected_files": expected_files,
        "missing_expected_files": missing_files,
        "dependency_modules": dependencies,
        "missing_dependencies": missing_dependencies,
        "torch_runtime": torch_runtime_status(device),
        "import_targets": [],
        "did_instantiate_model": False,
        "did_download_weights": False,
        "did_run_inference": False,
        "status": "not_run",
    }
    if not repo.exists():
        result["status"] = "repo_missing"
        return result
    with temporary_sys_path(repo):
        result["import_targets"] = [check_import_target(target) for target in IMPORT_TARGETS]
    failed = [target for target in result["import_targets"] if target["status"] != "passed"]
    if missing_files:
        result["status"] = "repo_incomplete"
    elif failed:
        result["status"] = "imports_failed"
    elif missing_dependencies:
        result["status"] = "dependencies_missing"
    else:
        result["status"] = "imports_passed"
    return result


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sampled_frames(project_root, run_id, frame_count):
    frame_dir = project_root / "outputs" / run_id / "sampled_frames"
    return sorted(frame_dir.glob("frame_*.jpg"))[:frame_count]


def safe_shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return [str(dim) for dim in shape]


def summarize_sam3_outputs(outputs):
    summary = {"type": type(outputs).__name__}
    if isinstance(outputs, dict):
        summary["keys"] = sorted(str(key) for key in outputs.keys())
        for key in ["masks", "boxes", "scores"]:
            if key in outputs:
                summary[key] = {"type": type(outputs[key]).__name__, "shape": safe_shape(outputs[key])}
    return summary


def torch_dtype(torch, dtype_name):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def cast_tensors(value, dtype, _seen=None):
    try:
        import torch
    except Exception:
        return value
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return value
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        _seen.add(value_id)
        for key, item in list(value.items()):
            value[key] = cast_tensors(item, dtype, _seen)
        return value
    if isinstance(value, list):
        _seen.add(value_id)
        for index, item in enumerate(value):
            value[index] = cast_tensors(item, dtype, _seen)
        return value
    if isinstance(value, tuple):
        _seen.add(value_id)
        return tuple(cast_tensors(item, dtype, _seen) for item in value)
    if hasattr(value, "__dict__") and value.__class__.__module__.split(".")[0] in {"sam3"}:
        _seen.add(value_id)
        for key, item in vars(value).items():
            try:
                setattr(value, key, cast_tensors(item, dtype, _seen))
            except Exception:
                continue
    return value


def dtype_label(value):
    if value is None:
        return None
    return str(value).replace("torch.", "")


def tensor_record(name, tensor):
    return {
        "name": name,
        "dtype": dtype_label(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "shape": safe_shape(tensor),
    }


def module_dtype_snapshot(module, limit=20):
    snapshot = {"parameters": [], "buffers": []}
    if module is None:
        return snapshot
    try:
        for index, (name, param) in enumerate(module.named_parameters()):
            if index >= limit:
                break
            snapshot["parameters"].append(tensor_record(name, param))
    except Exception as exc:
        snapshot["parameters_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    try:
        for index, (name, buffer) in enumerate(module.named_buffers()):
            if index >= limit:
                break
            snapshot["buffers"].append(tensor_record(name, buffer))
    except Exception as exc:
        snapshot["buffers_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    return snapshot


def selected_attributes(obj, label):
    needles = ("dtype", "precision", "autocast", "bf16", "fp16")
    records = []
    if obj is None:
        return records
    names = set()
    try:
        names.update(vars(obj).keys())
    except Exception:
        pass
    try:
        names.update(name for name in dir(obj) if any(needle in name.lower() for needle in needles))
    except Exception:
        pass
    for name in sorted(names):
        if not any(needle in name.lower() for needle in needles):
            continue
        try:
            value = getattr(obj, name)
            rendered = repr(value)
            value_type = type(value).__name__
        except Exception as exc:
            rendered = "<{}: {}>".format(exc.__class__.__name__, exc)
            value_type = None
        if len(rendered) > 500:
            rendered = rendered[:497] + "..."
        records.append({"object": label, "name": name, "type": value_type, "repr": rendered})
    return records


def collect_tensor_tree(value, limit=20, prefix="state", _seen=None, records=None):
    try:
        import torch
    except Exception:
        return []
    if records is None:
        records = []
    if _seen is None:
        _seen = set()
    if len(records) >= limit:
        return records
    value_id = id(value)
    if value_id in _seen:
        return records
    if isinstance(value, torch.Tensor):
        records.append(tensor_record(prefix, value))
        return records
    if isinstance(value, dict):
        _seen.add(value_id)
        for key, item in value.items():
            collect_tensor_tree(item, limit, "{}.{}".format(prefix, key), _seen, records)
            if len(records) >= limit:
                break
    elif isinstance(value, (list, tuple)):
        _seen.add(value_id)
        for index, item in enumerate(value):
            collect_tensor_tree(item, limit, "{}[{}]".format(prefix, index), _seen, records)
            if len(records) >= limit:
                break
    elif hasattr(value, "__dict__") and value.__class__.__module__.split(".")[0] in {"sam3"}:
        _seen.add(value_id)
        for key, item in vars(value).items():
            collect_tensor_tree(item, limit, "{}.{}".format(prefix, key), _seen, records)
            if len(records) >= limit:
                break
    return records


def autocast_status(torch):
    status = {}
    try:
        status["autocast_enabled"] = bool(torch.is_autocast_enabled())
    except Exception as exc:
        status["autocast_enabled_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    if hasattr(torch, "is_autocast_cpu_enabled"):
        try:
            status["autocast_cpu_enabled"] = bool(torch.is_autocast_cpu_enabled())
        except Exception as exc:
            status["autocast_cpu_enabled_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    if hasattr(torch, "get_autocast_gpu_dtype"):
        try:
            status["autocast_cuda_dtype"] = dtype_label(torch.get_autocast_gpu_dtype())
        except Exception as exc:
            status["autocast_cuda_dtype_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    elif hasattr(torch, "get_autocast_dtype"):
        try:
            status["autocast_cuda_dtype"] = dtype_label(torch.get_autocast_dtype("cuda"))
        except Exception as exc:
            status["autocast_cuda_dtype_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    if hasattr(torch, "get_autocast_cpu_dtype"):
        try:
            status["autocast_cpu_dtype"] = dtype_label(torch.get_autocast_cpu_dtype())
        except Exception as exc:
            status["autocast_cpu_dtype_error"] = "{}: {}".format(exc.__class__.__name__, exc)
    return status


def dtype_diagnostics(torch, stage, selected_dtype, model=None, processor=None, state=None):
    record = {
        "stage": stage,
        "selected_dtype": dtype_label(selected_dtype),
        "grad_enabled": bool(torch.is_grad_enabled()),
        "autocast": autocast_status(torch),
        "model": module_dtype_snapshot(model, limit=20),
        "attributes": [],
        "state_tensors": collect_tensor_tree(state, limit=20, prefix="state") if state is not None else [],
    }
    record["attributes"].extend(selected_attributes(model, "model"))
    record["attributes"].extend(selected_attributes(processor, "processor"))
    return record


def force_module_dtype(model, dtype_name, dtype):
    if dtype_name == "float32":
        model = model.float()
    elif dtype_name == "float16":
        model = model.half()
    else:
        model = model.to(dtype=dtype)
    for param in model.parameters():
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
    for buffer in model.buffers():
        if buffer.is_floating_point() and buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype=dtype)
    return model


@contextmanager
def smoke_autocast_policy(torch, device, dtype_name):
    """Override SAM 3's internal bf16 autocast only inside the smoke path."""
    if device != "cuda":
        yield
        return

    selected_dtype = torch_dtype(torch, dtype_name)
    original_autocast = torch.autocast
    original_amp_autocast = getattr(getattr(torch, "amp", None), "autocast", None)

    def disabled_autocast(*args, **kwargs):
        kwargs["enabled"] = False
        return original_autocast(*args, **kwargs)

    def disabled_amp_autocast(*args, **kwargs):
        kwargs["enabled"] = False
        return original_amp_autocast(*args, **kwargs)

    def selected_autocast(*args, **kwargs):
        kwargs["dtype"] = selected_dtype
        return original_autocast(*args, **kwargs)

    def selected_amp_autocast(*args, **kwargs):
        kwargs["dtype"] = selected_dtype
        return original_amp_autocast(*args, **kwargs)

    if dtype_name == "float32":
        torch.autocast = disabled_autocast
        if original_amp_autocast is not None:
            torch.amp.autocast = disabled_amp_autocast
        try:
            with original_autocast(device_type="cuda", enabled=False):
                yield
        finally:
            torch.autocast = original_autocast
            if original_amp_autocast is not None:
                torch.amp.autocast = original_amp_autocast
        return

    torch.autocast = selected_autocast
    if original_amp_autocast is not None:
        torch.amp.autocast = selected_amp_autocast
    try:
        with original_autocast(device_type="cuda", dtype=selected_dtype):
            yield
    finally:
        torch.autocast = original_autocast
        if original_amp_autocast is not None:
            torch.amp.autocast = original_amp_autocast



@contextmanager
def smoke_fused_kernel_policy(disable_fused_kernels, selected_dtype, patch_metadata):
    """Bypass SAM 3's bf16 fused MLP helper only inside smoke inference."""
    patch_metadata["disabled"] = bool(disable_fused_kernels)
    patch_metadata["calls"] = []
    if not disable_fused_kernels:
        yield
        return

    import torch
    import sam3.perflib.fused as fused_module
    import sam3.model.vitdet as vitdet_module

    original_fused_addmm_act = fused_module.addmm_act
    original_vitdet_addmm_act = vitdet_module.addmm_act
    max_logged_calls = 20

    patch_metadata["original_addmm_act"] = repr(original_fused_addmm_act)
    patch_metadata["original_vitdet_addmm_act"] = repr(original_vitdet_addmm_act)

    def safe_addmm_act(activation, linear, mat1):
        linear_dtype = getattr(getattr(linear, "weight", None), "dtype", selected_dtype)
        input_dtype = getattr(mat1, "dtype", None)
        weight_dtype = getattr(getattr(linear, "weight", None), "dtype", None)
        bias_dtype = getattr(getattr(linear, "bias", None), "dtype", None)
        x = mat1.to(dtype=linear_dtype) if getattr(mat1, "is_floating_point", lambda: False)() else mat1
        y = linear(x)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            y = torch.nn.functional.relu(y)
        elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            y = torch.nn.functional.gelu(y)
        elif isinstance(activation, type) and issubclass(activation, torch.nn.Module):
            y = activation()(y)
        elif callable(activation):
            y = activation(y)
        else:
            raise ValueError("Unexpected activation {}".format(activation))
        if y.is_floating_point() and y.dtype != linear_dtype:
            y = y.to(dtype=linear_dtype)
        if len(patch_metadata["calls"]) < max_logged_calls:
            patch_metadata["calls"].append(
                {
                    "activation": getattr(activation, "__name__", repr(activation)),
                    "input_dtype": dtype_label(input_dtype),
                    "weight_dtype": dtype_label(weight_dtype),
                    "bias_dtype": dtype_label(bias_dtype),
                    "linear_dtype": dtype_label(linear_dtype),
                    "output_dtype": dtype_label(getattr(y, "dtype", None)),
                    "input_shape": safe_shape(mat1),
                    "output_shape": safe_shape(y),
                    "grad_enabled": bool(torch.is_grad_enabled()),
                }
            )
        return y

    fused_module.addmm_act = safe_addmm_act
    vitdet_module.addmm_act = safe_addmm_act
    patch_metadata["patched_addmm_act"] = repr(safe_addmm_act)
    patch_metadata["patched_vitdet_addmm_act"] = repr(vitdet_module.addmm_act)
    try:
        yield
    finally:
        fused_module.addmm_act = original_fused_addmm_act
        vitdet_module.addmm_act = original_vitdet_addmm_act
        patch_metadata["restored"] = True


def set_image_with_dtype(processor, image, dtype, diagnostics=None, torch_module=None, selected_dtype=None, model=None):
    import PIL
    import torch
    import numpy as np
    from torchvision.transforms import v2

    torch_for_diag = torch_module or torch
    dtype_for_diag = selected_dtype or dtype
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    elif isinstance(image, (torch.Tensor, np.ndarray)):
        height, width = image.shape[-2:]
    else:
        raise ValueError("Image must be a PIL image or a tensor")

    image_tensor = v2.functional.to_image(image).to(processor.device)
    image_tensor = processor.transform(image_tensor).unsqueeze(0).to(dtype=dtype)
    if diagnostics is not None:
        diagnostics.append(
            dtype_diagnostics(
                torch_for_diag,
                "inside_set_image_with_dtype_before_forward_image",
                dtype_for_diag,
                model=model,
                processor=processor,
                state={"image_tensor": image_tensor},
            )
        )
    state = {
        "original_height": height,
        "original_width": width,
        "backbone_out": processor.model.backbone.forward_image(image_tensor),
    }
    if diagnostics is not None:
        diagnostics.append(
            dtype_diagnostics(
                torch_for_diag,
                "inside_set_image_with_dtype_after_forward_image",
                dtype_for_diag,
                model=model,
                processor=processor,
                state=state,
            )
        )
    inst_interactivity_en = processor.model.inst_interactive_predictor is not None
    if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
        sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
        sam2_backbone_out["backbone_fpn"][0] = processor.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
            sam2_backbone_out["backbone_fpn"][0]
        )
        sam2_backbone_out["backbone_fpn"][1] = processor.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
            sam2_backbone_out["backbone_fpn"][1]
        )
    return cast_tensors(state, dtype)


def set_text_prompt_with_dtype(processor, prompt, state, dtype, diagnostics=None, torch_module=None, selected_dtype=None, model=None):
    if "backbone_out" not in state:
        raise ValueError("You must call set_image before set_text_prompt")
    torch_for_diag = torch_module
    dtype_for_diag = selected_dtype or dtype
    if diagnostics is not None and torch_for_diag is not None:
        diagnostics.append(
            dtype_diagnostics(
                torch_for_diag,
                "inside_set_text_prompt_before_forward_text",
                dtype_for_diag,
                model=model,
                processor=processor,
                state=state,
            )
        )
    text_outputs = processor.model.backbone.forward_text([prompt], device=processor.device)
    state["backbone_out"].update(cast_tensors(text_outputs, dtype))
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
    state = cast_tensors(state, dtype)
    if diagnostics is not None and torch_for_diag is not None:
        diagnostics.append(
            dtype_diagnostics(
                torch_for_diag,
                "inside_set_text_prompt_before_forward_grounding",
                dtype_for_diag,
                model=model,
                processor=processor,
                state=state,
            )
        )
    return cast_tensors(processor._forward_grounding(state), dtype)


def save_mask_overlay(image, masks, output_dir, frame_index):
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return []
    if masks is None or not hasattr(masks, "detach"):
        return []
    mask_array = masks.detach().to("cpu")
    if len(mask_array.shape) == 4 and int(mask_array.shape[1]) == 1:
        mask_array = mask_array[:, 0]
    if len(mask_array.shape) == 2:
        combined = mask_array.bool().numpy()
    elif len(mask_array.shape) >= 3 and int(mask_array.shape[0]) > 0:
        combined = mask_array.bool().any(dim=0).numpy()
    else:
        combined = np.zeros((image.height, image.width), dtype=bool)
    if combined.shape != (image.height, image.width):
        combined_img = Image.fromarray((combined.astype(np.uint8) * 255), mode="L").resize(
            (image.width, image.height), resample=Image.Resampling.NEAREST
        )
        combined = np.array(combined_img) > 0
    base = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = 40
    overlay[..., 2] = 40
    overlay[..., 3] = combined.astype(np.uint8) * 115
    blended = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "frame_{:03d}_mask_overlay.png".format(frame_index)
    mask_path = output_dir / "frame_{:03d}_combined_mask.png".format(frame_index)
    blended.convert("RGB").save(overlay_path)
    Image.fromarray((combined.astype(np.uint8) * 255), mode="L").save(mask_path)
    return [str(overlay_path), str(mask_path)]


def run_guarded_inference(repo_path, frames, output_dir, device, allow_download_weights, prompt, dtype_name="float32", disable_fused_kernels=True):
    result = {
        "status": "not_run",
        "reason": None,
        "allow_download_weights_supplied": bool(allow_download_weights),
        "device": device,
        "selected_dtype": dtype_name,
        "prompt": prompt,
        "frame_count": len(frames),
        "frames": [str(path) for path in frames],
        "mask_counts_by_frame": [],
        "preview_artifacts": [],
        "output_summary": None,
        "dtype_diagnostics": [],
        "fused_kernel_patch": {"disabled": bool(disable_fused_kernels)},
        "error": None,
    }
    if not allow_download_weights:
        result["status"] = "skipped"
        result["reason"] = "Inference requires --allow-download-weights because SAM 3 builders may download checkpoints."
        return result
    if not frames:
        result["status"] = "failed"
        result["error"] = {"type": "ValueError", "message": "No sampled frames were found.", "traceback": None}
        return result

    repo = Path(repo_path)
    current_stage = "startup"
    torch = None
    model = None
    processor = None
    state = None
    selected_dtype = None
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        selected_dtype = torch_dtype(torch, dtype_name)
        from PIL import Image
        with temporary_sys_path(repo):
            current_stage = "import_sam3_builder"
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            with torch.inference_mode(), smoke_autocast_policy(torch, device, dtype_name), smoke_fused_kernel_policy(
                disable_fused_kernels, selected_dtype, result["fused_kernel_patch"]
            ):
                result["dtype_diagnostics"].append(dtype_diagnostics(torch, "before_model_construction", selected_dtype))
                current_stage = "model_construction"
                model = build_sam3_image_model(device=device)
                current_stage = "force_model_dtype"
                model = force_module_dtype(model, dtype_name, selected_dtype)
                model.eval()
                result["dtype_diagnostics"].append(
                    dtype_diagnostics(torch, "after_model_dtype_override", selected_dtype, model=model)
                )
                current_stage = "processor_construction"
                processor = Sam3Processor(model, device=device)
                result["dtype_diagnostics"].append(
                    dtype_diagnostics(torch, "after_processor_construction", selected_dtype, model=model, processor=processor)
                )
                frame_outputs = []
                for frame_index, frame in enumerate(frames):
                    current_stage = "load_frame_{}".format(frame_index)
                    image = Image.open(frame).convert("RGB")
                    result["dtype_diagnostics"].append(
                        dtype_diagnostics(
                            torch,
                            "before_set_image_frame_{}".format(frame_index),
                            selected_dtype,
                            model=model,
                            processor=processor,
                        )
                    )
                    current_stage = "set_image_frame_{}".format(frame_index)
                    state = set_image_with_dtype(
                        processor,
                        image,
                        selected_dtype,
                        diagnostics=result["dtype_diagnostics"],
                        torch_module=torch,
                        selected_dtype=selected_dtype,
                        model=model,
                    )
                    state = cast_tensors(state, selected_dtype)
                    result["dtype_diagnostics"].append(
                        dtype_diagnostics(
                            torch,
                            "before_set_text_prompt_frame_{}".format(frame_index),
                            selected_dtype,
                            model=model,
                            processor=processor,
                            state=state,
                        )
                    )
                    current_stage = "set_text_prompt_frame_{}".format(frame_index)
                    output = set_text_prompt_with_dtype(
                        processor,
                        prompt,
                        state,
                        selected_dtype,
                        diagnostics=result["dtype_diagnostics"],
                        torch_module=torch,
                        selected_dtype=selected_dtype,
                        model=model,
                    )
                    output = cast_tensors(output, selected_dtype)
                    result["dtype_diagnostics"].append(
                        dtype_diagnostics(
                            torch,
                            "after_forward_frame_{}".format(frame_index),
                            selected_dtype,
                            model=model,
                            processor=processor,
                            state=output,
                        )
                    )
                    frame_summary = summarize_sam3_outputs(output)
                    mask_count = 0
                    masks = output.get("masks") if isinstance(output, dict) else None
                    if masks is not None and hasattr(masks, "shape") and len(masks.shape) > 0:
                        mask_count = int(masks.shape[0])
                        result["preview_artifacts"].extend(save_mask_overlay(image, masks, output_dir, frame_index))
                    result["mask_counts_by_frame"].append({"frame_index": frame_index, "mask_count": mask_count})
                    frame_outputs.append(frame_summary)
                result["output_summary"] = frame_outputs
                result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        tb = traceback.format_exc()
        result["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": tb,
            "failure_stage": current_stage,
        }
        if torch is not None and selected_dtype is not None:
            try:
                result["dtype_diagnostics"].append(
                    dtype_diagnostics(
                        torch,
                        "exception_at_{}".format(current_stage),
                        selected_dtype,
                        model=model,
                        processor=processor,
                        state=state,
                    )
                )
            except Exception as diag_exc:
                result["diagnostic_error"] = "{}: {}".format(diag_exc.__class__.__name__, diag_exc)
    return result

