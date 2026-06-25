import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path


IMPORT_TARGETS = [
    {
        "name": "VGGT model class",
        "module": "vggt.models.vggt",
        "attributes": ["VGGT"],
    },
    {
        "name": "VGGT image loaders",
        "module": "vggt.utils.load_fn",
        "attributes": ["load_and_preprocess_images", "load_and_preprocess_images_square"],
    },
    {
        "name": "VGGT camera conversion",
        "module": "vggt.utils.pose_enc",
        "attributes": ["pose_encoding_to_extri_intri"],
    },
    {
        "name": "VGGT geometry utilities",
        "module": "vggt.utils.geometry",
        "attributes": ["unproject_depth_map_to_point_map"],
    },
]


OPTIONAL_COLMAP_TARGETS = [
    {
        "name": "VGGT track prediction helper",
        "module": "vggt.dependency.track_predict",
        "attributes": ["predict_tracks"],
    },
    {
        "name": "VGGT COLMAP export helpers",
        "module": "vggt.dependency.np_to_pycolmap",
        "attributes": ["batch_np_matrix_to_pycolmap", "batch_np_matrix_to_pycolmap_wo_track"],
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


def _check_target(target):
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


def run_vggt_import_check(repo_path="../vggt", include_optional=True):
    repo = Path(repo_path)
    result = {
        "status": "not_run",
        "repo_path": str(repo),
        "repo_present": repo.exists(),
        "added_to_sys_path": str(repo.resolve()) if repo.exists() else None,
        "did_instantiate_model": False,
        "did_download_weights": False,
        "did_run_inference": False,
        "core_imports": [],
        "optional_colmap_imports": [],
        "missing_or_failed": [],
    }
    if not repo.exists():
        result["status"] = "repo_missing"
        result["missing_or_failed"].append("repo_missing")
        return result

    with temporary_sys_path(repo):
        result["core_imports"] = [_check_target(target) for target in IMPORT_TARGETS]
        if include_optional:
            result["optional_colmap_imports"] = [_check_target(target) for target in OPTIONAL_COLMAP_TARGETS]

    failed = []
    for record in result["core_imports"] + result["optional_colmap_imports"]:
        if record["status"] != "passed":
            failed.append({"module": record["module"], "error": record["error"]})
    result["missing_or_failed"] = failed
    result["status"] = "passed" if not failed else "failed"
    return result


def print_human_summary(result):
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "passed":
        print("VGGT import smoke check passed. No model was instantiated and no weights were downloaded.")
    elif result["status"] == "repo_missing":
        print("VGGT repo is missing. Expected ../vggt.")
    else:
        print("VGGT import smoke check failed. Install requirements/vggt-smoke.txt in the project .venv, using the correct PyTorch CUDA wheel index for this machine.")
