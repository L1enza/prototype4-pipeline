from pathlib import Path
import importlib.util

from prototype4_pipeline.integrations.vggt_import_check import run_vggt_import_check


VGGT_EXPECTED_FILES = {
    "readme": "README.md",
    "pyproject": "pyproject.toml",
    "requirements": "requirements.txt",
    "package_docs": "docs/package.md",
    "colmap_demo": "demo_colmap.py",
    "viser_demo": "demo_viser.py",
    "gradio_demo": "demo_gradio.py",
    "model_module": "vggt/models/vggt.py",
    "load_utils": "vggt/utils/load_fn.py",
    "pose_utils": "vggt/utils/pose_enc.py",
    "geometry_utils": "vggt/utils/geometry.py",
    "helper_utils": "vggt/utils/helper.py",
    "track_predict": "vggt/dependency/track_predict.py",
    "colmap_export": "vggt/dependency/np_to_pycolmap.py",
}


VGGT_IMPORT_PATHS = {
    "VGGT": "vggt.models.vggt.VGGT",
    "load_and_preprocess_images_square": "vggt.utils.load_fn.load_and_preprocess_images_square",
    "pose_encoding_to_extri_intri": "vggt.utils.pose_enc.pose_encoding_to_extri_intri",
    "unproject_depth_map_to_point_map": "vggt.utils.geometry.unproject_depth_map_to_point_map",
    "predict_tracks": "vggt.dependency.track_predict.predict_tracks",
    "batch_np_matrix_to_pycolmap": "vggt.dependency.np_to_pycolmap.batch_np_matrix_to_pycolmap",
}


VGGT_DEPENDENCY_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "Pillow": "PIL",
    "huggingface_hub": "huggingface_hub",
    "einops": "einops",
    "safetensors": "safetensors",
    "opencv-python": "cv2",
    "hydra-core": "hydra",
    "LightGlue": "lightglue",
    "pycolmap": "pycolmap",
    "trimesh": "trimesh",
}


class VGGTRunner:
    def __init__(self, context):
        self.context = context
        self.config = context.config.get("models", {}).get("vggt", {})
        self.repo = Path(context.config.get("paths", {}).get("vggt_repo", "../vggt"))

    def _repo_file_status(self):
        files = {}
        missing = []
        for name, relative_path in VGGT_EXPECTED_FILES.items():
            path = self.repo / relative_path
            present = path.exists()
            files[name] = {
                "path": str(path),
                "present": present,
            }
            if not present:
                missing.append(relative_path)
        return files, missing

    def _dependency_status(self):
        modules = {}
        missing = []
        for package_name, module_name in VGGT_DEPENDENCY_MODULES.items():
            present = importlib.util.find_spec(module_name) is not None
            modules[package_name] = {
                "module": module_name,
                "available": present,
            }
            if not present:
                missing.append(package_name)
        return modules, missing

    def _entry_points(self):
        return {
            "colmap_export_demo": {
                "path": str(self.repo / "demo_colmap.py"),
                "expected_input": "scene_dir/images/*",
                "expected_outputs": [
                    "scene_dir/sparse/cameras.bin",
                    "scene_dir/sparse/images.bin",
                    "scene_dir/sparse/points3D.bin",
                    "scene_dir/sparse/points.ply",
                ],
                "prototype_status": "prepared_only_not_executed",
            },
            "viser_demo": {
                "path": str(self.repo / "demo_viser.py"),
                "expected_input": "--image_folder <folder-of-images>",
                "prototype_status": "documented_not_used_by_pipeline_yet",
            },
            "gradio_demo": {
                "path": str(self.repo / "demo_gradio.py"),
                "expected_input": "interactive upload or example media",
                "prototype_status": "documented_not_used_by_pipeline_yet",
            },
            "python_api": {
                "model_class": "vggt.models.vggt.VGGT",
                "image_loader": "vggt.utils.load_fn.load_and_preprocess_images_square",
                "camera_conversion": "vggt.utils.pose_enc.pose_encoding_to_extri_intri",
                "depth_unprojection": "vggt.utils.geometry.unproject_depth_map_to_point_map",
                "prototype_status": "validated_by_file_layout_not_imported",
            },
        }

    def _requirements_text(self):
        requirements_path = self.repo / "requirements.txt"
        if not requirements_path.exists():
            return []
        return [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def run(self, frames):
        scene_dir = self.context.dirs["world"] / self.config.get("scene_subdir", "vggt_scene")
        image_dir = scene_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_manifest = []
        for frame in frames:
            src = Path(frame["path"])
            dst = image_dir / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    dst.write_bytes(src.read_bytes())
            image_manifest.append({"frame_index": frame["frame_index"], "image": str(dst)})

        command = ["python3", str(self.repo / "demo_colmap.py"), "--scene_dir", str(scene_dir)]
        if self.config.get("use_bundle_adjustment"):
            command.append("--use_ba")

        timestamps = [frame["timestamp_sec"] for frame in frames]
        expected_files, missing_files = self._repo_file_status()
        dependency_modules, missing_dependencies = self._dependency_status()
        import_smoke_test = run_vggt_import_check(self.repo, include_optional=True)
        repo_present = self.repo.exists()
        validation_status = "ready_for_dry_run"
        if not repo_present or missing_files:
            validation_status = "repo_incomplete"
        elif import_smoke_test.get("status") == "passed":
            validation_status = "import_smoke_passed"
        elif missing_dependencies:
            validation_status = "dependencies_missing_for_import_smoke"
        else:
            validation_status = "import_smoke_failed"

        return {
            "status": "placeholder" if not self.config.get("enabled") else "not_executed_by_prototype",
            "stage": "vggt_world_reconstruction",
            "dry_run": not self.config.get("enabled"),
            "repo": str(self.repo),
            "repo_present": repo_present,
            "clone_command_if_missing": "git clone https://github.com/facebookresearch/vggt.git ../vggt",
            "entrypoint": str(self.repo / "demo_colmap.py"),
            "available_entry_points": self._entry_points(),
            "validation": {
                "status": validation_status,
                "checks_are_lightweight": True,
                "did_run_import_smoke_test": True,
                "did_import_vggt_modules": import_smoke_test.get("status") == "passed",
                "did_instantiate_model": False,
                "did_download_weights": False,
                "did_run_inference": False,
                "import_smoke_test": import_smoke_test,
                "expected_files": expected_files,
                "missing_expected_files": missing_files,
                "expected_import_paths": VGGT_IMPORT_PATHS,
                "dependency_modules": dependency_modules,
                "missing_dependencies_for_inference": missing_dependencies,
                "requirements_txt": self._requirements_text(),
            },
            "would_run": command,
            "input_summary": {
                "frame_count": len(frames),
                "timestamp_start_sec": timestamps[0] if timestamps else None,
                "timestamp_end_sec": timestamps[-1] if timestamps else None,
                "source_video": str(self.context.video_path),
                "vggt_expected_scene_layout": str(scene_dir / "images"),
            },
            "inputs": {"frames": image_manifest},
            "outputs": {
                "output_dir": str(self.context.dirs["world"]),
                "scene_dir": str(scene_dir),
                "image_dir": str(image_dir),
                "colmap_sparse_dir": str(scene_dir / "sparse"),
                "camera_geometry": str(self.context.dirs["world"] / "camera_geometry.json"),
                "scene_points": str(self.context.dirs["world"] / "scene_points.ply"),
                "field_alignment": str(self.context.dirs["world"] / "field_alignment.json"),
            },
            "counts": {
                "prepared_images": len(image_manifest),
            },
        }
