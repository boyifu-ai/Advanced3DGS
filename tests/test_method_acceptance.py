from __future__ import annotations

import ast
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts.install_catalog_method_extensions import (
    extract_archive_safely,
    install_module_aliases,
)
from scripts.audit_method_acceptance_coverage import audit_coverage
from scripts.audit_validation_acceptance import audit_validation_acceptance
from scripts.audit_method_training_acceptance import audit as audit_training_acceptance
from scripts.audit_benchmark_protocol import (
    EXPECTED_SCENES as BENCHMARK_EXPECTED_SCENES,
    MIP360_OUTDOOR as BENCHMARK_MIP360_OUTDOOR,
)
from scripts.audit_resolution_protocol import SCENES as RESOLUTION_SCENES
from scripts.audit_method_metrics_acceptance import (
    audit as audit_metrics_acceptance,
    audit_all_reports as audit_all_metrics_reports,
)
from scripts.audit_method_e2e_readiness import audit_e2e_readiness
from scripts.install_catalog_method_python_dependencies import (
    install_method as install_python_dependencies,
    moving_vcs_ref,
    package_import_probe,
    package_is_importable,
    package_version_requirement,
    resolved_vcs_commit,
)
from scripts.check_method_profiles import audit_configuration
from scripts.prepare_method_compatibility import (
    patch_future_annotations,
    prepare_method,
    remove_unused_module_imports,
)
from scripts.run_method_save_check import stable_verified_save_probe, verified_completion
from scripts.run_method_training import (
    auto_training_port_args,
    forwarded_has_flag,
    reader_patch_errors,
    remove_disposable_failed_output,
)
from scripts.render_mini_splatting import spherical_harmonic_degree
from scripts.render_contextgs import contextgs_device_safe_source
from scripts.render_octree_gs import octree_scalar_ape_source
from unified3dgs.method_backend import classify_failure
from scripts.run_method_stage import framework_contract_errors
from scripts.patch_hac_plus_camera_transform import (
    PATCH_MARKER as HAC_CAMERA_PATCH_MARKER,
    patch_text as patch_hac_camera_text,
)
from scripts.patch_hac_plus_chunked_mlp import (
    PATCH_MARKER as HAC_CHUNKED_MLP_PATCH_MARKER,
    patch_text as patch_hac_chunked_mlp_text,
)
from scripts.patch_third_party_readers import (
    NO_WRITE_POINT_CLOUD_BLOCK,
    POINT_CLOUD_BLOCK_PATTERN,
    REPOS as PATCH_READER_REPOS,
    patch_file as patch_dataset_reader_file,
    repair_missing_callback_functions,
    replace_colmap_point_cloud_block,
)
from scripts.check_method_preflight import repair_selected_readers
from scripts.verify_method_metrics import (
    command_for,
    training_source_mentions_flag,
)
from scripts.summarize_method_failures import collect_failures
from unified3dgs.method_catalog import (
    IMAGE_ALIASES,
    MethodPreflight,
    build_training_command,
    build_acceptance_command,
    build_method_env,
    extension_import_script,
    first_declared_flag,
    catalog_extension_spec,
    result_output_roots,
    run_capture,
    signature_value_matches,
    static_declared_flags,
    static_required_flags,
    reader_safety_errors,
    undefined_scene_callback_names,
    method_specific_extra_arg_errors,
    official_dataset_args,
    load_confirmed_catalog,
    last_json_value,
)
from unified3dgs.dataset_overlay import (
    MAX_ACCEPTANCE_POINTS,
    normalize_point_cloud,
    prepare_dataset_overlay,
    write_colmap_point_cloud,
)
from unified3dgs.methods.catalog import CatalogMethodAdapter
from unified3dgs.methods.base import MethodRunConfig
from unified3dgs.methods.student_splatting_scooping import (
    StudentSplattingScoopingAdapter,
)
from unified3dgs.methods.vanilla_3dgs import Vanilla3DGSAdapter
from unified3dgs.metrics_io import discover_render_pair, pair_images, resolved_model_output
from unified3dgs.methods.registry import available_methods, get_adapter
from unified3dgs_menu import (
    CATALOG_METHODS,
    EXTENDED_WORKFLOW_METHODS,
    METHODS,
    reset_pair_for_clean_rerun,
    select_gpu,
)


class CatalogMethodPreparationTests(unittest.TestCase):
    def test_failed_output_without_model_artifacts_is_removed_for_clean_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "method_outputs"
            output.mkdir()
            (output / "cfg_args").write_text("Namespace()\n", encoding="utf-8")
            (output / "input.ply").write_text("ply\n", encoding="utf-8")
            (output / "unified3dgs_training_report.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "saved_files": [],
                        "resolved_result_roots": [str(output)],
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(remove_disposable_failed_output(output))
            self.assertFalse(output.exists())

    def test_failed_output_with_model_artifact_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "method_outputs"
            point_cloud = output / "point_cloud" / "iteration_100" / "point_cloud.ply"
            point_cloud.parent.mkdir(parents=True)
            point_cloud.write_text("ply\n", encoding="utf-8")
            (output / "unified3dgs_training_report.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "saved_files": [],
                        "resolved_result_roots": [str(output)],
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(remove_disposable_failed_output(output))
            self.assertTrue(point_cloud.is_file())

    def test_successful_output_is_never_removed_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "method_outputs"
            output.mkdir()
            (output / "unified3dgs_training_report.json").write_text(
                json.dumps({"passed": True, "saved_files": []}),
                encoding="utf-8",
            )

            self.assertFalse(remove_disposable_failed_output(output))
            self.assertTrue(output.is_dir())

    def test_force_clean_rerun_only_removes_selected_scene_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            selected = root / "method" / "dataset" / "scene"
            sibling = root / "method" / "dataset" / "other_scene"
            (selected / "method_outputs").mkdir(parents=True)
            (selected / "method_outputs" / "checkpoint.pth").write_text(
                "checkpoint", encoding="utf-8"
            )
            sibling.mkdir(parents=True)
            (sibling / "keep.txt").write_text("keep", encoding="utf-8")

            reset_pair_for_clean_rerun(selected, root)

            self.assertFalse(selected.exists())
            self.assertTrue((sibling / "keep.txt").is_file())

    def test_force_clean_rerun_rejects_validation_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            root.mkdir()
            with self.assertRaises(ValueError):
                reset_pair_for_clean_rerun(root, root)

    def test_all_verified_methods_are_registered_in_top_level_menu(self) -> None:
        registered = set(available_methods())
        menu_keys = {method.key for method in METHODS}
        confirmed_count = len(load_confirmed_catalog())
        self.assertEqual(len(EXTENDED_WORKFLOW_METHODS), 5)
        self.assertEqual(len(CATALOG_METHODS), confirmed_count)
        self.assertEqual(len(METHODS), len(EXTENDED_WORKFLOW_METHODS) + confirmed_count)
        self.assertEqual(menu_keys, registered)
        self.assertNotIn("mmgs", registered)
        self.assertNotIn("fregs", registered)
        for method in CATALOG_METHODS:
            self.assertIsInstance(get_adapter(method.key), CatalogMethodAdapter)

    def test_formal_benchmark_uses_seven_mip360_scenes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (root / "scripts" / "run_validation.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            set(BENCHMARK_EXPECTED_SCENES["mip360"]),
            {"bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"},
        )
        self.assertEqual(BENCHMARK_MIP360_OUTDOOR, {"bicycle", "garden", "stump"})
        self.assertNotIn("|mip360/flowers", runner)
        self.assertNotIn("|mip360/treehill", runner)
        self.assertNotIn(
            ("mip360", "flowers"),
            {(family, scene) for family, scene, _ in RESOLUTION_SCENES},
        )
        self.assertNotIn(
            ("mip360", "treehill"),
            {(family, scene) for family, scene, _ in RESOLUTION_SCENES},
        )

    def test_validation_runner_has_resumable_failure_collection_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (root / "scripts" / "run_validation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"', runner)
        self.assertIn('CLEAN_FAILED_PAIR="${CLEAN_FAILED_PAIR:-0}"', runner)
        self.assertIn('STAGE_TIMEOUT_SECONDS="${STAGE_TIMEOUT_SECONDS:-0}"', runner)
        self.assertIn('classification="$(classify_stage_failure', runner)
        self.assertIn('echo "hardware_limit"', runner)
        self.assertIn('echo "program_error"', runner)
        self.assertIn('if [[ "${BENCHMARK_PROTOCOL}" != "1" ]]', runner)
        self.assertIn('validation_failures.tsv', runner)
        self.assertIn('--set "render_iteration=${actual_iteration}"', runner)
        self.assertIn('READINESS_METHODS=()', runner)
        self.assertIn(
            'vanilla_3dgs|2dgs|3dgs_mcmc|3dhgs|sss)',
            runner,
        )
        self.assertIn('--methods "${READINESS_METHODS[@]}"', runner)
        readiness_start = runner.index('READINESS_METHODS=()')
        readiness_end = runner.index(
            'for record in "${DATASET_RECORDS[@]}"', readiness_start
        )
        readiness_block = runner[readiness_start:readiness_end]
        self.assertNotIn('--methods "${METHODS_TO_RUN[@]}"', readiness_block)

    def test_release_shell_scripts_have_valid_shebang_bytes(self) -> None:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        for path in scripts.glob("*.sh"):
            with self.subTest(path=path.name):
                self.assertFalse(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_vanilla_training_gets_an_isolated_gui_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MethodRunConfig(
                method="vanilla_3dgs",
                action="train",
                dataset_path=root / "dataset",
                output_path=root / "output",
                config_path=root / "config.yaml",
                project_root=root,
                values={"iterations": 1},
                dry_run=True,
            )
            args = Vanilla3DGSAdapter().build_action_args("train", config)

        self.assertIn("--port", args)
        port = int(args[args.index("--port") + 1])
        self.assertGreaterEqual(port, 20000)
        self.assertLess(port, 40000)

    def test_vanilla_training_respects_forwarded_gui_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MethodRunConfig(
                method="vanilla_3dgs",
                action="train",
                dataset_path=root / "dataset",
                output_path=root / "output",
                config_path=root / "config.yaml",
                project_root=root,
                values={"iterations": 1},
                dry_run=True,
                extra_args=["--port=6012"],
            )
            command = Vanilla3DGSAdapter().build_command("train", config).command

        self.assertIn("--port=6012", command)
        self.assertNotIn("--port", command)

    def test_sss_evaluation_uses_official_functions_with_unified_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MethodRunConfig(
                method="sss",
                action="evaluate",
                dataset_path=root / "dataset",
                output_path=root / "output",
                config_path=root / "config.yaml",
                project_root=root,
                values={"iterations": 1, "render_iteration": 1},
                dry_run=True,
            )
            command = StudentSplattingScoopingAdapter().build_command(
                "evaluate", config
            ).command

        self.assertIn("evaluate_render_pairs_official.py", " ".join(command))
        self.assertEqual(command[command.index("--style") + 1], "standard_3dgs")
        self.assertEqual(command[command.index("--iteration") + 1], "1")

    def test_render_pairing_matches_numeric_suffixes_before_sorted_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renders = root / "renders"
            gt = root / "gt"
            renders.mkdir()
            gt.mkdir()
            for name in ("r00002.png", "r00010.png"):
                (renders / name).write_bytes(b"image")
            for name in ("gt_00010.png", "gt_00002.png"):
                (gt / name).write_bytes(b"image")

            pairs = pair_images(renders, gt)

        self.assertEqual(
            [(render.name, target.name) for render, target in pairs],
            [
                ("r00002.png", "gt_00002.png"),
                ("r00010.png", "gt_00010.png"),
            ],
        )

    def test_optional_capability_audit_preserves_short_training_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / "render.py").write_text(
                "import argparse\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('-s')\n"
                "p.add_argument('-m')\n",
                encoding="utf-8",
            )
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps({"methods": [{"method": "example", "passed": True}]}),
                encoding="utf-8",
            )
            payload = audit_e2e_readiness(
                coverage,
                catalog=[
                    {
                        "key": "example",
                        "title": "Example",
                        "local_path": str(repo),
                    }
                ],
                profiles=[{"key": "example"}],
            )
            self.assertEqual(payload["short_training_verified_count"], 1)
            self.assertEqual(payload["static_optional_capability_passed_count"], 1)
            self.assertFalse(payload["all_runtime_optional_capabilities_verified"])

    def test_catalog_adapter_exposes_configurable_long_training(self) -> None:
        adapter = get_adapter("ges")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MethodRunConfig(
                method="ges",
                action="train",
                dataset_path=root / "dataset",
                output_path=root / "output",
                config_path=root / "config.yaml",
                project_root=root,
                values={"iterations": 2},
                dry_run=True,
            )
            command = adapter.build_command("train", config).command
            self.assertIn("run_method_training.py", " ".join(command))
            self.assertEqual(command[command.index("--iterations") + 1], "2")
            render_command = adapter.build_command("render", config).command
            self.assertIn("run_method_stage.py", " ".join(render_command))
            self.assertIn("--stage", render_command)
            self.assertEqual(render_command[render_command.index("--stage") + 1], "render")
            eval_command = adapter.build_command("evaluate", config).command
            self.assertEqual(eval_command[eval_command.index("--stage") + 1], "eval")

    def test_long_training_builder_uses_requested_final_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "train.py"
            entry.write_text("", encoding="utf-8")
            result = MethodPreflight(
                key="example",
                title="Example",
                repo=root,
                entry=entry,
                command_flags={
                    "source": "-s",
                    "output": "-m",
                    "iterations": "--iterations",
                    "save": "--save_iterations",
                    "test": "--test_iterations",
                    "eval": "--eval",
                    "images": "--images",
                },
                details={"static_declared_cli_options": ["--resolution"]},
            )
            command = build_training_command(
                result,
                root / "dataset",
                root / "output",
                iterations=30000,
                images="images_4",
                resolution=-1,
                eval_enabled=True,
                test_iterations=-1,
            )
            self.assertEqual(command[command.index("--iterations") + 1], "30000")
            self.assertEqual(command[command.index("--save_iterations") + 1], "30000")
            self.assertIn("images_4", command)
            self.assertIn("--resolution", command)

    def test_catalog_long_training_auto_assigns_unique_gui_port(self) -> None:
        result = MethodPreflight(
            key="example",
            title="Example",
            repo=Path.cwd(),
            details={"static_declared_cli_options": ["--port"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            args = auto_training_port_args(
                result,
                [],
                Path(directory) / "method_outputs",
            )
        self.assertEqual(args[0], "--port")
        self.assertTrue(args[1].isdigit())

    def test_catalog_long_training_respects_user_gui_port(self) -> None:
        result = MethodPreflight(
            key="example",
            title="Example",
            repo=Path.cwd(),
            details={"static_declared_cli_options": ["--port"]},
        )
        self.assertTrue(forwarded_has_flag(["--port", "6011"], "--port"))
        self.assertTrue(forwarded_has_flag(["--port=6012"], "--port"))
        self.assertEqual(
            auto_training_port_args(result, ["--port", "6011"], Path.cwd()),
            [],
        )

    def test_catalog_long_training_blocks_unpatched_standard_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            reader = repo / "scene" / "dataset_readers.py"
            reader.parent.mkdir()
            reader.write_text(
                "def fetchPly(path):\n"
                "    return None\n\n"
                "def read(path):\n"
                "    ply_path = 'points3D.ply'\n"
                "    storePly(ply_path, [], [])\n",
                encoding="utf-8",
            )
            errors = reader_patch_errors(repo)
            self.assertEqual(len(errors), 2)
            reader.write_text(
                "def fetchPly(path):\n"
                "    '''Unified 3DGS robust PLY reader patch.'''\n"
                "    return None\n\n"
                "def fetchColmapPointCloud():\n"
                "    '''Unified 3DGS no dataset-write point cloud patch.'''\n"
                "    pass\n",
                encoding="utf-8",
            )
            self.assertEqual(reader_patch_errors(repo), [])

    def test_declared_flag_fallback_recovers_dynamic_eval_option(self) -> None:
        self.assertEqual(
            first_declared_flag(
                ["--source_path", "--model_path", "--eval"],
                ("--eval",),
            ),
            "--eval",
        )

    def test_method_acceptance_coverage_requires_each_confirmed_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts"
            first = attempts / "first"
            second = attempts / "second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            saved_a = first / "a_iteration_1.ply"
            saved_b = second / "b_iteration_1.ply"
            saved_a.write_text("a", encoding="utf-8")
            saved_b.write_text("b", encoding="utf-8")
            (first / "acceptance_results.json").write_text(
                json.dumps(
                    [
                        {
                            "method": "a",
                            "status": "passed",
                            "command": ["python", "train.py", "--iterations", "1"],
                            "saved_files": [str(saved_a)],
                            "unexpected_iteration_artifacts": [],
                            "completion_mode": "process_exit",
                        },
                        {
                            "method": "b",
                            "status": "failed",
                            "command": ["python", "train.py", "--iterations", "1"],
                            "saved_files": [],
                            "unexpected_iteration_artifacts": [],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            catalog = [{"key": "a", "title": "A"}, {"key": "b", "title": "B"}]
            incomplete = audit_coverage(root, catalog)
            self.assertFalse(incomplete["all_passed"])
            self.assertEqual(incomplete["missing_or_unverified_methods"], ["b"])
            (second / "acceptance_results.json").write_text(
                json.dumps(
                    [
                        {
                            "method": "b",
                            "status": "passed",
                            "command": ["python", "train.py", "--iterations=1"],
                            "saved_files": [str(saved_b)],
                            "unexpected_iteration_artifacts": [],
                            "completion_mode": "verified_save_stop",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            complete = audit_coverage(root, catalog)
            self.assertTrue(complete["all_passed"])
            self.assertEqual(complete["passed_method_count"], 2)

    def test_configurable_training_acceptance_requires_live_saved_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempts" / "first"
            output = attempt / "ges" / "method_outputs"
            saved = output / "point_cloud" / "iteration_1" / "point_cloud.ply"
            saved.parent.mkdir(parents=True)
            saved.write_text("result", encoding="utf-8")
            (output / "unified3dgs_training_report.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "iterations": 1,
                        "saved_files": [str(saved)],
                    }
                ),
                encoding="utf-8",
            )
            (attempt / "acceptance_results.json").write_text(
                json.dumps(
                    {
                        "methods": [
                            {
                                "method": "ges",
                                "exit_code": 0,
                                "verified": True,
                                "output": str(output),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = audit_training_acceptance(root)
            ges = next(row for row in payload["methods"] if row["method"] == "ges")
            self.assertTrue(ges["passed"])
            saved.unlink()
            payload = audit_training_acceptance(root)
            ges = next(row for row in payload["methods"] if row["method"] == "ges")
            self.assertFalse(ges["passed"])

    def test_metrics_acceptance_requires_all_method_dataset_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempts" / "first"
            attempt.mkdir(parents=True)
            (attempt / "metrics_acceptance_results.json").write_text(
                json.dumps(
                    {
                        "protocol_mode": "official_short",
                        "results": [
                            {
                                "method": "ges",
                                "dataset": "mip360/garden",
                                "passed": True,
                                "official_protocol": True,
                                "official_runtime_verified": True,
                                "metrics": {
                                    "psnr": 1.0,
                                    "ssim": 1.0,
                                    "lpips": 0.0,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = audit_metrics_acceptance(
                attempt / "metrics_acceptance_results.json",
                expected_dataset_count=3,
            )
            self.assertFalse(payload["all_passed"])
            self.assertEqual(payload["passed_result_count"], 1)
            self.assertIn("beta_splatting", payload["missing_methods"])

    def test_metrics_acceptance_merges_selective_retry_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for attempt_name, dataset in (
                ("first", "mip360/garden"),
                ("second", "tandt/train"),
            ):
                attempt = root / "attempts" / attempt_name
                attempt.mkdir(parents=True)
                (attempt / "metrics_acceptance_results.json").write_text(
                    json.dumps(
                        {
                            "protocol_mode": "official_short",
                            "results": [
                                {
                                    "method": "ges",
                                    "dataset": dataset,
                                    "passed": True,
                                    "official_protocol": True,
                                    "official_runtime_verified": True,
                                    "metrics": {
                                        "psnr": 1.0,
                                        "ssim": 1.0,
                                        "lpips": 0.0,
                                    },
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            payload = audit_all_metrics_reports(root)
            self.assertEqual(payload["passed_result_count"], 2)
            self.assertFalse(payload["all_passed"])

    def test_render_pair_discovery_supports_scaled_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "method_outputs"
            pair = output / "test" / "ours_1"
            renders = pair / "renders_1"
            gt = pair / "gt_1"
            renders.mkdir(parents=True)
            gt.mkdir(parents=True)
            (renders / "00000.png").write_bytes(b"render")
            (gt / "00000.png").write_bytes(b"gt")
            discovered = discover_render_pair(output, 1)
            self.assertEqual(discovered.renders, renders)
            self.assertEqual(discovered.gt, gt)
            self.assertEqual(len(discovered.pairs), 1)

    def test_render_pair_discovery_supports_gof_style_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "method_outputs"
            renders = output / "test_preds_1.0"
            gt = output / "test_gt_1.0"
            renders.mkdir(parents=True)
            gt.mkdir(parents=True)
            (renders / "00000.png").write_bytes(b"render")
            (gt / "00000.png").write_bytes(b"gt")
            discovered = discover_render_pair(output, 1)
            self.assertEqual(discovered.renders, renders)
            self.assertEqual(discovered.gt, gt)
            self.assertEqual(len(discovered.pairs), 1)

    def test_mini_splatting_saved_sh_degree_is_inferred_from_properties(self) -> None:
        self.assertEqual(spherical_harmonic_degree(["x", "y", "z", "f_dc_0"]), 0)
        degree_three = [f"f_rest_{index}" for index in range(45)]
        self.assertEqual(spherical_harmonic_degree(degree_three), 3)
        with self.assertRaises(ValueError):
            spherical_harmonic_degree(["f_rest_0"])

    def test_framework_render_contract_reports_upstream_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "render.py"
            source.write_text("def render_sets():\n    pass\n", encoding="utf-8")
            profile = {
                "framework_render_contracts": [
                    {
                        "path": "render.py",
                        "contains": ["def render_sets", "ape_code[0]"],
                    }
                ]
            }
            errors = framework_contract_errors(repo, profile)
            self.assertEqual(len(errors), 1)
            self.assertIn("ape_code[0]", errors[0])
            source.write_text(
                "def render_sets():\n    return ape_code[0]\n",
                encoding="utf-8",
            )
            self.assertEqual(framework_contract_errors(repo, profile), [])

    def test_resolved_model_output_uses_reported_derived_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "method_outputs"
            derived = root / "method_outputs_variant"
            saved = derived / "point_cloud" / "iteration_1" / "point_cloud.ply"
            saved.parent.mkdir(parents=True)
            saved.write_text("ply", encoding="utf-8")
            output.mkdir()
            (output / "unified3dgs_training_report.json").write_text(
                json.dumps(
                    {
                        "resolved_result_roots": [str(output), str(derived)],
                        "saved_files": [str(saved)],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolved_model_output(output), derived.resolve())

    def test_safe_archive_extracts_setup_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "extension.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("extension/setup.py", "print('setup')\n")
            setup_root = extract_archive_safely(archive, root / "extracted")
            self.assertEqual(setup_root.name, "extension")
            self.assertTrue((setup_root / "setup.py").is_file())

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside/setup.py", "print('unsafe')\n")
            with self.assertRaises(ValueError):
                extract_archive_safely(archive, root / "extracted")

    def test_catalog_extension_spec_includes_archives_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "extension.zip").write_bytes(b"placeholder")
            sources, modules, errors = catalog_extension_spec(
                repo,
                {
                    "archived_extensions": [
                        {"archive": "extension.zip", "modules": ["source_module"]}
                    ],
                    "module_aliases": [
                        {
                            "source": "source_module",
                            "target": "target_module",
                            "submodules": ["_C"],
                        }
                    ],
                },
            )
            self.assertEqual(errors, [])
            self.assertIn("archive:extension.zip", sources)
            self.assertIn("target_module", modules)
            self.assertIn("target_module._C", modules)

    def test_module_alias_imports_source_and_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            source = target / "source_module"
            source.mkdir()
            (source / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
            (source / "_C.py").write_text("READY = True\n", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(target)
            records = install_module_aliases(
                [
                    {
                        "source": "source_module",
                        "target": "target_module",
                        "submodules": ["_C"],
                    }
                ],
                target,
                env,
            )
            self.assertEqual(len(records), 1)
            self.assertTrue((target / "target_module" / "__init__.py").is_file())

    def test_python38_annotation_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.py"
            path.write_text(
                '"""Scene module."""\n\nclass Scene:\n    value: int | str\n',
                encoding="utf-8",
            )
            first = patch_future_annotations(path, run_real=True)
            second = patch_future_annotations(path, run_real=True)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(first["status"], "patched")
            self.assertEqual(second["status"], "already_patched")
            self.assertEqual(text.count("from __future__ import annotations"), 1)
            self.assertLess(
                text.index('"""Scene module."""'),
                text.index("from __future__ import annotations"),
            )

    def test_static_required_flags_find_method_specific_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            entry = repo / "train.py"
            entry.write_text(
                "import argparse\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--config', required=True)\n"
                "parser.add_argument('--optional')\n",
                encoding="utf-8",
            )
            self.assertEqual(static_required_flags(repo, entry), ["--config"])

    def test_static_declared_flags_find_dynamic_parameter_group_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            arguments = repo / "arguments"
            arguments.mkdir()
            entry = repo / "train.py"
            entry.write_text("from arguments import Params\n", encoding="utf-8")
            (arguments / "__init__.py").write_text(
                "class Params:\n"
                "    def __init__(self, parser):\n"
                "        self._n_offsets = 10\n"
                "        self.voxel_size = 0.001\n"
                "        for key, value in vars(self).items():\n"
                "            parser.add_argument('--' + key.lstrip('_'))\n"
                "        parser.add_argument('--literal')\n",
                encoding="utf-8",
            )
            flags = static_declared_flags(repo, entry)
            self.assertIn("--n_offsets", flags)
            self.assertIn("--voxel_size", flags)
            self.assertIn("--literal", flags)

    def test_hac_plus_acceptance_profile_cannot_override_n_offsets(self) -> None:
        errors = method_specific_extra_arg_errors(
            "hac_plus", ["--resolution", "8", "--n_offsets", "2"]
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("preserve upstream n_offsets=10", errors[0])
        self.assertEqual(
            method_specific_extra_arg_errors("hac_plus", ["--resolution", "8"]),
            [],
        )

    def test_hac_plus_acceptance_command_preserves_model_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            repo = root / "repo"
            (dataset / "images_8").mkdir(parents=True)
            repo.mkdir()
            entry = repo / "train.py"
            entry.write_text("", encoding="utf-8")
            result = MethodPreflight(
                key="hac_plus",
                title="HAC++",
                repo=repo,
                entry=entry,
                command_flags={
                    "source": "-s",
                    "output": "-m",
                    "iterations": "--iterations",
                    "save": "--save_iterations",
                    "test": "--test_iterations",
                    "eval": None,
                    "images": None,
                },
            )
            command = build_acceptance_command(
                result,
                dataset,
                output,
                {"extra_args": ["--resolution", "8"]},
            )
            self.assertEqual(command[-2:], ["--resolution", "8"])
            self.assertNotIn("--n_offsets", command)

    def test_verified_save_can_complete_after_upstream_post_save_failure(self) -> None:
        saved = [Path("point_cloud/iteration_1/point_cloud.ply")]
        passed, mode = verified_completion(
            status=1,
            stopped_after_save=False,
            saved=saved,
            unexpected=[],
            profile={"stop_after_verified_save": True},
        )
        self.assertTrue(passed)
        self.assertEqual(mode, "verified_save_after_process_exit")
        failed, _ = verified_completion(
            status=1,
            stopped_after_save=False,
            saved=saved,
            unexpected=[],
            profile={},
        )
        self.assertFalse(failed)

    def test_unused_optional_import_is_removed_but_used_import_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            unused = repo / "unused.py"
            used = repo / "used.py"
            unused.write_text("import plas\nVALUE = 1\n", encoding="utf-8")
            used.write_text("import plas\nVALUE = plas.VALUE\n", encoding="utf-8")
            record = remove_unused_module_imports(repo, "plas", run_real=True)
            self.assertEqual(record["status"], "still_required")
            unused_tree = ast.parse(unused.read_text(encoding="utf-8"))
            self.assertFalse(
                any(isinstance(node, ast.Import) for node in ast.walk(unused_tree))
            )
            self.assertIn("import plas", used.read_text(encoding="utf-8"))

    def test_required_plas_import_is_not_a_compatibility_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "used.py").write_text(
                "from plas import sort_with_plas\nVALUE = sort_with_plas\n",
                encoding="utf-8",
            )
            with mock.patch(
                "scripts.prepare_method_compatibility.resolve_project_path",
                return_value=repo,
            ):
                record, failed = prepare_method(
                    {"key": "beta_splatting", "local_path": "unused"}, run_real=False
                )
            self.assertFalse(failed)
            self.assertEqual(record["status"], "still_required")

    def test_vcs_dependency_can_use_separate_version_requirement(self) -> None:
        package = {
            "requirement": "git+https://example.invalid/repo.git@v1",
            "version_requirement": "package==1.0",
        }
        self.assertEqual(package_version_requirement(package), "package==1.0")

    def test_moving_vcs_dependency_is_detected(self) -> None:
        self.assertTrue(
            moving_vcs_ref(
                {
                    "requirement": "git+https://example.invalid/repo.git@main",
                    "allow_moving_ref": True,
                }
            )
        )
        self.assertFalse(
            moving_vcs_ref(
                {
                    "requirement": "git+https://example.invalid/repo.git@v1.0",
                    "allow_moving_ref": True,
                }
            )
        )

    def test_isolated_dependency_must_resolve_from_method_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            module = target / "isolated_example"
            module.mkdir()
            (module / "__init__.py").write_text("READY = True\n", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(target)
            package = {
                "requirement": "isolated-example",
                "import_name": "isolated_example",
                "version_requirement": "",
                "require_isolated": True,
            }
            self.assertTrue(package_is_importable(package, env, target))
            self.assertFalse(package_is_importable(package, env, root / "other"))

    def test_dependency_probe_rejects_incomplete_import_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            module = target / "incomplete_example"
            module.mkdir()
            (module / "__init__.py").write_text(
                "import unified3dgs_dependency_that_does_not_exist\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(target)
            package = {
                "requirement": "incomplete-example",
                "import_name": "incomplete_example",
                "version_requirement": "",
                "require_isolated": True,
            }
            self.assertFalse(package_is_importable(package, env, target))
            passed, output = package_import_probe(package, env, target)
            self.assertFalse(passed)
            self.assertIn("unified3dgs_dependency_that_does_not_exist", output)

    def test_already_importable_package_does_not_overwrite_method_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            with mock.patch(
                "scripts.install_catalog_method_python_dependencies.PROJECT_ROOT", root
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.resolve_project_path",
                return_value=repo,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.build_method_env",
                return_value=os.environ.copy(),
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.package_is_importable",
                return_value=True,
            ):
                record = install_python_dependencies(
                    {"key": "example", "local_path": "unused"},
                    {"packages": ["already-there"], "manual_blockers": []},
                    run_real=True,
                    timeout_seconds=1,
                    min_free_disk_gb=0,
                    pip_timeout_seconds=1,
                    pip_retries=0,
                )
            self.assertEqual(record["method"], "example")
            self.assertEqual(record["status"], "installed")

    def test_approved_dependency_closure_omits_no_deps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            commands = []

            def capture(command, *_args):
                commands.append(command)
                return 0

            with mock.patch(
                "scripts.install_catalog_method_python_dependencies.PROJECT_ROOT", root
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.resolve_project_path",
                return_value=repo,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.build_method_env",
                return_value=os.environ.copy(),
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.package_is_importable",
                return_value=False,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.package_import_probe",
                return_value=(True, ""),
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.disk_error",
                return_value=None,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.run_streaming_build",
                side_effect=capture,
            ):
                record = install_python_dependencies(
                    {"key": "example", "local_path": "unused"},
                    {
                        "packages": [
                            {
                                "requirement": "nerfview==0.1.3",
                                "import_name": "nerfview",
                                "install_dependencies": True,
                            }
                        ],
                        "manual_blockers": [],
                    },
                    run_real=True,
                    timeout_seconds=1,
                    min_free_disk_gb=0,
                    pip_timeout_seconds=1,
                    pip_retries=0,
                )
            self.assertEqual(record["status"], "installed")
            self.assertEqual(len(commands), 1)
            self.assertNotIn("--no-deps", commands[0])

    def test_failed_import_is_revalidated_after_complete_dependency_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            with mock.patch(
                "scripts.install_catalog_method_python_dependencies.PROJECT_ROOT", root
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.resolve_project_path",
                return_value=repo,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.build_method_env",
                return_value=os.environ.copy(),
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.package_is_importable",
                return_value=False,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.package_import_probe",
                side_effect=[
                    (False, "missing later dependency"),
                    (True, ""),
                    (True, ""),
                ],
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.disk_error",
                return_value=None,
            ), mock.patch(
                "scripts.install_catalog_method_python_dependencies.run_streaming_build",
                return_value=0,
            ):
                record = install_python_dependencies(
                    {"key": "example", "local_path": "unused"},
                    {
                        "packages": ["consumer-package", "later-dependency"],
                        "manual_blockers": [],
                    },
                    run_real=True,
                    timeout_seconds=1,
                    min_free_disk_gb=0,
                    pip_timeout_seconds=1,
                    pip_retries=0,
                )
            self.assertEqual(record["status"], "installed")
            self.assertEqual(
                record["installs"][0]["status"],
                "installed_after_dependency_closure",
            )
            self.assertTrue(
                record["installs"][0]["revalidated_after_all_packages"]
            )

    def test_vcs_dependency_commit_is_recorded_from_pip_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "pip.log"
            log.write_text(
                "Resolved https://example.invalid/repo.git to commit "
                "ABCDEF1234567890\n",
                encoding="utf-8",
            )
            self.assertEqual(resolved_vcs_commit(log), "abcdef1234567890")

    def test_run_capture_decodes_timeout_bytes(self) -> None:
        timeout = subprocess.TimeoutExpired(
            cmd=["git", "submodule", "status"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        with mock.patch("unified3dgs.method_catalog.subprocess.run", side_effect=timeout):
            code, output = run_capture(
                ["git", "submodule", "status"],
                Path.cwd(),
                os.environ.copy(),
                1,
            )
        self.assertEqual(code, 124)
        self.assertEqual(output, "partial stdout\npartial stderr")

    def test_beta_environment_enables_python38_functools_cache_compatibility(self) -> None:
        env = build_method_env("beta_splatting", Path.cwd())
        self.assertEqual(env["UNIFIED3DGS_PY38_FUNCTOOLS_CACHE"], "1")
        other = build_method_env("contextgs", Path.cwd())
        self.assertNotIn("UNIFIED3DGS_PY38_FUNCTOOLS_CACHE", other)

    def test_octree_environment_enables_numpy_legacy_aliases_only_for_octree(self) -> None:
        env = build_method_env("octree_gs", Path.cwd())
        self.assertEqual(env["UNIFIED3DGS_NUMPY_LEGACY_ALIASES"], "1")
        other = build_method_env("contextgs", Path.cwd())
        self.assertNotIn("UNIFIED3DGS_NUMPY_LEGACY_ALIASES", other)

    def test_pgsr_environment_enables_nested_colmap_read_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory) / "shared"
            sparse = dataset_root / "scene" / "sparse"
            nested = sparse / "0"
            nested.mkdir(parents=True)
            (nested / "images.bin").write_bytes(b"nested-colmap")
            missing_flat = sparse / "images.bin"

            env = build_method_env("pgsr", Path.cwd())
            env["UNIFIED3DGS_READONLY_DATASET_ROOT"] = str(dataset_root)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os; from pathlib import Path; "
                        f"p=Path({str(missing_flat)!r}); "
                        "assert p.read_bytes() == b'nested-colmap'; "
                        "fd=os.open(str(p), os.O_RDONLY); "
                        "assert os.read(fd, 64) == b'nested-colmap'; os.close(fd)"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(missing_flat.exists())

        other = build_method_env("contextgs", Path.cwd())
        self.assertNotIn("UNIFIED3DGS_COLMAP_SPARSE_ZERO_FALLBACK", other)

    def test_octree_numpy_legacy_alias_runtime_guard(self) -> None:
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is unavailable in the local test runtime")
        env = build_method_env("octree_gs", Path.cwd())
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import numpy as np; assert np.int is int; print('ok')",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ok")

    def test_method_environments_share_torch_download_cache(self) -> None:
        first = build_method_env("hac_plus", Path.cwd())
        second = build_method_env("wavelet_gs", Path.cwd())
        self.assertEqual(first["TORCH_HOME"], second["TORCH_HOME"])
        self.assertIn("shared_torch_cache", first["TORCH_HOME"])

    def test_validation_acceptance_separates_hardware_and_program_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "gpu0"

            passed = worker / "passed" / "mip360" / "garden"
            (passed / "method_outputs").mkdir(parents=True)
            (passed / ".eval.done").write_text("done\n", encoding="utf-8")
            (passed / "method_outputs" / "results.json").write_text(
                json.dumps(
                    {
                        "ours_1": {
                            "PSNR": 10.0,
                            "SSIM": 0.5,
                            "LPIPS": 0.4,
                        }
                    }
                ),
                encoding="utf-8",
            )

            limited = worker / "limited" / "mip360" / "garden"
            (limited / "method_outputs").mkdir(parents=True)
            (limited / ".train.failed").write_text("failed\n", encoding="utf-8")
            (limited / "method_outputs" / "unified3dgs_training_report.json").write_text(
                json.dumps(
                    {
                        "failure_classification": {
                            "category": "hardware_limit_confirmed",
                            "objective_limit": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            failed = worker / "failed" / "mip360" / "garden"
            failed.mkdir(parents=True)
            (failed / ".train.failed").write_text("failed\n", encoding="utf-8")

            payload = audit_validation_acceptance(
                root,
                ["passed", "limited", "failed"],
                ["mip360/garden"],
                1,
            )
            accepted = audit_validation_acceptance(
                root,
                ["passed", "limited"],
                ["mip360/garden"],
                1,
            )

        self.assertEqual(payload["counts"]["metrics_passed"], 1)
        self.assertEqual(payload["counts"]["hardware_limited"], 1)
        self.assertEqual(payload["counts"]["program_failed"], 1)
        self.assertFalse(payload["framework_accepted"])
        self.assertTrue(accepted["framework_accepted"])
        self.assertFalse(accepted["metrics_complete"])

    def test_acceptance_environment_disables_wandb_and_limits_cuda_fragmentation(self) -> None:
        env = build_method_env("ges", Path.cwd())
        self.assertEqual(env["WANDB_MODE"], "disabled")
        self.assertEqual(env["WANDB_CONSOLE"], "off")
        self.assertIn("max_split_size_mb", env["PYTORCH_ALLOC_CONF"])
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
        self.assertEqual(
            IMAGE_ALIASES, ("images", "images_2", "images_4", "images_8")
        )

    def test_result_output_roots_include_profile_declared_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "method_outputs"
            sibling = root / "method_outputs_00_timestamp"
            output.mkdir()
            sibling.mkdir()
            roots = result_output_roots(
                output, {"output_globs": ["{output}_*"]}
            )
            self.assertEqual(roots, [output, sibling])

    def test_normalize_point_cloud_adds_normals_and_preserves_colors(self) -> None:
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            self.skipTest("numpy/plyfile are unavailable in the local test runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.ply"
            output = root / "normalized.ply"
            vertices = np.array(
                [(1.0, 2.0, 3.0, 10, 20, 30)],
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(
                str(source)
            )
            details = normalize_point_cloud(source, output)
            self.assertEqual(details["added_fields"], ["nx", "ny", "nz"])
            normalized = PlyData.read(str(output))["vertex"].data
            self.assertEqual(float(normalized["nx"][0]), 0.0)
            self.assertEqual(int(normalized["red"][0]), 10)

    def test_normalize_point_cloud_applies_deterministic_acceptance_point_cap(self) -> None:
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            self.skipTest("numpy/plyfile are unavailable in the local test runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.ply"
            output = root / "normalized.ply"
            vertices = np.array(
                [(float(index), 0.0, 0.0) for index in range(20)],
                dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
            )
            PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(
                str(source)
            )
            details = normalize_point_cloud(source, output, max_points=5)
            normalized = PlyData.read(str(output))["vertex"].data
            self.assertEqual(details["source_vertex_count"], 20)
            self.assertEqual(details["vertex_count"], 5)
            self.assertTrue(details["point_cap_applied"])
            self.assertEqual(float(normalized["x"][0]), 0.0)
            self.assertEqual(float(normalized["x"][-1]), 19.0)
            self.assertEqual(MAX_ACCEPTANCE_POINTS, 10_000)

    def test_reduced_colmap_points_match_normalized_ply(self) -> None:
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            self.skipTest("numpy/plyfile are unavailable in the local test runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "points3D.ply"
            binary = root / "points3D.bin"
            text = root / "points3D.txt"
            vertices = np.array(
                [
                    (1.0, 2.0, 3.0, 10, 20, 30),
                    (4.0, 5.0, 6.0, 40, 50, 60),
                ],
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
                str(ply)
            )
            details = write_colmap_point_cloud(ply, binary, text)
            self.assertEqual(details["point_count"], 2)
            payload = binary.read_bytes()
            self.assertEqual(struct.unpack("<Q", payload[:8])[0], 2)
            self.assertEqual(len(payload), 8 + 2 * struct.calcsize("<QdddBBBdQ"))
            self.assertIn("2 4 5 6 40 50 60 0", text.read_text(encoding="ascii"))

    def test_contextgs_device_patch_keeps_index_tensor_on_cuda_device(self) -> None:
        source = (
            "def multi_scale_generating(anchor, to_code):\n"
            "    return torch.arange(anchor.shape[0])[to_code]\n"
        )
        patched = contextgs_device_safe_source(source)
        self.assertIn("device=to_code.device", patched)
        self.assertNotIn("torch.arange(anchor.shape[0])[to_code]", patched)

    def test_octree_appearance_patch_normalizes_to_scalar(self) -> None:
        source = (
            "def generate_neural_gaussians(ape_code):\n"
            "    return ape_code < 0, ape_code[0]\n"
        )
        patched = octree_scalar_ape_source(source)
        self.assertIn("ape_code < 0, ape_code", patched)
        self.assertNotIn("ape_code[0]", patched)

    def test_hardware_limit_requires_official_backend_and_protocol(self) -> None:
        backend = mock.Mock(official=True, errors=[])
        output = (
            "torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 35.53 GiB (GPU 0; 23.68 GiB total capacity)"
        )
        result = classify_failure(output, backend, official_protocol=True)
        self.assertEqual(result["category"], "hardware_limit_confirmed")
        result = classify_failure(output, backend, official_protocol=False)
        self.assertEqual(result["category"], "resource_or_program_error")

    def test_program_failure_takes_priority_over_oom(self) -> None:
        backend = mock.Mock(official=True, errors=[])
        output = (
            "RuntimeError: numel: integer multiplication overflow\n"
            "CUDA out of memory. Tried to allocate 35.53 GiB "
            "(GPU 0; 23.68 GiB total capacity)"
        )
        result = classify_failure(output, backend, official_protocol=True)
        self.assertEqual(result["category"], "program_error")

    def test_metrics_acceptance_forwards_dataset_specific_training_args(self) -> None:
        command = command_for(
            "train_all.py",
            "hac_plus",
            "config.yaml",
            Path("/dataset"),
            Path("/output"),
            {"iterations": 1},
            ["--voxel_size", "0.01"],
        )
        self.assertEqual(command[-3:], ["--", "--voxel_size", "0.01"])

    def test_official_dataset_args_are_shared_by_menu_and_acceptance(self) -> None:
        profile = {
            "official_dataset_args": {
                "tandt": ["--voxel_size", "0.01"],
            }
        }
        self.assertEqual(
            official_dataset_args(
                profile,
                Path("/any/local/path/train"),
                label="tandt/train",
            ),
            ["--voxel_size", "0.01"],
        )

    def test_stable_verified_save_probe_requires_unchanged_exact_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "method_outputs"
            result = output / "point_cloud" / "iteration_1" / "point_cloud.ply"
            result.parent.mkdir(parents=True)
            probe = stable_verified_save_probe(
                output, {}, newer_than=0.0, settle_seconds=0.0
            )
            self.assertFalse(probe())
            result.write_bytes(b"result")
            self.assertFalse(probe())
            self.assertTrue(probe())

    def test_acceptance_dataset_overlay_rejects_output_outside_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory)
            with self.assertRaisesRegex(ValueError, "must stay inside the project"):
                prepare_dataset_overlay(Path.cwd(), outside)

    def test_extension_probe_imports_torch_before_native_modules(self) -> None:
        script = extension_import_script(["_gridencoder"])
        self.assertLess(script.index("import torch"), script.index("_gridencoder"))

    def test_probe_json_parser_ignores_warnings_after_payload(self) -> None:
        output = '{"torch_cuda": "11.8"}\n[W runtime warning] deprecated option\n'
        self.assertEqual(last_json_value(output), {"torch_cuda": "11.8"})

    def test_scene_callback_validator_rejects_undefined_function(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / "dataset_readers.py"
            reader.write_text(
                "def readColmapSceneInfo():\n    pass\n\n"
                "sceneLoadTypeCallbacks = {\n"
                "    'Colmap': readColmapSceneInfo,\n"
                "    'Blender': readNerfSyntheticInfo,\n"
                "}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                undefined_scene_callback_names(reader), ["readNerfSyntheticInfo"]
            )

    def test_reader_safety_rejects_incomplete_marked_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / "dataset_readers.py"
            reader.write_text(
                "def fetchPly(path):\n"
                "    '''Unified 3DGS robust PLY reader patch.'''\n"
                "    return None\n\n"
                "def readColmapSceneInfo():\n"
                "    '''Unified 3DGS no dataset-write point cloud patch.'''\n"
                "    ply_path = 'points3D.ply'\n"
                "    storePly(ply_path)\n"
                "    return fetchPly(ply_path)\n",
                encoding="utf-8",
            )
            errors = reader_safety_errors(reader, "scaffold_gs")
            self.assertEqual(len(errors), 1)
            self.assertIn("incomplete no-write patch", errors[0])
            self.assertIn("--method scaffold_gs", errors[0])

    def test_reader_repair_restores_callback_and_local_dependency(self) -> None:
        broken = (
            "def readColmapSceneInfo():\n    pass\n\n"
            "sceneLoadTypeCallbacks = {\n"
            "    'Colmap': readColmapSceneInfo,\n"
            "    'Blender': readNerfSyntheticInfo,\n"
            "}\n"
        )
        pristine = (
            "def readColmapSceneInfo():\n    pass\n\n"
            "def readCamerasFromTransforms():\n    return []\n\n"
            "def readNerfSyntheticInfo():\n"
            "    return readCamerasFromTransforms()\n\n"
            "sceneLoadTypeCallbacks = {\n"
            "    'Colmap': readColmapSceneInfo,\n"
            "    'Blender': readNerfSyntheticInfo,\n"
            "}\n"
        )
        with mock.patch(
            "scripts.patch_third_party_readers._pristine_reader_source",
            return_value=pristine,
        ):
            repaired, restored = repair_missing_callback_functions(
                Path("dataset_readers.py"), broken
            )
        self.assertEqual(
            restored, ["readCamerasFromTransforms", "readNerfSyntheticInfo"]
        )
        compile(repaired, "dataset_readers.py", "exec")
        self.assertIn("def readNerfSyntheticInfo", repaired)

    def test_no_write_patch_matches_scaffold_and_standard_reader_shapes(self) -> None:
        prefix = (
            '    ply_path = os.path.join(path, "sparse/0/points3D.ply")\n'
            '    bin_path = os.path.join(path, "sparse/0/points3D.bin")\n'
            '    txt_path = os.path.join(path, "sparse/0/points3D.txt")\n'
            "    if not os.path.exists(ply_path):\n"
            "        storePly(ply_path, xyz, rgb)\n"
        )
        scaffold = prefix + "    pcd = fetchPly(ply_path)\n"
        standard = (
            prefix
            + "    try:\n"
            + "        pcd = fetchPly(ply_path)\n"
            + "    except:\n"
            + "        pcd = None\n"
        )
        for source in (scaffold, standard):
            patched, count = POINT_CLOUD_BLOCK_PATTERN.subn(
                NO_WRITE_POINT_CLOUD_BLOCK, source, count=1
            )
            self.assertEqual(count, 1)
            self.assertIn("fetchColmapPointCloud", patched)

    def test_ast_no_write_patch_preserves_pgsr_sparse_layout(self) -> None:
        source = (
            "def readColmapSceneInfo(path, images, eval):\n"
            "    ply_path = os.path.join(path, 'sparse/points3D.ply')\n"
            "    bin_path = os.path.join(path, 'sparse/points3D.bin')\n"
            "    txt_path = os.path.join(path, 'sparse/points3D.txt')\n"
            "    if not os.path.exists(ply_path) or True:\n"
            "        xyz, rgb, _ = read_points3D_binary(bin_path)\n"
            "        storePly(ply_path, xyz, rgb)\n"
            "    try:\n"
            "        pcd = fetchPly(ply_path)\n"
            "    except:\n"
            "        pcd = None\n"
            "    return pcd\n"
        )
        patched, count = replace_colmap_point_cloud_block(
            Path("dataset_readers.py"), source
        )
        self.assertEqual(count, 1)
        self.assertIn("'sparse/points3D.ply'", patched)
        self.assertIn("fetchColmapPointCloud(ply_path, bin_path, txt_path)", patched)
        self.assertNotIn("storePly(ply_path", patched)
        compile(patched, "dataset_readers.py", "exec")

    def test_ast_no_write_patch_handles_beta_init_type_branch(self) -> None:
        source = (
            "def readColmapSceneInfo(path, images, eval, init_type='sfm'):\n"
            "    if init_type == 'sfm':\n"
            "        ply_path = os.path.join(path, 'sparse/0/points3D.ply')\n"
            "        bin_path = os.path.join(path, 'sparse/0/points3D.bin')\n"
            "        txt_path = os.path.join(path, 'sparse/0/points3D.txt')\n"
            "        if not os.path.exists(ply_path):\n"
            "            xyz, rgb, _ = read_points3D_binary(bin_path)\n"
            "            storePly(ply_path, xyz, rgb)\n"
            "    elif init_type == 'random':\n"
            "        ply_path = os.path.join(path, 'random.ply')\n"
            "        pcd = BasicPointCloud()\n"
            "    else:\n"
            "        raise ValueError(init_type)\n"
            "    try:\n"
            "        pcd = fetchPly(ply_path)\n"
            "    except:\n"
            "        pcd = None\n"
            "    return pcd\n"
        )
        patched, count = replace_colmap_point_cloud_block(
            Path("Beta-Splatting/scene/dataset_readers.py"), source
        )
        self.assertEqual(count, 1)
        self.assertIn("fetchColmapPointCloud(ply_path, bin_path, txt_path)", patched)
        self.assertNotIn("pcd = fetchPly(ply_path)", patched)
        self.assertIn("pcd = BasicPointCloud()", patched)
        compile(patched, "dataset_readers.py", "exec")

    def test_beta_reader_patch_redirects_random_initialization(self) -> None:
        source = (
            "def fetchPly(path):\n    return None\n\n"
            "def storePly(path, xyz, rgb):\n    pass\n\n"
            "def readColmapSceneInfo(path, images, eval, init_type='sfm'):\n"
            "    nerf_normalization = {'radius': 1.0}\n"
            "    if init_type == \"sfm\":\n"
            "        ply_path = os.path.join(path, \"sparse/0/points3D.ply\")\n"
            "        bin_path = os.path.join(path, \"sparse/0/points3D.bin\")\n"
            "        txt_path = os.path.join(path, \"sparse/0/points3D.txt\")\n"
            "        if not os.path.exists(ply_path):\n"
            "            xyz, rgb, _ = read_points3D_binary(bin_path)\n"
            "            storePly(ply_path, xyz, rgb)\n"
            "    elif init_type == \"random\":\n"
            "        ply_path = os.path.join(path, \"random.ply\")\n"
            "        pcd = BasicPointCloud()\n"
            "        storePly(ply_path, xyz, rgb)\n"
            "    else:\n"
            "        raise ValueError(init_type)\n"
            "    try:\n"
            "        pcd = fetchPly(ply_path)\n"
            "    except:\n"
            "        pcd = None\n"
            "    return pcd\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            reader = (
                Path(directory)
                / "Beta-Splatting"
                / "scene"
                / "dataset_readers.py"
            )
            reader.parent.mkdir(parents=True)
            reader.write_text(source, encoding="utf-8")
            status = patch_dataset_reader_file(reader)
            patched = reader.read_text(encoding="utf-8")
        self.assertIn("mcmc_random_patched", status)
        self.assertIn('os.environ.get("UNIFIED3DGS_OUTPUT_PATH")', patched)
        self.assertNotIn('os.path.join(path, "random.ply")', patched)
        compile(patched, "dataset_readers.py", "exec")

    def test_reader_patch_repairs_partial_reader_idempotently(self) -> None:
        broken = (
            "def fetchPly(path):\n"
            "    return path\n\n"
            "def storePly(path, xyz, rgb):\n"
            "    pass\n\n"
            "def readColmapSceneInfo(path, images, eval):\n"
            "    ply_path = os.path.join(path, 'sparse/points3D.ply')\n"
            "    bin_path = os.path.join(path, 'sparse/points3D.bin')\n"
            "    txt_path = os.path.join(path, 'sparse/points3D.txt')\n"
            "    if not os.path.exists(ply_path):\n"
            "        xyz, rgb, _ = read_points3D_binary(bin_path)\n"
            "        storePly(ply_path, xyz, rgb)\n"
            "    pcd = fetchPly(ply_path)\n"
            "    return pcd\n\n"
            "sceneLoadTypeCallbacks = {\n"
            "    'Colmap': readColmapSceneInfo,\n"
            "    'Blender': readNerfSyntheticInfo,\n"
            "}\n"
        )
        pristine = broken.replace(
            "sceneLoadTypeCallbacks = {",
            "def readNerfSyntheticInfo():\n    return None\n\n"
            "sceneLoadTypeCallbacks = {",
        )
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / "dataset_readers.py"
            reader.write_text(broken, encoding="utf-8")
            with mock.patch(
                "scripts.patch_third_party_readers._pristine_reader_source",
                return_value=pristine,
            ):
                first_status = patch_dataset_reader_file(reader)
                first = reader.read_text(encoding="utf-8")
                second_status = patch_dataset_reader_file(reader)
                second = reader.read_text(encoding="utf-8")
            safety_errors = reader_safety_errors(reader, "example")
        self.assertIn("point_cloud_block_patched", first_status)
        self.assertIn("callbacks_restored=readNerfSyntheticInfo", first_status)
        self.assertIn("point_cloud_block_already_patched", second_status)
        self.assertEqual(first, second)
        self.assertEqual(safety_errors, [])

    def test_preflight_reader_repair_reports_every_selected_method(self) -> None:
        selected = [
            ({"key": "beta_splatting"}, {}),
            ({"key": "contextgs"}, {}),
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.check_method_preflight.patch_file",
            side_effect=["patched", RuntimeError("broken reader")],
        ):
            records, errors = repair_selected_readers(selected, Path(directory))
        self.assertEqual([record["method"] for record in records], [
            "beta_splatting",
            "contextgs",
        ])
        self.assertEqual(records[0]["status"], "patched")
        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(len(errors), 1)
        self.assertIn("contextgs", errors[0])

    @mock.patch(
        "unified3dgs_menu.query_gpu_status",
        return_value=(
            ["0", "1"],
            ["0, GPU, 24576, 24000, 576, 0, 30"],
            "",
        ),
    )
    @mock.patch("builtins.input", return_value="0")
    def test_gpu_zero_is_selectable(self, _input: mock.Mock, _query: mock.Mock) -> None:
        params = {}
        self.assertTrue(select_gpu(params))
        self.assertEqual(params["cuda_visible_devices"], "0")

    @mock.patch(
        "unified3dgs_menu.query_gpu_status",
        return_value=(["0"], ["0, GPU, 24576, 24000, 576, 0, 30"], ""),
    )
    @mock.patch("builtins.input", return_value="B")
    def test_gpu_back_uses_unambiguous_key(
        self, _input: mock.Mock, _query: mock.Mock
    ) -> None:
        self.assertFalse(select_gpu({}))

    def test_training_command_uses_preflight_runtime_python(self) -> None:
        result = MethodPreflight(
            key="example",
            title="Example",
            repo=Path("repo"),
            entry=Path("repo/train.py"),
            command_flags={
                "source": "--source_path",
                "output": "--model_path",
                "iterations": "--iterations",
                "save": "--save_iterations",
            },
            details={"runtime_python": "/opt/framework/bin/python"},
        )
        command = build_training_command(
            result,
            Path("dataset"),
            Path("output"),
            iterations=1,
        )
        self.assertEqual(command[0], "/opt/framework/bin/python")

    def test_signature_python_executable_allows_resolved_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "python3.7"
            alias = root / "python"
            target.write_text("", encoding="utf-8")
            try:
                alias.symlink_to(target.name)
            except (OSError, NotImplementedError):
                alias = target
            self.assertTrue(signature_value_matches("python_executable", alias, target))
            self.assertFalse(signature_value_matches("torch_cuda", "11.6", "11.8"))

    def test_training_source_mentions_flag_uses_entry_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.py"
            train.write_text(
                "parser.add_argument('--lmbda', default=0.004)\n",
                encoding="utf-8",
            )
            payload = {"entry": str(train)}
            self.assertTrue(training_source_mentions_flag(payload, "--lmbda"))
            self.assertFalse(training_source_mentions_flag(payload, "--not_real"))

    def test_hac_camera_transform_patch_is_idempotent(self) -> None:
        source = (
            "class Camera:\n"
            "    def __init__(self):\n"
            "        self.full_proj_transform = "
            "(self.world_view_transform.unsqueeze(0).bmm"
            "(self.projection_matrix.unsqueeze(0))).squeeze(0)\n"
        )
        patched, status = patch_hac_camera_text(source)
        self.assertEqual(status, "patched")
        self.assertIn(HAC_CAMERA_PATCH_MARKER, patched)
        self.assertIn("CUBLAS_STATUS_NOT_SUPPORTED", patched)
        patched_again, status_again = patch_hac_camera_text(patched)
        self.assertEqual(status_again, "already_patched")
        self.assertEqual(patched_again, patched)

    def test_hac_chunked_mlp_patch_is_idempotent(self) -> None:
        source = (
            "import torch\n\n"
            "def generate_neural_gaussians(viewpoint_camera, pc, visible_mask):\n"
            "    bank_weight = pc.get_featurebank_mlp(cat_view)\n"
            "    neural_opacity = pc.get_opacity_mlp(cat_local_view)\n"
            "    color = pc.get_color_mlp(cat_local_view)\n"
            "    scale_rot = pc.get_cov_mlp(cat_local_view)\n"
        )
        patched, status = patch_hac_chunked_mlp_text(source)
        self.assertEqual(status, "patched")
        self.assertIn(HAC_CHUNKED_MLP_PATCH_MARKER, patched)
        self.assertIn("_unified3dgs_chunked_mlp_forward(pc.get_opacity_mlp", patched)
        self.assertIn("_unified3dgs_chunked_mlp_forward(pc.get_color_mlp", patched)
        self.assertIn("_unified3dgs_chunked_mlp_forward(pc.get_cov_mlp", patched)
        self.assertIn("_unified3dgs_chunked_mlp_forward(pc.get_featurebank_mlp", patched)
        patched_again, status_again = patch_hac_chunked_mlp_text(patched)
        self.assertEqual(status_again, "already_patched")
        self.assertEqual(patched_again, patched)

    def test_current_profiles_use_real_upstream_sources(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dependencies = json.loads(
            (root / "configs" / "method_python_dependencies.json").read_text(
                encoding="utf-8"
            )
        )
        compact = next(item for item in dependencies if item["key"] == "compact_3dgs")
        compact_requirements = [
            package["requirement"] if isinstance(package, dict) else package
            for package in compact["packages"]
        ]
        self.assertTrue(
            any(
                "@v1.6#subdirectory=bindings/torch" in value
                for value in compact_requirements
            )
        )
        self.assertFalse(any("@v1.7" in value for value in compact_requirements))
        beta = next(item for item in dependencies if item["key"] == "beta_splatting")
        beta_requirements = [
            package["requirement"] if isinstance(package, dict) else package
            for package in beta["packages"]
        ]
        self.assertIn("kornia-rs==0.1.0", beta_requirements)
        self.assertIn("splines==0.3.3", beta_requirements)
        self.assertLess(
            beta_requirements.index("splines==0.3.3"),
            beta_requirements.index("nerfview==0.1.3"),
        )
        self.assertLess(
            beta_requirements.index("jaxtyping"),
            beta_requirements.index("nerfview==0.1.3"),
        )
        nerfview = next(
            package
            for package in beta["packages"]
            if isinstance(package, dict)
            and package.get("requirement") == "nerfview==0.1.3"
        )
        self.assertTrue(nerfview["install_dependencies"])
        self.assertNotIn("viser", beta_requirements)
        lapjv = next(
            package
            for package in beta["packages"]
            if isinstance(package, dict)
            and package.get("requirement") == "lapjv==1.3.27"
        )
        self.assertTrue(lapjv["no_build_isolation"])

        profiles = json.loads(
            (root / "configs" / "method_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        context = next(item for item in profiles if item["key"] == "contextgs")
        beta_profile = next(item for item in profiles if item["key"] == "beta_splatting")
        hac_profile = next(item for item in profiles if item["key"] == "hac_plus")
        mini_profile = next(item for item in profiles if item["key"] == "mini_splatting")
        octree_profile = next(item for item in profiles if item["key"] == "octree_gs")
        pgsr_profile = next(item for item in profiles if item["key"] == "pgsr")
        self.assertEqual(
            beta_profile["framework_render_entry"],
            "scripts/render_beta_splatting.py",
        )
        self.assertEqual(
            context["framework_render_entry"],
            "scripts/render_contextgs.py",
        )
        self.assertEqual(context["acceptance_resolution"], 256)
        self.assertTrue(context["stop_after_verified_save"])
        self.assertEqual(
            mini_profile["framework_render_entry"],
            "scripts/render_mini_splatting.py",
        )
        self.assertEqual(
            octree_profile["framework_render_entry"],
            "scripts/render_octree_gs.py",
        )
        self.assertEqual(
            octree_profile["framework_render_contracts"][0]["path"],
            "gaussian_renderer/__init__.py",
        )
        self.assertEqual(
            hac_profile["framework_render_entry"],
            "scripts/render_hac_plus.py",
        )
        self.assertTrue(pgsr_profile["colmap_sparse_zero_fallback"])
        self.assertTrue(beta_profile["stop_after_verified_save"])
        self.assertTrue(beta_profile["stop_after_verified_save_reason"])
        self.assertEqual(hac_profile["extra_args"], [])
        self.assertEqual(
            hac_profile["official_dataset_args"]["tandt"][
                hac_profile["official_dataset_args"]["tandt"].index("--voxel_size"):
                hac_profile["official_dataset_args"]["tandt"].index("--voxel_size") + 2
            ],
            ["--voxel_size", "0.01"],
        )
        self.assertEqual(
            hac_profile["official_backend"]["requirements"]["torch"],
            "1.12.1",
        )
        self.assertNotIn("--n_offsets", hac_profile["extra_args"])
        self.assertTrue(hac_profile["stop_after_verified_save"])
        self.assertTrue(hac_profile["stop_after_verified_save_reason"])
        self.assertNotIn("archived_extensions", context)
        self.assertEqual(len(context["external_extensions"]), 2)
        static_only = {
            item["key"]: item
            for item in profiles
            if item.get("static_help_only") is True
        }
        self.assertEqual(
            set(static_only),
            {"3dcs", "contextgs", "hac_plus", "octree_gs", "scaffold_gs", "wavelet_gs"},
        )
        self.assertTrue(
            all(item.get("static_help_reason") for item in static_only.values())
        )
        self.assertTrue(
            {
                "beta_splatting",
                "compact_3dgs",
                "ges",
                "lightgaussian",
            }.issubset(PATCH_READER_REPOS)
        )
        ges = next(item for item in profiles if item["key"] == "ges")
        self.assertIn("--nowandb", ges["extra_args"])
        self.assertEqual(ges["output_globs"], ["{output}_*"])

    def test_hac_setup_pins_torch_compatible_mkl(self) -> None:
        setup = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_hac_plus_official_backend.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"mkl<2024.1"', setup)
        self.assertIn('"intel-openmp<2024.1"', setup)
        self.assertIn("probe_torch", setup)
        self.assertLess(
            setup.index("Verifying the official PyTorch runtime"),
            setup.index("einops==0.6.1"),
        )
        self.assertIn(
            'OFFICIAL_CUDA_HOME="${UNIFIED3DGS_HAC_PLUS_CUDA_HOME:-/usr/local/cuda-11.6}"',
            setup,
        )
        self.assertNotIn('CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"', setup)
        self.assertLess(
            setup.index("import torch\nimport arithmetic"),
            setup.index('print("arithmetic:"'),
        )
        self.assertIn("import diff_gaussian_rasterization._C", setup)
        self.assertIn("import simple_knn._C", setup)
        self.assertNotIn('print("simple_knn:", simple_knn.__file__)', setup)
        self.assertIn("extension CUDA_HOME mismatch", setup)
        self.assertIn("--override-channels", setup)
        self.assertLess(setup.index("--override-channels"), setup.index("-c pytorch"))

    def test_complete_configuration_audit_passes(self) -> None:
        audit = audit_configuration()
        self.assertTrue(audit["passed"], audit["errors"])
        self.assertEqual(audit["confirmed_method_count"], len(load_confirmed_catalog()))

    def test_failure_summary_includes_dependency_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "dependency.log"
            log.write_text("first\nroot cause\n", encoding="utf-8")
            (root / "python_dependency_report.json").write_text(
                json.dumps(
                    [
                        {
                            "method": "example",
                            "status": "failed",
                            "error": "install failed",
                            "installs": [
                                {
                                    "package": "example-package",
                                    "status": "failed",
                                    "log": str(log),
                                    "import_error": "missing transitive dependency",
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            summary = collect_failures(root)
            self.assertEqual(summary["failure_count"], 1)
            self.assertIn(
                "root cause", summary["failures"][0]["items"][0]["log_tail"]
            )
            self.assertEqual(
                summary["failures"][0]["items"][0]["import_error"],
                "missing transitive dependency",
            )


if __name__ == "__main__":
    unittest.main()
