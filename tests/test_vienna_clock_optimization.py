import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pipelines import vienna_T2 as pipeline
from src.preprocessing.physiological_data.PolarisInfraredTrackerReader import PolarisInfraredTrackerReader
from src.utils.clock_optimization import optimize_clock_correction
from src.utils.sharpness_index import sharpness_index


class SharpnessTests(unittest.TestCase):
    def test_article_reference_values(self):
        # Reference values from origin/article:article/sharpness_index.py.
        generator = torch.Generator().manual_seed(123)
        image = torch.rand((32, 32), generator=generator, dtype=torch.float64)
        for mode, expected in enumerate([
                0.0957572552225209, 0.13420501230763304,
                0.19481042957736291, 0.2172372577133482]):
            self.assertAlmostEqual(float(sharpness_index(image, mode)), expected, places=6)

    def test_constant_complex_and_scaling(self):
        self.assertEqual(float(sharpness_index(torch.ones(8, 8))), 0)
        image = torch.rand((17, 24), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        score = sharpness_index(image)
        self.assertGreaterEqual(float(score), 0)
        torch.testing.assert_close(sharpness_index(3 * image), score)
        torch.testing.assert_close(sharpness_index(image.to(torch.complex128) * 1j), score)

    def test_invalid_inputs(self):
        for image in (torch.ones(3, 4, 5), torch.ones(1, 8), torch.full((8, 8), float('nan'))):
            with self.assertRaises(ValueError):
                sharpness_index(image)


class OptimizationTests(unittest.TestCase):
    def test_maximize_positive_sharpness(self):
        calls = []
        def objective(x):
            calls.append(x)
            return -(10 - (x - 0.37)**2)
        result = optimize_clock_correction(objective, learning_rate=0.5, max_iterations=20)
        self.assertAlmostEqual(result.correction_seconds, 0.37, places=6)
        self.assertAlmostEqual(-result.objective, 10)
        self.assertEqual(len(calls), len(set(calls)))
        accepted_scores = [objective(x) for x in result.accepted_corrections]
        self.assertTrue(all(b < a for a, b in zip(accepted_scores, accepted_scores[1:])))

    def test_boundary_optimum_and_no_out_of_bounds_evaluations(self):
        result = optimize_clock_correction(lambda x: -x, bounds=(-0.2, 0.3), max_iterations=20)
        self.assertEqual(result.correction_seconds, 0.3)
        self.assertTrue(all(-0.2 <= row['correction_seconds'] <= 0.3 for row in result.evaluations))

    def test_backtracking_and_best_probe(self):
        result = optimize_clock_correction(lambda x: (x - 0.03)**2,
            learning_rate=100, max_step=1, max_iterations=15)
        self.assertLess(result.objective, 1e-6)
        self.assertEqual(result.objective, min(row['objective'] for row in result.evaluations))

    def test_plateau_stops_and_invalid_scores_raise(self):
        result = optimize_clock_correction(lambda _: -5.0)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.correction_seconds, 0)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            optimize_clock_correction(lambda _: float('nan'))
        with self.assertRaises(ValueError):
            optimize_clock_correction(lambda _: 0, initial=2)


def fixture_data(root):
    times = np.linspace(-5, 0, 101)
    acquisition = np.linspace(-4.9, 0, 100)
    sync = dict(physiological_time_seconds=[times] * 3,
                physiological_values=[np.sin(times), np.cos(times), 2 * np.sin(times)],
                acquisition_time_seconds=acquisition, slice_indices=np.tile([0, 1], 50))
    outputs = SimpleNamespace(manifest={}, flush=Mock(), snapshot=Mock())
    data = SimpleNamespace(Nz=2, t_device='cpu', params=SimpleNamespace(
        physio_clock_drift_seconds=0.0, run_folder=str(root), _run_outputs=outputs),
        raw_data_preparer=SimpleNamespace(synchronization=sync,
            physiological_reader=PolarisInfraredTrackerReader('largest-amplitude')))
    return data, sync


class WorkerFixture:
    """Minimal preprocessing fixture; workers still execute scoring and output code."""
    def __init__(self, root):
        self.Nz = 3
        self.params = SimpleNamespace(seed=7, run_folder=str(root), ResolutionLevels=[1.0],
                                      physio_clock_drift_seconds=0.25)
        self.zero_fill_shapes = ((8, 8), (8, 8))
        self.export_series_number = 1001
        self.export_reference = None
        self.export_uids = {}

    def run_slice_pipeline(self, slice_idx, *, output_folder=None):
        # Reproduce real slice selection's output binding to catch routing regressions.
        pipeline.bind_output_paths(self.params, output_folder if output_folder is not None else
            Path(self.params.run_folder) / 'reconstructions' / f'slice_{slice_idx + 1:03d}')
        self.kspace = torch.rand((1, 8, 8), dtype=torch.float64) + slice_idx
        self.smaps = self.sampling_idx = self.motion_signal = None
        self.kspace_scale = 1
        self.image_no_moco = self.kspace.clone()
        self.motion_plot_context = None


class WorkerReconstructor:
    def __init__(self, kspace, *args, **kwargs):
        self.image = kspace
        if not kwargs['params'].save_reconstruction_outputs:
            raise AssertionError('Every clock evaluation must enable reconstruction image saving.')

    def run(self):
        return self.image, None


def worker_dicom_fixture(image, output_path, **kwargs):
    # Verify the export branch without needing scanner metadata in this fixture.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(kwargs['slice_index']))


class PipelineTests(unittest.TestCase):
    def test_real_slice_selection_honors_custom_output_folder(self):
        from src.preprocessing.DataLoader import DataLoader
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = DataLoader.__new__(DataLoader)
            data.params = SimpleNamespace(data_dimension='2D', kspace_sampling_type='from-data',
                                          run_folder=str(root), ResolutionLevels=[1.0])
            data.filename = ('raw.mrd', 'motion.tsv')
            data._source_kspace = torch.zeros((1, 1, 8, 8, 2), dtype=torch.complex128)
            data._source_reference_kspace = None
            data._source_motion_data = torch.ones((2, 8, 1))
            data._source_idx_ky = torch.arange(8).repeat(2, 1)
            data._source_idx_nex = torch.zeros((2, 8), dtype=torch.int64)
            data._configure_realworld_motion_inputs = Mock()
            folder = root / 'baselines/zero_clock/slice_002'
            data._select_loaded_slice(1, output_folder=folder)
            self.assertEqual(Path(data.params.results_folder), folder / 'results')
            self.assertTrue((folder / '.metadata.json').is_file())
            self.assertFalse((root / 'reconstructions').exists())

    def test_bounds_and_alignment_do_not_accumulate_shifts(self):
        data, sync = fixture_data('.')
        lower, upper = pipeline.clock_bounds(sync, 0, (-2, 2))
        self.assertAlmostEqual(lower, -1)
        self.assertAlmostEqual(upper, 1.1)
        pipeline.apply_clock_correction(data, sync, 0, -0.3)
        motion = data._source_motion_data.clone()
        pipeline.apply_clock_correction(data, sync, 0, 0.2)
        pipeline.apply_clock_correction(data, sync, 0, -0.3)
        torch.testing.assert_close(data._source_motion_data, motion)
        self.assertEqual(tuple(motion.shape), (2, 50, 1))
        np.testing.assert_array_equal(sync['physiological_time_seconds'][0], np.linspace(-5, 0, 101))
        np.testing.assert_allclose(data.raw_data_preparer.synchronization['physiological_time_seconds'][0],
                                   sync['physiological_time_seconds'][0] - 0.3)

    def test_fork_workers_repeat_scores_and_export_all_slices(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = WorkerFixture(root)
            with patch.object(pipeline, 'JointReconstructor', WorkerReconstructor), \
                 patch.object(pipeline, 'write_reconstruction_dicom', worker_dicom_fixture), \
                 patch.object(pipeline, 'ProcessPoolExecutor', wraps=pipeline.ProcessPoolExecutor) as pool:
                trial = pipeline.reconstruct_all_slices(data, root / 'trial', None)
                self.assertEqual(pool.call_args.kwargs['max_workers'], data.Nz)
                final = pipeline.reconstruct_all_slices(data, root / 'final', 2, final=True)
                self.assertEqual(pool.call_args.kwargs['max_workers'], 2)
            self.assertEqual([row['slice_idx'] for row in final], [0, 1, 2])
            self.assertEqual([row['sharpness_index'] for row in trial],
                             [row['sharpness_index'] for row in final])
            for row in final:
                self.assertTrue((root / row['dicom_file']).is_file())
                self.assertTrue((root / 'final' / f"slice_{row['slice_number']:03d}" / '.metadata.json').is_file())
            self.assertFalse(hasattr(data.params, 'reconstruction_folder'))

    def test_comparison_images_and_shared_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = torch.rand((2, 8, 8), dtype=torch.float64)
            optimized = root / 'reconstructions/slice_001/results'
            baseline = root / 'baselines/zero_clock/slice_001/results'
            optimized.mkdir(parents=True)
            baseline.mkdir(parents=True)
            torch.save(image, optimized / 'image_reconstructed.pt')
            torch.save(image * 0.9, baseline / 'image_reconstructed.pt')
            torch.save(image * 0.8, baseline / 'image_no_motion_correction.pt')
            rows = pipeline.save_comparisons(root, 1, 0.25)
            self.assertTrue((root / rows[0]['image']).is_file())
            images = torch.load(root / 'comparisons/slice_001/images.pt', weights_only=True)
            self.assertEqual(len(images), 3)
            self.assertEqual(len(rows[0]['sharpness']), 3)
            self.assertGreater(rows[0]['display_window'][1], 0)

    def test_disable_clock_optimization_uses_fixed_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, motion = root / 'raw.mrd', root / 'R1.tsv'
            raw.touch()
            motion.touch()
            for correction in (0.0, 0.2):
                data, _ = fixture_data(root / str(correction))
                data.params.physio_clock_drift_seconds = correction
                args = [str(raw), str(motion), '--no-clock-optimization']
                if correction:
                    args += ['--initial-correction', str(correction)]
                calls = []

                def reconstruct(data, folder, workers, *, final=False):
                    calls.append((data.params.physio_clock_drift_seconds, final))
                    return [dict(slice_idx=i, sharpness_index=1.0) for i in range(data.Nz)]

                with patch.object(pipeline, 'load_all_slices', return_value=data), \
                     patch.object(pipeline, 'optimize_clock_correction', side_effect=AssertionError('must skip')), \
                     patch.object(pipeline, 'clock_bounds', side_effect=AssertionError('must skip')), \
                     patch.object(pipeline, 'reconstruct_all_slices', side_effect=reconstruct), \
                     patch.object(pipeline, 'grics_zero_fill_shapes', return_value=((8, 8), (8, 8))), \
                     patch.object(pipeline, 'save_comparisons', return_value=[]):
                    pipeline.main(args)
                self.assertEqual(calls, [(correction, True), (0.0, False)])
                self.assertFalse(data.params._run_outputs.manifest['clock_optimization_enabled'])
                self.assertEqual(data.params.physio_clock_drift_seconds, correction)

    def test_reuse_latest_skips_optimizer_and_preserves_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / 'runs/20260101T120000/optimization'
            old.mkdir(parents=True)
            (old / 'result.json').write_text(json.dumps({'correction_seconds': -0.3}))
            raw, motion = root / 'raw.mrd', root / 'R1.tsv'
            raw.touch()
            motion.touch()
            args = [str(raw), str(motion), '--output-root', str(root / 'runs'),
                    '--reuse-clock-run', 'latest']
            parsed = pipeline.parse_args(args)
            self.assertTrue(parsed.flip_for_display)
            self.assertEqual(parsed.initial_correction, -0.3)
            self.assertFalse(pipeline.parse_args(args + ['--no-flip-for-display']).flip_for_display)
            data, _ = fixture_data(root / 'new')
            data.params.flip_for_display = True
            seen = []

            def reconstruct(data, folder, workers, *, final=False):
                seen.append((data.params.physio_clock_drift_seconds, final))
                return [dict(slice_idx=i, sharpness_index=1.0) for i in range(data.Nz)]

            with patch.object(pipeline, 'load_all_slices', return_value=data), \
                 patch.object(pipeline, 'optimize_clock_correction', side_effect=AssertionError('must skip')), \
                 patch.object(pipeline, 'reconstruct_all_slices', side_effect=reconstruct), \
                 patch.object(pipeline, 'grics_zero_fill_shapes', return_value=((8, 8), (8, 8))), \
                 patch.object(pipeline, 'save_comparisons', return_value=[]) as compare:
                pipeline.main(args)
            self.assertEqual(seen, [(-0.3, True), (0.0, False)])
            self.assertEqual(data.params.physio_clock_drift_seconds, -0.3)
            self.assertTrue(compare.call_args.kwargs['flip_for_display'])
            saved = json.loads((root / 'new/optimization/result.json').read_text())
            self.assertEqual(saved['reused_from'], str(old.parent.resolve()))

    def test_orchestration_scores_every_slice_and_exports_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, motion_file = root / 'raw.mrd', root / 'R1.tsv'
            raw.touch()
            motion_file.touch()
            data, sync = fixture_data(root)
            evaluations = []

            def reconstruct(data, folder, workers, *, final=False):
                x = data.params.physio_clock_drift_seconds
                evaluations.append((x, final))
                return [dict(slice_idx=i, slice_number=i + 1, sharpness_index=10 + i - (x - 0.35)**2)
                        for i in range(data.Nz)]

            with patch.object(pipeline, 'load_all_slices', return_value=data), \
                 patch.object(pipeline, 'reconstruct_all_slices', side_effect=reconstruct), \
                 patch.object(pipeline, 'grics_zero_fill_shapes', return_value=((8, 8), (8, 8))), \
                 patch.object(pipeline, 'save_comparisons', return_value=[]) as compare:
                pipeline.main([str(raw), str(motion_file), '--learning-rate', '0.5', '--max-workers', '1'])
            summary = json.loads((root / 'optimization/result.json').read_text())
            history = json.loads((root / 'optimization/history.json').read_text())
            final = json.loads((root / 'sharpness.json').read_text())
            self.assertAlmostEqual(summary['correction_seconds'], 0.35, places=6)
            self.assertEqual(evaluations[-2], (summary['correction_seconds'], True))
            self.assertEqual(evaluations[-1], (0.0, False))
            compare.assert_called_once()
            self.assertTrue(all(len(trial['slice_results']) == 2 for trial in history))
            self.assertEqual(len(final), 2)
            self.assertAlmostEqual(summary['mean_sharpness_index'], 10.5)
            self.assertEqual(data.params.physio_clock_drift_seconds, summary['correction_seconds'])


if __name__ == '__main__':
    unittest.main()
