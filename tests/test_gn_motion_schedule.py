"""Regression tests for the GRICS++ Gauss-Newton update schedule."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch

from src.reconstruction.JointReconstructor import JointReconstructor


class _Logger:
    @contextmanager
    def iterations(self, level_index):
        yield range(3)

    def record_residual(self, value):
        pass

    def iteration_finished(self, *args, **kwargs):
        pass

    def iteration_stopped_early(self):
        raise AssertionError("constant residual must not stop the level")


class GnMotionScheduleTests(unittest.TestCase):
    def _run_level(self, level_index, update_final_motion=False):
        recon = object.__new__(JointReconstructor)
        recon.params = SimpleNamespace(lambda_r=0.0, ResolutionLevels=[0.5, 1.0])
        recon._current_level_idx = level_index
        recon.external_image_regularizer = None
        recon._last_image_cg_info = None
        recon._last_motion_cg_info = None
        decisions = []

        def iteration(data, **kwargs):
            decisions.append(kwargs["update_motion"])
            image = data["ReconstructedImage"] + 1
            prior_motion = data["MotionModel"].clone()
            motion = prior_motion + (1 if kwargs["update_motion"] else 0)
            data["ReconstructedImage"] = image
            data["MotionModel"] = motion
            return SimpleNamespace(
                image=image, motion=motion, motion_for_residual=prior_motion,
                residual=torch.zeros(1), motion_update=None,
            )

        recon.gauss_newton_iteration = iteration
        data = {
            "KspaceData": torch.ones(1),
            "ReconstructedImage": torch.zeros(1),
            "MotionModel": torch.zeros(1),
        }
        image, motion = recon._run_resolution_level(
            data, level_index=level_index, gauss_newton_iterations_at_level=3,
            level_count=2, update_final_motion=update_final_motion,
            gn_early_stopping=True, logger=_Logger(),
        )
        return decisions, image, motion

    def test_each_level_ends_with_image_only_iteration(self):
        for level_index in (0, 1):
            with self.subTest(level_index=level_index):
                decisions, image, motion = self._run_level(level_index)
                self.assertEqual(decisions, [True, True, False])
                torch.testing.assert_close(image, torch.tensor([3.0]))
                torch.testing.assert_close(motion, torch.tensor([2.0]))

    def test_explicit_final_motion_override_is_preserved(self):
        decisions, _, motion = self._run_level(0, update_final_motion=True)
        self.assertEqual(decisions, [True, True, True])
        # The accepted image was formed with the motion before the final update.
        torch.testing.assert_close(motion, torch.tensor([2.0]))


if __name__ == "__main__":
    unittest.main()
