"""Environment curriculum parameter generation mixin for PPO training."""

from ppo_train.config.runtime_config import (
    CURRICULUM_DENSITY_RANGES,
    CURRICULUM_STAGE_EDGES,
)



class EnvCurriculumMixin:
    """Mixin holding environment curriculum parameter generation methods."""

    _CURRICULUM_STAGE_EDGES = CURRICULUM_STAGE_EDGES

    def _current_curriculum_stage(self):
        """Return the active curriculum stage for the configured mode."""
        if self.curriculum_mode == "gate":
            return self.curriculum_stage

        p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))
        if p < 0.02:
            return 0
        if p < 0.05:
            return 1
        if p < 0.1:
            return 2
        return 3

    def _curriculum_density(self):
        """Compute curriculum traffic density from training progress or gate stage."""
        # Normalized progress p in [0, 1]
        p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))

        stage = self._current_curriculum_stage()
        if self.curriculum_mode == "progress":
            seg_edges = self._CURRICULUM_STAGE_EDGES
            seg_p = (p - seg_edges[stage]) / max(1e-8, (seg_edges[stage + 1] - seg_edges[stage]))
            seg_p = max(0.0, min(1.0, seg_p)) ** self.curriculum_alpha
        else:
            seg_p = 1.0

        d0, d1 = CURRICULUM_DENSITY_RANGES[stage]
        return d0 + (d1 - d0) * seg_p
