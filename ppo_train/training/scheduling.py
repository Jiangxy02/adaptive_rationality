"""Curriculum and scalar scheduling mixin for PPO expert training."""



class SchedulingMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    def _compute_scheduled_lr(self):
        # Progress p in [0, 1].
        p = min(1.0, float(self.global_step) / float(self.args.total_timesteps + 1e-8))
        # warmup
        if self.warmup_ratio > 0.0:
            wu = self.warmup_ratio
            if p < wu:
                # Linearly warm up from 0 to lr_init.
                return self.lr_init * (p / wu)
            # Renormalize progress after removing the warmup span.
            p = (p - wu) / max(1e-8, (1.0 - wu))
            p = max(0.0, min(1.0, p))

        if self.lr_schedule == "constant":
            return self.lr_init

        if self.lr_schedule == "linear":
            # Linearly decay from lr_init to lr_min, reaching lr_min at 80% progress.
            end_ratio = 0.8
            q = min(1.0, p / end_ratio)
            return self.lr_init + (self.lr_min - self.lr_init) * q

        if self.lr_schedule == "cosine":
            # Cosine annealing from lr_init to lr_min.
            import math
            cos_term = 0.5 * (1 + math.cos(math.pi * p))
            return self.lr_min + (self.lr_init - self.lr_min) * cos_term

        if self.lr_schedule == "stage":
            stage = self._current_curriculum_stage()
            if stage == 0:
                return self.lr_init
            elif stage == 1:
                return max(self.lr_min, self.lr_init * 0.7)
            elif stage == 2:
                return max(self.lr_min, self.lr_init * 0.5)
            else:  # stage 3
                return max(self.lr_min, self.lr_init * 0.3)

        # Fallback.
        return self.lr_init


    def _update_entropy_coef(self):
        """Update the entropy coefficient with linear decay."""
        # Compute training progress.
        progress = self.global_step / self.args.total_timesteps

        if progress <= self.entropy_decay_end_ratio:
            # Linearly interpolate within the decay window.
            decay_progress = progress / self.entropy_decay_end_ratio
            self.current_entropy_coef = (
                self.entropy_coef_start -
                (self.entropy_coef_start - self.entropy_coef_end) * decay_progress
            )
        else:
            # Keep the final value after the decay window ends.
            self.current_entropy_coef = self.entropy_coef_end

        return self.current_entropy_coef
