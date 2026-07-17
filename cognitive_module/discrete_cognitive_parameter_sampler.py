
import numpy as np
import itertools
import random
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import Dict, Any, Tuple, List, Optional

_DISCRETE_SAMPLER_UPDATE_STEPS_DEFAULT = 5
_DISCRETE_SAMPLER_BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT = (0.0, 10.0)
_DISCRETE_SAMPLER_BIAS_INVERSE_TTA_COEF_DENSITY_DEFAULT = 4
_DISCRETE_SAMPLER_PERCEPTION_SIGMA0_RANGE_DEFAULT = (0.0, 1.0)
_DISCRETE_SAMPLER_PERCEPTION_SIGMA0_DENSITY_DEFAULT = 4
_DISCRETE_SAMPLER_PERCEPTION_SIGMA_MAX_RANGE_DEFAULT = (0.0, 5.0)
_DISCRETE_SAMPLER_PERCEPTION_SIGMA_MAX_DENSITY_DEFAULT = 4
_DISCRETE_SAMPLER_DELAY_STEPS_RANGE_DEFAULT = (0, 20)
_DISCRETE_SAMPLER_DELAY_STEPS_DENSITY_DEFAULT = 3
_DISCRETE_SAMPLER_SHUFFLE_DEFAULT = True
_DISCRETE_SAMPLER_ENABLE_VISUALIZATION_DEFAULT = True
_DISCRETE_SAMPLER_SAVE_HISTORY_DEFAULT = True


class DiscreteCognitiveParameterSampler:
    """Enumerate a cognitive-parameter grid while dropping invalid sigma pairs."""

    def __init__(self,
                 update_steps: int = _DISCRETE_SAMPLER_UPDATE_STEPS_DEFAULT,
                 bias_inverse_tta_coef_range: Tuple[float, float] = _DISCRETE_SAMPLER_BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT,
                 bias_inverse_tta_coef_density: int = _DISCRETE_SAMPLER_BIAS_INVERSE_TTA_COEF_DENSITY_DEFAULT,
                 perception_sigma0_range: Tuple[float, float] = _DISCRETE_SAMPLER_PERCEPTION_SIGMA0_RANGE_DEFAULT,
                 perception_sigma0_density: int = _DISCRETE_SAMPLER_PERCEPTION_SIGMA0_DENSITY_DEFAULT,
                 perception_sigma_max_range: Tuple[float, float] = _DISCRETE_SAMPLER_PERCEPTION_SIGMA_MAX_RANGE_DEFAULT,
                 perception_sigma_max_density: int = _DISCRETE_SAMPLER_PERCEPTION_SIGMA_MAX_DENSITY_DEFAULT,
                 delay_steps_range: Tuple[int, int] = _DISCRETE_SAMPLER_DELAY_STEPS_RANGE_DEFAULT,
                 delay_steps_density: int = _DISCRETE_SAMPLER_DELAY_STEPS_DENSITY_DEFAULT,
                 shuffle: bool = _DISCRETE_SAMPLER_SHUFFLE_DEFAULT,
                 enable_visualization: bool = _DISCRETE_SAMPLER_ENABLE_VISUALIZATION_DEFAULT,
                 save_history: bool = _DISCRETE_SAMPLER_SAVE_HISTORY_DEFAULT,
                 seed: Optional[int] = None):

        self.update_steps = update_steps


        self.bias_inverse_tta_coef_range = bias_inverse_tta_coef_range
        self.perception_sigma0_range = perception_sigma0_range
        self.perception_sigma_max_range = perception_sigma_max_range
        self.delay_steps_range = delay_steps_range


        self.bias_inverse_tta_coef_density = bias_inverse_tta_coef_density
        self.perception_sigma0_density = perception_sigma0_density
        self.perception_sigma_max_density = perception_sigma_max_density
        self.delay_steps_density = delay_steps_density


        self.shuffle = shuffle
        self.enable_visualization = enable_visualization
        self.save_history = save_history
        self.seed = seed
        self._rng = random.Random(seed)
        self.bias_inverse_tta_coef_values = np.linspace(*bias_inverse_tta_coef_range, bias_inverse_tta_coef_density)
        self.perception_sigma0_values = np.linspace(*perception_sigma0_range, perception_sigma0_density)
        self.perception_sigma_max_values = np.linspace(*perception_sigma_max_range, perception_sigma_max_density)

        self.delay_steps_values = np.arange(delay_steps_range[0], delay_steps_range[1] + 1, dtype=int)


        self.param_grid: List[Dict[str, Any]] = []
        for bias, sigma0, sigma_max, delay in itertools.product(
            self.bias_inverse_tta_coef_values,
            self.perception_sigma0_values,
            self.perception_sigma_max_values,
            self.delay_steps_values
        ):
            if sigma0 > sigma_max:
                continue
            self.param_grid.append({
                'bias_inverse_tta_coef': float(bias),
                'perception_sigma0': float(sigma0),
                'perception_sigma_max': float(sigma_max),
                'delay_steps': int(delay)
            })

        if not self.param_grid:
            raise ValueError("The perception noise sampling ranges contain no valid combination: perception_sigma0 must be <= perception_sigma_max")


        if shuffle:
            self._rng.shuffle(self.param_grid)

        self.index = 0


        self.current_params = self.param_grid[0].copy() if self.param_grid else {}


        self.param_history = []
        self.step_history = []
        self.timestamp_history = []


        self._last_update_step = 0
        self._total_updates = 0


        if self.save_history:
            self._record_parameter_update(0, "initialization")

    def sample(self) -> Dict[str, Any]:
        return self._rng.choice(self.param_grid)

    def next(self) -> Dict[str, Any]:
        if self.index >= len(self.param_grid):
            self.index = 0
        params = self.param_grid[self.index]
        self.index += 1
        return params

    def all_combinations(self) -> List[Dict[str, Any]]:
        return self.param_grid

    def size(self) -> int:
        return len(self.param_grid)

    def should_update_parameters(self, current_step: int) -> bool:
        return current_step - self._last_update_step >= self.update_steps

    def update_parameters(self, current_step: int, force_update: bool = False) -> Dict[str, Any]:
        if not force_update and not self.should_update_parameters(current_step):
            return self.current_params


        self.current_params = self.sample()
        self._last_update_step = current_step
        self._total_updates += 1


        if self.save_history:
            self._record_parameter_update(current_step, "sampling")

        return self.current_params

    def _record_parameter_update(self, step: int, update_type: str):
        from datetime import datetime
        timestamp = datetime.now().isoformat()

        self.param_history.append(self.current_params.copy())
        self.step_history.append(step)
        self.timestamp_history.append(timestamp)

    def get_current_parameters(self) -> Dict[str, Any]:
        return self.current_params.copy()

    def _state_config(self) -> Dict[str, Any]:
        """Return the configuration that must match for an exact resume."""
        return {
            'update_steps': self.update_steps,
            'bias_inverse_tta_coef_range': tuple(self.bias_inverse_tta_coef_range),
            'bias_inverse_tta_coef_density': self.bias_inverse_tta_coef_density,
            'perception_sigma0_range': tuple(self.perception_sigma0_range),
            'perception_sigma0_density': self.perception_sigma0_density,
            'perception_sigma_max_range': tuple(self.perception_sigma_max_range),
            'perception_sigma_max_density': self.perception_sigma_max_density,
            'delay_steps_range': tuple(self.delay_steps_range),
            'delay_steps_density': self.delay_steps_density,
            'shuffle': self.shuffle,
            'enable_visualization': self.enable_visualization,
            'save_history': self.save_history,
            'seed': self.seed,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Capture all sampler-owned state needed for an exact resume."""
        return {
            'state_version': 2,
            'sampler_type': 'discrete',
            'config': deepcopy(self._state_config()),
            'rng_state': deepcopy(self._rng.getstate()),
            'param_grid': deepcopy(self.param_grid),
            'index': self.index,
            'current_params': deepcopy(self.current_params),
            'last_update_step': self._last_update_step,
            'total_updates': self._total_updates,
            'param_history': deepcopy(self.param_history),
            'step_history': deepcopy(self.step_history),
            'timestamp_history': deepcopy(self.timestamp_history),
        }

    @staticmethod
    def _parameter_key(params: Mapping) -> Tuple[float, float, float, int]:
        return (
            float(params['bias_inverse_tta_coef']),
            float(params['perception_sigma0']),
            float(params['perception_sigma_max']),
            int(params['delay_steps']),
        )

    def _configured_parameter_keys(self) -> Counter:
        """Rebuild the configured grid independently of mutable runtime order."""
        biases = np.linspace(
            *self.bias_inverse_tta_coef_range, self.bias_inverse_tta_coef_density
        )
        sigma0_values = np.linspace(
            *self.perception_sigma0_range, self.perception_sigma0_density
        )
        sigma_max_values = np.linspace(
            *self.perception_sigma_max_range, self.perception_sigma_max_density
        )
        delays = np.arange(
            self.delay_steps_range[0], self.delay_steps_range[1] + 1, dtype=int
        )
        return Counter(
            (float(bias), float(sigma0), float(sigma_max), int(delay))
            for bias, sigma0, sigma_max, delay in itertools.product(
                biases, sigma0_values, sigma_max_values, delays
            )
            if sigma0 <= sigma_max
        )

    def _validate_state_parameters(self, params: Any, field: str) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            raise ValueError(f"{field} must be a parameter mapping")

        expected_keys = {
            'bias_inverse_tta_coef',
            'perception_sigma0',
            'perception_sigma_max',
            'delay_steps',
        }
        if set(params) != expected_keys:
            raise ValueError(f"{field} has invalid parameter keys")

        bias = params['bias_inverse_tta_coef']
        sigma0 = params['perception_sigma0']
        sigma_max = params['perception_sigma_max']
        delay = params['delay_steps']
        for name, value in (
            ('bias_inverse_tta_coef', bias),
            ('perception_sigma0', sigma0),
            ('perception_sigma_max', sigma_max),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{field}.{name} must be a finite real number")
        if isinstance(delay, bool) or not isinstance(delay, Integral):
            raise ValueError(f"{field}.delay_steps must be an integer")
        if sigma0 > sigma_max:
            raise ValueError(f"{field} requires perception_sigma0 <= perception_sigma_max")
        return deepcopy(dict(params))

    def load_state_dict(self, state: Mapping) -> None:
        """Restore an exact-resume state after validating it without mutation."""
        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")

        expected_fields = {
            'state_version',
            'sampler_type',
            'config',
            'rng_state',
            'param_grid',
            'index',
            'current_params',
            'last_update_step',
            'total_updates',
            'param_history',
            'step_history',
            'timestamp_history',
        }
        if set(state) != expected_fields:
            raise ValueError("sampler state has missing or unexpected fields")
        if state['state_version'] != 2 or state['sampler_type'] != 'discrete':
            raise ValueError("unsupported discrete sampler state")

        config = state['config']
        if not isinstance(config, Mapping) or dict(config) != self._state_config():
            raise ValueError("discrete sampler configuration mismatch")

        try:
            validated_rng = random.Random()
            validated_rng.setstate(deepcopy(state['rng_state']))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid discrete sampler RNG state") from exc

        param_grid = state['param_grid']
        if not isinstance(param_grid, list) or not param_grid:
            raise ValueError("param_grid must be a non-empty list")
        validated_grid = [
            self._validate_state_parameters(params, f'param_grid[{index}]')
            for index, params in enumerate(param_grid)
        ]
        expected_grid = self._configured_parameter_keys()
        loaded_grid = Counter(self._parameter_key(params) for params in validated_grid)
        if loaded_grid != expected_grid:
            raise ValueError("param_grid does not match the configured discrete grid")

        index = state['index']
        if (
            isinstance(index, bool)
            or not isinstance(index, Integral)
            or not 0 <= index <= len(validated_grid)
        ):
            raise ValueError("index is outside the discrete grid")

        current_params = self._validate_state_parameters(
            state['current_params'], 'current_params'
        )
        if self._parameter_key(current_params) not in loaded_grid:
            raise ValueError("current_params is not in param_grid")

        param_history = state['param_history']
        step_history = state['step_history']
        timestamp_history = state['timestamp_history']
        if not isinstance(param_history, list):
            raise ValueError("param_history must be a list")
        if not isinstance(step_history, list) or not isinstance(timestamp_history, list):
            raise ValueError("step_history and timestamp_history must be lists")
        if not len(param_history) == len(step_history) == len(timestamp_history):
            raise ValueError("sampler history lengths must match")

        validated_history = []
        for history_index, params in enumerate(param_history):
            validated = self._validate_state_parameters(
                params, f'param_history[{history_index}]'
            )
            if self._parameter_key(validated) not in loaded_grid:
                raise ValueError(f"param_history[{history_index}] is not in param_grid")
            validated_history.append(validated)
        validated_steps = []
        for history_index, step in enumerate(step_history):
            if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
                raise ValueError(
                    f"step_history[{history_index}] must be a non-negative integer"
                )
            validated_steps.append(int(step))
        if not all(isinstance(timestamp, str) for timestamp in timestamp_history):
            raise ValueError("timestamp_history entries must be strings")

        last_update_step = state['last_update_step']
        total_updates = state['total_updates']
        if (
            isinstance(last_update_step, bool)
            or not isinstance(last_update_step, Integral)
            or last_update_step < 0
        ):
            raise ValueError("last_update_step must be a non-negative integer")
        if (
            isinstance(total_updates, bool)
            or not isinstance(total_updates, Integral)
            or total_updates < 0
        ):
            raise ValueError("total_updates must be a non-negative integer")

        self.param_grid = validated_grid
        self.index = int(index)
        self.current_params = current_params
        self._last_update_step = int(last_update_step)
        self._total_updates = int(total_updates)
        self.param_history = validated_history
        self.step_history = validated_steps
        self.timestamp_history = deepcopy(timestamp_history)
        self._rng.setstate(validated_rng.getstate())

    def get_parameter_history(self) -> Dict[str, list]:
        return {
            'parameters': self.param_history,
            'steps': self.step_history,
            'timestamps': self.timestamp_history
        }

    def get_statistics(self) -> Dict[str, Any]:
        if not self.param_history:
            return {}


        bias_values = [p['bias_inverse_tta_coef'] for p in self.param_history]
        sigma0_values = [p['perception_sigma0'] for p in self.param_history]
        sigma_max_values = [p['perception_sigma_max'] for p in self.param_history]
        delay_values = [p['delay_steps'] for p in self.param_history]

        stats = {
            'total_updates': self._total_updates,
            'last_update_step': self._last_update_step,
            'update_frequency': self.update_steps,
            'total_combinations': len(self.param_grid),

            'bias_inverse_tta_coef': {
                'mean': np.mean(bias_values),
                'std': np.std(bias_values),
                'min': np.min(bias_values),
                'max': np.max(bias_values),
                'range': self.bias_inverse_tta_coef_range,
                'discrete_values': self.bias_inverse_tta_coef_values.tolist()
            },

            'perception_sigma0': {
                'mean': np.mean(sigma0_values),
                'std': np.std(sigma0_values),
                'min': np.min(sigma0_values),
                'max': np.max(sigma0_values),
                'range': self.perception_sigma0_range,
                'discrete_values': self.perception_sigma0_values.tolist()
            },

            'perception_sigma_max': {
                'mean': np.mean(sigma_max_values),
                'std': np.std(sigma_max_values),
                'min': np.min(sigma_max_values),
                'max': np.max(sigma_max_values),
                'range': self.perception_sigma_max_range,
                'discrete_values': self.perception_sigma_max_values.tolist()
            },

            'delay_steps': {
                'mean': np.mean(delay_values),
                'std': np.std(delay_values),
                'min': np.min(delay_values),
                'max': np.max(delay_values),
                'range': self.delay_steps_range,
                'discrete_values': self.delay_steps_values.tolist(),
                'value_counts': {int(i): delay_values.count(i) for i in set(delay_values)}
            }
        }

        return stats

    def save_history_to_file(self, file_path: str):
        if not self.save_history or not self.param_history:
            return


        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        from datetime import datetime
        history_data = {
            'sampler_config': {
                'sampler_type': 'discrete',
                'update_steps': self.update_steps,
                'bias_inverse_tta_coef_range': self.bias_inverse_tta_coef_range,
                'perception_sigma0_range': self.perception_sigma0_range,
                'perception_sigma_max_range': self.perception_sigma_max_range,
                'delay_steps_range': self.delay_steps_range,
                'total_combinations': len(self.param_grid)
            },
            'parameter_history': convert_numpy_types(self.param_history),
            'step_history': convert_numpy_types(self.step_history),
            'timestamp_history': self.timestamp_history,
            'statistics': convert_numpy_types(self.get_statistics()),
            'export_timestamp': datetime.now().isoformat()
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:

            try:
                simplified_data = {
                    'sampler_config': history_data['sampler_config'],
                    'total_updates': len(self.param_history),
                    'current_parameters': convert_numpy_types(self.current_params),
                    'export_timestamp': datetime.now().isoformat()
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    def generate_parameter_visualization(self, output_dir: str = "cognitive_visualization",
                                       session_name: str = None) -> str:
        if not self.enable_visualization or len(self.param_history) < 2:
            return None

        try:
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime

            if session_name is None:
                session_name = f"discrete_sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


            os.makedirs(output_dir, exist_ok=True)


            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Discrete Cognitive Parameter Sampling - {session_name}', fontsize=14)


            bias_values = [p['bias_inverse_tta_coef'] for p in self.param_history]
            sigma0_values = [p['perception_sigma0'] for p in self.param_history]
            sigma_max_values = [p['perception_sigma_max'] for p in self.param_history]
            delay_values = [p['delay_steps'] for p in self.param_history]


            axes[0, 0].plot(self.step_history, bias_values, 'o-', alpha=0.7)
            axes[0, 0].set_title('Bias Inverse TTA Coefficient')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True)


            axes[0, 1].plot(self.step_history, sigma0_values, 's-', alpha=0.7, color='orange')
            axes[0, 1].set_title('Perception Sigma0')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True)


            axes[1, 0].plot(self.step_history, sigma_max_values, '^-', alpha=0.7, color='green')
            axes[1, 0].set_title('Perception Sigma Max')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True)


            axes[1, 1].plot(self.step_history, delay_values, 'd-', alpha=0.7, color='red')
            axes[1, 1].set_title('Delay Steps')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].grid(True)

            plt.tight_layout()


            output_file = os.path.join(output_dir, f"{session_name}_parameter_sampling.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            return output_file

        except Exception as e:
            return None
