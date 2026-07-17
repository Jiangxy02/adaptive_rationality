

import numpy as np
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import json
import os

_SAMPLER_UPDATE_STEPS_DEFAULT = 5
_SAMPLER_BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT = (0.0, 10.0)
_SAMPLER_PERCEPTION_SIGMA0_RANGE_DEFAULT = (0.0, 1.0)
_SAMPLER_PERCEPTION_SIGMA_MAX_RANGE_DEFAULT = (0.0, 5.0)
_SAMPLER_DELAY_STEPS_RANGE_DEFAULT = (0, 20)
_SAMPLER_ENABLE_VISUALIZATION_DEFAULT = True
_SAMPLER_SAVE_HISTORY_DEFAULT = True
_MAX_SIGMA_REJECTION_ATTEMPTS = 100


def _validated_real_range(name, values, *, non_negative=False):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two endpoints")
    lower, upper = values
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError(f"{name} endpoints must be real numbers")
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} endpoints must be finite")
    if lower > upper:
        raise ValueError(f"{name} lower endpoint must be <= upper endpoint")
    if non_negative and lower < 0:
        raise ValueError(f"{name} endpoints must be non-negative")
    return lower, upper


def _validated_integer_range(name, values, *, non_negative=False):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two endpoints")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{name} endpoints must be integers")
    lower, upper = (int(value) for value in values)
    if lower > upper:
        raise ValueError(f"{name} lower endpoint must be <= upper endpoint")
    if non_negative and lower < 0:
        raise ValueError(f"{name} endpoints must be non-negative")
    return lower, upper


def _validate_sigma_range_pair(sigma0_range, sigma_max_range):
    sigma0_low, sigma0_high = sigma0_range
    sigma_max_low, sigma_max_high = sigma_max_range
    both_fixed_equal = (
        sigma0_low == sigma0_high == sigma_max_low == sigma_max_high
    )
    if sigma0_low >= sigma_max_high and not both_fixed_equal:
        raise ValueError(
            "perception sigma ranges contain no sampleable sigma0 <= sigma_max pair"
        )


class CognitiveParameterSampler:

    def __init__(self,
                 update_steps: int = _SAMPLER_UPDATE_STEPS_DEFAULT,
                 bias_inverse_tta_coef_range: Tuple[float, float] = _SAMPLER_BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT,
                 perception_sigma0_range: Tuple[float, float] = _SAMPLER_PERCEPTION_SIGMA0_RANGE_DEFAULT,
                 perception_sigma_max_range: Tuple[float, float] = _SAMPLER_PERCEPTION_SIGMA_MAX_RANGE_DEFAULT,
                 delay_steps_range: Tuple[int, int] = _SAMPLER_DELAY_STEPS_RANGE_DEFAULT,
                 enable_visualization: bool = _SAMPLER_ENABLE_VISUALIZATION_DEFAULT,
                 save_history: bool = _SAMPLER_SAVE_HISTORY_DEFAULT,
                 seed: Optional[int] = None):
        if (
            isinstance(update_steps, bool)
            or not isinstance(update_steps, Integral)
            or update_steps <= 0
        ):
            raise ValueError("update_steps must be a positive integer")
        self.update_steps = int(update_steps)


        self.bias_inverse_tta_coef_range = _validated_real_range(
            "bias_inverse_tta_coef_range",
            bias_inverse_tta_coef_range,
            non_negative=True,
        )
        self.perception_sigma0_range = _validated_real_range(
            "perception_sigma0_range",
            perception_sigma0_range,
            non_negative=True,
        )
        self.perception_sigma_max_range = _validated_real_range(
            "perception_sigma_max_range",
            perception_sigma_max_range,
            non_negative=True,
        )
        _validate_sigma_range_pair(
            self.perception_sigma0_range,
            self.perception_sigma_max_range,
        )
        self.delay_steps_range = _validated_integer_range(
            "delay_steps_range",
            delay_steps_range,
            non_negative=True,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)


        self.current_params = self._sample_initial_parameters()


        self.param_history = []
        self.step_history = []
        self.timestamp_history = []


        self.enable_visualization = enable_visualization
        self.save_history = save_history


        self._last_update_step = 0
        self._total_updates = 0


        if self.save_history:
            self._record_parameter_update(0, "initialization")

    def _sample_initial_parameters(self) -> Dict[str, Any]:
        sigma0, sigma_max = self._sample_perception_sigmas()
        return {
            'bias_inverse_tta_coef': float(
                self.rng.uniform(*self.bias_inverse_tta_coef_range)
            ),
            'perception_sigma0': sigma0,
            'perception_sigma_max': sigma_max,
            'delay_steps': int(
                self.rng.integers(
                    self.delay_steps_range[0], self.delay_steps_range[1] + 1
                )
            ),
        }

    def _sample_perception_sigmas(self) -> Tuple[float, float]:
        """Rejection-sample a valid ``sigma0 <= sigma_max`` perception pair."""
        for _ in range(_MAX_SIGMA_REJECTION_ATTEMPTS):
            sigma0 = self.rng.uniform(*self.perception_sigma0_range)
            sigma_max = self.rng.uniform(*self.perception_sigma_max_range)
            if sigma0 <= sigma_max:
                return float(sigma0), float(sigma_max)

        raise RuntimeError(
            "unable to sample perception sigma0 <= sigma_max after "
            f"{_MAX_SIGMA_REJECTION_ATTEMPTS} attempts"
        )

    def should_update_parameters(self, current_step: int) -> bool:
        return current_step - self._last_update_step >= self.update_steps

    def update_parameters(self, current_step: int, force_update: bool = False) -> Dict[str, Any]:
        if not force_update and not self.should_update_parameters(current_step):
            return self.current_params


        sigma0, sigma_max = self._sample_perception_sigmas()
        new_params = {
            'bias_inverse_tta_coef': float(
                self.rng.uniform(*self.bias_inverse_tta_coef_range)
            ),
            'perception_sigma0': sigma0,
            'perception_sigma_max': sigma_max,
            'delay_steps': int(
                self.rng.integers(
                    self.delay_steps_range[0], self.delay_steps_range[1] + 1
                )
            ),
        }


        self.current_params = new_params
        self._last_update_step = current_step
        self._total_updates += 1


        if self.save_history:
            self._record_parameter_update(current_step, "sampling")

        return self.current_params

    def _record_parameter_update(self, step: int, update_type: str):
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
            'perception_sigma0_range': tuple(self.perception_sigma0_range),
            'perception_sigma_max_range': tuple(self.perception_sigma_max_range),
            'delay_steps_range': tuple(self.delay_steps_range),
            'enable_visualization': self.enable_visualization,
            'save_history': self.save_history,
            'seed': self.seed,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Capture all sampler-owned state needed for an exact resume."""
        return {
            'state_version': 2,
            'sampler_type': 'continuous',
            'config': deepcopy(self._state_config()),
            'rng_state': deepcopy(self.rng.bit_generator.state),
            'current_params': deepcopy(self.current_params),
            'last_update_step': self._last_update_step,
            'total_updates': self._total_updates,
            'param_history': deepcopy(self.param_history),
            'step_history': deepcopy(self.step_history),
            'timestamp_history': deepcopy(self.timestamp_history),
        }

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

        ranges = (
            ('bias_inverse_tta_coef', bias, self.bias_inverse_tta_coef_range),
            ('perception_sigma0', sigma0, self.perception_sigma0_range),
            ('perception_sigma_max', sigma_max, self.perception_sigma_max_range),
            ('delay_steps', delay, self.delay_steps_range),
        )
        for name, value, (lower, upper) in ranges:
            if not lower <= value <= upper:
                raise ValueError(f"{field}.{name} is outside the configured range")
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
            'current_params',
            'last_update_step',
            'total_updates',
            'param_history',
            'step_history',
            'timestamp_history',
        }
        if set(state) != expected_fields:
            raise ValueError("sampler state has missing or unexpected fields")
        if state['state_version'] != 2 or state['sampler_type'] != 'continuous':
            raise ValueError("unsupported continuous sampler state")

        config = state['config']
        if not isinstance(config, Mapping) or dict(config) != self._state_config():
            raise ValueError("continuous sampler configuration mismatch")

        rng_state = state['rng_state']
        try:
            validated_rng = np.random.default_rng()
            validated_rng.bit_generator.state = deepcopy(rng_state)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("invalid continuous sampler RNG state") from exc

        current_params = self._validate_state_parameters(
            state['current_params'], 'current_params'
        )
        param_history = state['param_history']
        step_history = state['step_history']
        timestamp_history = state['timestamp_history']
        if not isinstance(param_history, list):
            raise ValueError("param_history must be a list")
        if not isinstance(step_history, list) or not isinstance(timestamp_history, list):
            raise ValueError("step_history and timestamp_history must be lists")
        if not len(param_history) == len(step_history) == len(timestamp_history):
            raise ValueError("sampler history lengths must match")

        validated_history = [
            self._validate_state_parameters(params, f'param_history[{index}]')
            for index, params in enumerate(param_history)
        ]
        validated_steps = []
        for index, step in enumerate(step_history):
            if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
                raise ValueError(f"step_history[{index}] must be a non-negative integer")
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

        self.current_params = current_params
        self._last_update_step = int(last_update_step)
        self._total_updates = int(total_updates)
        self.param_history = validated_history
        self.step_history = validated_steps
        self.timestamp_history = deepcopy(timestamp_history)
        self.rng.bit_generator.state = deepcopy(validated_rng.bit_generator.state)

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

            'bias_inverse_tta_coef': {
                'mean': np.mean(bias_values),
                'std': np.std(bias_values),
                'min': np.min(bias_values),
                'max': np.max(bias_values),
                'range': self.bias_inverse_tta_coef_range
            },

            'perception_sigma0': {
                'mean': np.mean(sigma0_values),
                'std': np.std(sigma0_values),
                'min': np.min(sigma0_values),
                'max': np.max(sigma0_values),
                'range': self.perception_sigma0_range
            },

            'perception_sigma_max': {
                'mean': np.mean(sigma_max_values),
                'std': np.std(sigma_max_values),
                'min': np.min(sigma_max_values),
                'max': np.max(sigma_max_values),
                'range': self.perception_sigma_max_range
            },

            'delay_steps': {
                'mean': np.mean(delay_values),
                'std': np.std(delay_values),
                'min': np.min(delay_values),
                'max': np.max(delay_values),
                'range': self.delay_steps_range,
                'value_counts': {i: delay_values.count(i) for i in set(delay_values)}
            }
        }

        return stats

    def update_config(self, **kwargs):
        update_steps = kwargs.get('update_steps', self.update_steps)
        if (
            isinstance(update_steps, bool)
            or not isinstance(update_steps, Integral)
            or update_steps <= 0
        ):
            raise ValueError("update_steps must be a positive integer")
        bias_range = _validated_real_range(
            "bias_inverse_tta_coef_range",
            kwargs.get('bias_inverse_tta_coef_range', self.bias_inverse_tta_coef_range),
            non_negative=True,
        )
        sigma0_range = _validated_real_range(
            "perception_sigma0_range",
            kwargs.get('perception_sigma0_range', self.perception_sigma0_range),
            non_negative=True,
        )
        sigma_max_range = _validated_real_range(
            "perception_sigma_max_range",
            kwargs.get('perception_sigma_max_range', self.perception_sigma_max_range),
            non_negative=True,
        )
        _validate_sigma_range_pair(sigma0_range, sigma_max_range)
        delay_range = _validated_integer_range(
            "delay_steps_range",
            kwargs.get('delay_steps_range', self.delay_steps_range),
            non_negative=True,
        )

        self.update_steps = int(update_steps)
        self.bias_inverse_tta_coef_range = bias_range
        self.perception_sigma0_range = sigma0_range
        self.perception_sigma_max_range = sigma_max_range
        self.delay_steps_range = delay_range


    def reset(self):
        self._last_update_step = 0
        self._total_updates = 0


        self.current_params = self._sample_initial_parameters()


        if not self.save_history:
            self.param_history.clear()
            self.step_history.clear()
            self.timestamp_history.clear()



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

        history_data = {
            'sampler_config': {
                'update_steps': self.update_steps,
                'bias_inverse_tta_coef_range': self.bias_inverse_tta_coef_range,
                'perception_sigma0_range': self.perception_sigma0_range,
                'perception_sigma_max_range': self.perception_sigma_max_range,
                'delay_steps_range': self.delay_steps_range
            },
            'parameter_history': convert_numpy_types(self.param_history),
            'step_history': convert_numpy_types(self.step_history),
            'timestamp_history': self.timestamp_history,
            'statistics': convert_numpy_types(self.get_statistics()),
            'export_timestamp': datetime.now().isoformat()
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"Parameter sampling history saved: {file_path}")
        except Exception as e:
            print(f"Failed to save parameter sampling history: {e}")

            try:
                simple_history = {
                    'total_updates': self._total_updates,
                    'last_update_step': self._last_update_step,
                    'current_parameters': convert_numpy_types(self.current_params),
                    'export_timestamp': datetime.now().isoformat()
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_history, f, indent=2, ensure_ascii=False)
                print(f"Simplified parameter history saved: {file_path}")
            except Exception as e2:
                print(f"Saving the simplified history also failed: {e2}")

    def load_history_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)


            if 'sampler_config' in history_data:
                config = history_data['sampler_config']
                self.update_steps = config.get('update_steps', self.update_steps)
                self.bias_inverse_tta_coef_range = config.get('bias_inverse_tta_coef_range', self.bias_inverse_tta_coef_range)
                self.perception_sigma0_range = config.get('perception_sigma0_range', self.perception_sigma0_range)
                self.perception_sigma_max_range = config.get('perception_sigma_max_range', self.perception_sigma_max_range)
                self.delay_steps_range = config.get('delay_steps_range', self.delay_steps_range)


            if 'parameter_history' in history_data:
                self.param_history = history_data['parameter_history']
                self.step_history = history_data['step_history']
                self.timestamp_history = history_data['timestamp_history']


                if self.param_history:
                    self.current_params = self.param_history[-1].copy()
                    self._last_update_step = self.step_history[-1] if self.step_history else 0
                    self._total_updates = len(self.param_history)

            print(f"Parameter sampling history loaded: {file_path}")
            print(f"History entries: {len(self.param_history)}")
            print(f"Last update step: {self._last_update_step}")
            print(f"Last update step: {self._last_update_step}")

        except Exception as e:
            print(f"Failed to load parameter sampling history: {e}")

    def generate_parameter_visualization(self, output_dir: str = None, session_name: str = None):
        if not self.enable_visualization or not self.param_history:
            print("No visualization data is available to generate plots")
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed, so visualization plots cannot be generated")
            return None

        try:

            if output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_dir = "./outputs/fig_cog"
                output_dir = os.path.join(base_dir, f"cognitive_analysis_{timestamp}", "parameter_sampling")

            if session_name is None:
                session_name = f"parameter_sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


            os.makedirs(output_dir, exist_ok=True)


            steps = np.array(self.step_history)
            bias_values = [p['bias_inverse_tta_coef'] for p in self.param_history]
            sigma0_values = [p['perception_sigma0'] for p in self.param_history]
            sigma_max_values = [p['perception_sigma_max'] for p in self.param_history]
            delay_values = [p['delay_steps'] for p in self.param_history]


            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Cognitive Parameter Sampling Analysis - {session_name}', fontsize=16, fontweight='bold')


            ax1 = axes[0, 0]
            ax1.plot(steps, bias_values, 'b-o', linewidth=2, markersize=6, alpha=0.8)
            ax1.axhline(y=self.bias_inverse_tta_coef_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax1.axhline(y=self.bias_inverse_tta_coef_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax1.set_title('Looming-Aversion Weight c', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Coefficient Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)


            ax2 = axes[0, 1]
            ax2.plot(steps, sigma0_values, 'g-o', linewidth=2, markersize=6, alpha=0.8)
            ax2.axhline(y=self.perception_sigma0_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax2.axhline(y=self.perception_sigma0_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax2.set_title('Perception Noise Standard Deviation (sigma0)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Simulation Steps')
            ax2.set_ylabel('Sigma Value (m)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)


            ax3 = axes[1, 0]
            ax3.plot(steps, sigma_max_values, 'm-o', linewidth=2, markersize=6, alpha=0.8)
            ax3.axhline(y=self.perception_sigma_max_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax3.axhline(y=self.perception_sigma_max_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax3.set_title('Perception Noise Max Standard Deviation (sigma_max)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Simulation Steps')
            ax3.set_ylabel('Sigma Max Value (m)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)


            ax4 = axes[1, 1]
            ax4.plot(steps, delay_values, 'c-o', linewidth=2, markersize=6, alpha=0.8)
            ax4.axhline(y=self.delay_steps_range[0], color='r', linestyle='--', alpha=0.5, label='Min Range')
            ax4.axhline(y=self.delay_steps_range[1], color='r', linestyle='--', alpha=0.5, label='Max Range')
            ax4.set_title('Action Delay Steps (delay_steps)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Simulation Steps')
            ax4.set_ylabel('Delay Steps')
            ax4.legend()
            ax4.grid(True, alpha=0.3)


            plt.tight_layout()


            output_file = os.path.join(output_dir, f"{session_name}_parameter_analysis.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Parameter sampling visualization saved: {output_file}")


            self._generate_parameter_report(output_dir, session_name)

            return output_file

        except Exception as e:
            print(f"Failed to generate parameter sampling visualization: {e}")
            return None

    def _generate_parameter_report(self, output_dir: str, session_name: str):
        report_file = os.path.join(output_dir, f"{session_name}_parameter_report.txt")

        stats = self.get_statistics()

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Cognitive Parameter Sampling Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session name: {session_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Sampler configuration:\n")
            f.write(f"  Parameter update frequency: {self.update_steps} steps\n")
            f.write(f"  Total updates: {stats.get('total_updates', 0)}\n")
            f.write(f"  Last update step: {stats.get('last_update_step', 0)}\n\n")

            f.write("Parameter range configuration:\n")
            f.write(f"  Looming aversion weight c: {self.bias_inverse_tta_coef_range}\n")
            f.write(f"  Perception noise standard deviation: {self.perception_sigma0_range} (m)\n")
            f.write(f"  Maximum perception noise standard deviation: {self.perception_sigma_max_range} (m)\n")
            f.write(f"  Action delay steps: {self.delay_steps_range}\n\n")

            f.write("Parameter sampling statistics:\n")
            for param_name, param_stats in stats.items():
                if isinstance(param_stats, dict) and 'mean' in param_stats:
                    f.write(f"  {param_name}:\n")
                    f.write(f"    Mean: {param_stats['mean']:.6f}\n")
                    f.write(f"    Standard deviation: {param_stats['std']:.6f}\n")
                    f.write(f"    Minimum: {param_stats['min']:.6f}\n")
                    f.write(f"    Maximum: {param_stats['max']:.6f}\n")
                    f.write(f"    Configured range: {param_stats['range']}\n")
                    if 'value_counts' in param_stats:
                        f.write(f"    Value distribution: {param_stats['value_counts']}\n")
                    f.write("\n")

            f.write("Notes:\n")
            f.write("  - Parameters are sampled uniformly within the configured ranges\n")
            f.write("  - Parameters are updated every {self.update_steps} steps\n")
            f.write("  - Updated parameters are applied to the corresponding cognitive modules immediately\n")
            f.write("  - Parameter history recording and visualization analysis are supported\n")
            f.write("  - sigma_max defines the maximum noise cap in the Weber-Fechner law\n")

        print(f"Parameter sampling analysis report saved: {report_file}")

    def get_status(self):
        return {
            'update_steps': self.update_steps,
            'current_parameters': self.current_params.copy(),
            'total_updates': self._total_updates,
            'last_update_step': self._last_update_step,
            'enable_visualization': self.enable_visualization,
            'save_history': self.save_history,
            'history_count': len(self.param_history)
        }
