#!/usr/bin/env python3

import math

import numpy as np


def safe_energy_calculation_patch():
    try:
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        from metadrive.utils.math import norm

        original_update_energy = BaseVehicle._update_energy_consumption

        def safe_update_energy_consumption(self):
            """Prevent overflow in MetaDrive energy-consumption updates."""
            if (
                (hasattr(self, "crash_vehicle") and self.crash_vehicle)
                or (hasattr(self, "crash_building") and self.crash_building)
                or (hasattr(self, "crash_object") and self.crash_object)
                or (hasattr(self, "crash_human") and self.crash_human)
                or (hasattr(self, "crash_sidewalk") and self.crash_sidewalk)
            ):
                return 0.0, self.energy_consumption

            distance = (
                norm(
                    self.last_position[0] - self.position[0],
                    self.last_position[1] - self.position[1],
                )
                / 1000
            )

            safe_speed = min(self.speed_km_h, 300.0)
            safe_speed = max(safe_speed, 0.0)

            try:
                exponent = 0.01 * safe_speed
                if exponent > 4.6:
                    step_energy = 3.25 * 100.0 * distance / 100
                else:
                    step_energy = 3.25 * math.pow(np.e, exponent) * distance / 100

                step_energy = step_energy * 1000

                if math.isnan(step_energy) or math.isinf(step_energy):
                    step_energy = 0.0

                self.energy_consumption += step_energy
                return step_energy, self.energy_consumption

            except (OverflowError, ValueError) as e:
                print(
                    "Energy-consumption calculation failed: "
                    f"{e}, speed={self.speed_km_h:.2f} km/h; using fallback value"
                )
                default_energy = 0.1 * distance * 1000
                self.energy_consumption += default_energy
                return default_energy, self.energy_consumption

        BaseVehicle._update_energy_consumption = safe_update_energy_consumption
        return True

    except ImportError as e:
        print(f"Failed to import MetaDrive modules: {e}")
        return False
    except Exception as e:
        print(f"Failed to apply overflow patch: {e}")
        return False


def patch_math_functions():
    import math
    import numpy as np

    original_pow = math.pow

    def safe_pow(base, exponent):
        try:
            if base == np.e or abs(base - np.e) < 1e-10:
                if exponent > 100:
                    return original_pow(np.e, 100)
                elif exponent < -100:
                    return original_pow(np.e, -100)

            result = original_pow(base, exponent)
            if math.isinf(result) or math.isnan(result):
                return 1e10
            return result
        except OverflowError:
            print(f"Math overflow in pow({base}, {exponent}); using capped value")
            return 1e10 if exponent > 0 else 1e-10

    math.pow = safe_pow


def apply_all_fixes():
    patch_math_functions()
    success = safe_energy_calculation_patch()

    if success:
        print("All overflow safeguards were applied successfully")
    else:
        print("Some safeguards may not have been applied successfully")
        print("   Hint: rerun `pip install -e .` from the repository root")

    return success


if __name__ == "__main__":
    apply_all_fixes()
