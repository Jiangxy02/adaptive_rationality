"""Metric derivation and collection helpers extracted from PPOExpertReproduction."""


class MetricsComputeMixin:
    def _calculate_missing_metrics(self, env, info, env_idx=None):
        """
        Compute metrics missing from the MetaDrive ``info`` dict.

        Args:
            env: MetaDrive environment instance used to access the agent.
            info: Info dictionary returned by the environment.
            env_idx: Environment index used for lane-change tracking.

        Returns:
            A dictionary containing the derived metrics.
        """
        metrics = {}

        try:
            agent = env.agent if hasattr(env, 'agent') else None
            if agent is None:
                return metrics

            # 1. Lane-deviation computation.
            try:
                # Method 1: read the lateral distance to lane center directly.
                if hasattr(agent, 'lateral_distance_to_lane_center'):
                    metrics['lane_deviation'] = abs(agent.lateral_distance_to_lane_center)
                elif hasattr(agent, 'lane') and hasattr(agent, 'position'):
                    # Method 2: estimate lane deviation from steering and speed.
                    steering = getattr(agent, 'steering', 0.0)
                    speed = getattr(agent, 'speed', 0.0)

                    if hasattr(agent.lane, 'width'):
                        lane_width = agent.lane.width
                        estimated_deviation = abs(steering) * (1 + speed * 0.1) * lane_width * 0.3
                        metrics['lane_deviation'] = min(estimated_deviation, lane_width / 2)
                    else:
                        metrics['lane_deviation'] = abs(steering) * 0.5
                else:
                    metrics['lane_deviation'] = 0.0
            except Exception:
                metrics['lane_deviation'] = 0.0

            # 2. Lane-change detection.
            try:
                lane_change_detected = False

                # Method 1: trust the environment's lane_change flag.
                if isinstance(info, dict) and 'lane_change' in info:
                    lane_change_detected = bool(info['lane_change'])
                    if lane_change_detected:
                        if self.debug_lane_change:
                            print(f"[env {env_idx}] Lane change detected from info")
                        return {"lane_change": True}

                # Method 2: detect a change in lane index.
                if not lane_change_detected and hasattr(agent, 'lane_index'):
                    current_lane_index = agent.lane_index
                    env_agent_id = f"env_{env_idx}_lane_index"

                    if env_agent_id in self._last_lane_index:
                        if self._last_lane_index[env_agent_id] != current_lane_index:
                            if self.debug_lane_change:
                                print(f"[env {env_idx}] Lane index changed: {self._last_lane_index[env_agent_id]} -> {current_lane_index}")
                            self._last_lane_index[env_agent_id] = current_lane_index
                            return {"lane_change": True}
                        else:
                            self._last_lane_index[env_agent_id] = current_lane_index
                            return {"lane_change": False}
                    else:
                        self._last_lane_index[env_agent_id] = current_lane_index
                        if self.debug_lane_change:
                            print(f"[env {env_idx}] Recorded initial lane index: {current_lane_index}")
                        return {"lane_change": False}

                # Method 3: steering-and-speed heuristic.
                if not lane_change_detected:
                    try:
                        steering = getattr(agent, 'steering', 0.0)
                        speed = getattr(agent, 'speed', 0.0)

                        # Large sustained steering at moderate speed may indicate a lane change.
                        if abs(steering) > 0.3 and 5.0 < speed < 20.0:
                            if env_idx is not None:
                                steering_key = f"env_{env_idx}_steering"
                            else:
                                steering_key = f"agent_{getattr(agent, 'id', id(agent))}_steering"

                            if steering_key not in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0

                            self._last_lane_index[steering_key] += 1

                            if self._last_lane_index[steering_key] >= 3:
                                if self.debug_lane_change:
                                    print(f"[env {env_idx}] Lane change detected from steering pattern: steering={steering:.3f}, speed={speed:.3f}, consecutive_steps={self._last_lane_index[steering_key]}")
                                self._last_lane_index[steering_key] = 0
                                return {"lane_change": True}
                        else:
                            steering_key = f"env_{env_idx}_steering_pattern"
                            if steering_key in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0

                    except Exception as e:
                        if self.debug_lane_change:
                            print(f"[env {env_idx}] Steering-pattern detection failed: {e}")

                # Method 4: surrogate detection for straight-road scenarios.
                if not lane_change_detected:
                    try:
                        steering = getattr(agent, 'steering', 0.0)
                        speed = getattr(agent, 'speed', 0.0)

                        # In straight-road scenarios, detect sustained lane-change intent.
                        if abs(steering) > 0.15 and 5.0 < speed < 25.0:
                            steering_key = f"env_{env_idx}_steering_intent"

                            if steering_key not in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0

                            self._last_lane_index[steering_key] += 1

                            if self._last_lane_index[steering_key] >= 3:
                                if self.debug_lane_change:
                                    print(f"[env {env_idx}] Lane-change intent detected: steering={steering:.3f}, speed={speed:.3f}, consecutive_steps={self._last_lane_index[steering_key]}")
                                self._last_lane_index[steering_key] = 0
                                return {"lane_change": True}
                        else:
                            steering_key = f"env_{env_idx}_steering_intent"
                            if steering_key in self._last_lane_index:
                                self._last_lane_index[steering_key] = 0

                    except Exception as e:
                        if self.debug_lane_change:
                            print(f"[env {env_idx}] Lane-change-intent detection failed: {e}")

                metrics['lane_change'] = lane_change_detected

                if lane_change_detected:
                    print(f"Lane change detected successfully. env={env_idx if env_idx is not None else 'N/A'}, lane_index={getattr(agent, 'lane_index', 'N/A')}")

            except Exception as e:
                print(f"Lane-change detection error: {e}")
                metrics['lane_change'] = False

        except Exception as e:
            metrics = {
                'lane_deviation': 0.0,
                'lane_change': False
            }

        return metrics

    def _detect_lane_change_enhanced(self, agent, env_idx, info):
        return False

    def _collect_speed_control_metrics(self):
        """
        Collect metric data for the speed-control reward.

        Returns:
            A dictionary containing the speed-control reward metrics.
        """
        if not self.args.use_speed_control_reward:
            return None

        try:
            speed_control_data = {}

            # Use the first environment instance for data collection.
            env_instance = None
            if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                env_instance = self.envs.envs[0]
            elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                if len(self.envs.venv.envs) > 0:
                    env_instance = self.envs.venv.envs[0]

            if env_instance is None:
                return None

            if not hasattr(env_instance, '_compute_speed_control_reward'):
                return None

            if hasattr(env_instance, 'agent') and env_instance.agent:
                vehicle = env_instance.agent
            elif hasattr(env_instance, 'agents') and len(env_instance.agents) > 0:
                vehicle_id = list(env_instance.agents.keys())[0]
                vehicle = env_instance.agents[vehicle_id]
            else:
                return None

            # Compute the speed-control reward.
            try:
                current_action = [0.0, 0.0]
                if hasattr(vehicle, 'current_action'):
                    current_action = vehicle.current_action

                total_reward = env_instance._compute_speed_control_reward(vehicle, current_action)

                # Pull detailed data from step_infos when available.
                if hasattr(env_instance, 'step_infos') and vehicle.id in env_instance.step_infos:
                    step_info = env_instance.step_infos[vehicle.id]

                    speed_control_data.update({
                        'r_total': step_info.get('sc_r_total', total_reward),
                        'r_track': step_info.get('sc_r_track', 0.0),
                        'r_wall': step_info.get('sc_r_wall', 0.0),
                        'r_act_over': step_info.get('sc_r_act_over', 0.0),
                        'current_speed': step_info.get('sc_v', vehicle.speed),
                        'target_speed': step_info.get('sc_v_ref', env_instance.v_ref),
                        'speed_deviation': step_info.get('sc_dv', vehicle.speed - env_instance.v_ref),
                        'acceleration': step_info.get('sc_a', 0.0),
                        'speed_control_enable_tracking': step_info.get('sc_enable_tracking', True),
                        'speed_control_enable_soft_wall': step_info.get('sc_enable_soft_wall', True),
                        'speed_control_enable_behavior_guidance': step_info.get('sc_enable_behavior_guidance', True),
                    })
                else:
                    current_speed = vehicle.speed
                    target_speed = env_instance.v_ref
                    speed_deviation = current_speed - target_speed

                    speed_control_data.update({
                        'r_total': total_reward,
                        'r_track': 0.0,
                        'r_wall': 0.0,
                        'r_act_over': 0.0,
                        'current_speed': current_speed,
                        'target_speed': target_speed,
                        'speed_deviation': speed_deviation,
                        'acceleration': 0.0,
                    })

                current_speed = speed_control_data['current_speed']
                target_speed = speed_control_data['target_speed']
                speed_deviation = speed_control_data['speed_deviation']
                acceleration = speed_control_data['acceleration']

                speed_control_data['speed_ratio'] = current_speed / max(target_speed, 0.1) if target_speed > 0 else 0.0

                speed_control_data['acceleration_positive'] = max(acceleration, 0.0)
                speed_control_data['acceleration_negative'] = max(-acceleration, 0.0)

                speed_control_data['overspeed_flag'] = 1.0 if speed_deviation > 0 else 0.0

                speed_control_data['reward_mean'] = speed_control_data['r_total']
                speed_control_data['reward_std'] = 0.0
                speed_control_data['reward_min'] = speed_control_data['r_total']
                speed_control_data['reward_max'] = speed_control_data['r_total']

                return speed_control_data

            except Exception as e:
                print(f"Failed to compute speed-control reward: {e}")
                return None

        except Exception as e:
            print(f"Failed to collect speed-control metrics: {e}")
            return None
