
import numpy as np

class NavigationManager:


    def __init__(self, env):

        self.env = env

    @staticmethod
    def _iter_road_lanes(road_network):
        """Yield validated ``(lane_index, lane)`` pairs from a road network."""
        graph = road_network.graph
        if not hasattr(graph, "items"):
            raise TypeError("road network graph must be a mapping")

        for road_start, destinations in graph.items():
            if not hasattr(destinations, "items"):
                raise TypeError(
                    f"road destinations for {road_start!r} must be a mapping"
                )
            for road_end, lanes in destinations.items():
                if hasattr(lanes, "items"):
                    lane_items = lanes.items()
                elif isinstance(lanes, (list, tuple)):
                    lane_items = enumerate(lanes)
                else:
                    raise TypeError(
                        "lane container must be a list, tuple, or mapping; "
                        f"got {type(lanes).__name__} for "
                        f"{road_start!r}->{road_end!r}"
                    )

                for lane_id, lane in lane_items:
                    if lane is None:
                        continue
                    lane_index = getattr(lane, "index", None)
                    expected_index = (road_start, road_end, lane_id)
                    if tuple(lane_index or ()) != expected_index:
                        raise ValueError(
                            "lane index does not match its road-network location: "
                            f"expected {expected_index!r}, got {lane_index!r}"
                        )
                    yield expected_index, lane

    def set_custom_destination(self):


        max_x = float('-inf')
        target_y = 0.0


        if self.env.enable_background_vehicles and self.env.trajectory_dict:
            for vehicle_id, trajectory in self.env.trajectory_dict.items():
                for point in trajectory:
                    if point["x"] > max_x:
                        max_x = point["x"]
                        target_y = point["y"]


        if self.env.main_vehicle_trajectory:
            for point in self.env.main_vehicle_trajectory:
                if point["x"] > max_x:
                    max_x = point["x"]
                    target_y = point["y"]

        if max_x == float('-inf'):
            max_x = 500.0
            target_y = 0.0

        target_position = [max_x, target_y]


        self._find_and_set_target_lane(max_x, target_y, target_position)


        self.env.custom_destination = target_position

    def _find_and_set_target_lane(self, max_x, target_y, target_position):
        current_map = getattr(self.env.engine, 'current_map', None)
        if current_map is None:
            raise RuntimeError("destination setup requires an initialized map")

        closest_lane = None
        closest_end_position = None
        min_distance = float('inf')
        for _, lane in self._iter_road_lanes(current_map.road_network):
            end_position = lane.position(lane.length, 0)
            distance = np.hypot(
                end_position[0] - max_x,
                end_position[1] - target_y,
            )
            if distance < min_distance:
                min_distance = distance
                closest_lane = lane
                closest_end_position = end_position

        if closest_lane is None:
            raise RuntimeError("destination setup found no usable lanes")

        navigation = getattr(self.env.agent, 'navigation', None)
        current_lane = getattr(navigation, 'current_lane', None)
        if navigation is None or current_lane is None:
            raise RuntimeError("destination setup requires a current navigation lane")

        target_position[0] = max_x
        target_position[1] = closest_end_position[1]
        navigation.set_route(current_lane.index, closest_lane.index[1])

    def fix_pg_map_navigation(self):



        current_map = self.env.engine.current_map
        road_network = current_map.road_network
        current_lane = self.env.agent.navigation.current_lane
        if current_lane is None:
            return False

        target_lane_index = None
        max_distance = -1.0
        current_pos = self.env.agent.position
        for lane_index, lane in self._iter_road_lanes(road_network):
            lane_end_pos = lane.position(lane.length, 0)
            distance = np.hypot(
                lane_end_pos[0] - current_pos[0],
                lane_end_pos[1] - current_pos[1],
            )
            if distance > max_distance:
                max_distance = distance
                target_lane_index = lane_index

        if target_lane_index is None:
            return False

        self.env.agent.navigation.set_route(
            current_lane.index,
            target_lane_index[1],
        )
        return True

    def fix_lane_detection(self):



        try:
            agent = self.env.agent
            navigation = agent.navigation
            current_map = self.env.engine.current_map
            road_network = current_map.road_network
            agent_pos = agent.position





            best_lane = self._find_correct_lane(road_network, agent_pos)

            if best_lane:




                self._update_navigation_lanes(navigation, best_lane, road_network)


                return True
            else:

                return False

        except Exception as e:

            import traceback
            traceback.print_exc()
            return False

    def _find_correct_lane(self, road_network, agent_pos):
        best_lane = None
        min_distance = float('inf')

        for road_start in road_network.graph.keys():
            for road_end in road_network.graph[road_start].keys():
                lanes = road_network.graph[road_start][road_end]


                if hasattr(lanes, 'items'):
                    lane_items = lanes.items()
                elif isinstance(lanes, (list, tuple)):
                    lane_items = enumerate(lanes)
                else:
                    continue

                for lane_idx, lane in lane_items:
                    if lane:
                        try:

                            local_coords = lane.local_coordinates(agent_pos)
                            longitudinal = local_coords[0]
                            lateral = local_coords[1]


                            is_on_lane = (0 <= longitudinal <= lane.length) and (abs(lateral) < 5)

                            if is_on_lane:

                                distance = abs(lateral)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_lane = lane

                        except Exception as e:
                            continue

        return best_lane

    def _update_navigation_lanes(self, navigation, best_lane, road_network):

        navigation._current_lane = best_lane


        navigation.current_ref_lanes = [best_lane]



        next_lane = self._find_next_lane(best_lane, road_network)
        if next_lane:
            navigation.next_ref_lanes = [next_lane]

        else:
            navigation.next_ref_lanes = [best_lane]



        if hasattr(navigation, '_target_checkpoints_index'):
            navigation._target_checkpoints_index = [0, 1]



        navigation.update_localization(self.env.agent)

    def _find_next_lane(self, current_lane, road_network):

        try:
            current_index = current_lane.index


            if len(current_index) >= 3:
                current_start = current_index[0]
                current_end = current_index[1]
                lane_idx = current_index[2]


                if current_end in road_network.graph:
                    for next_end in road_network.graph[current_end].keys():
                        lanes = road_network.graph[current_end][next_end]


                        if hasattr(lanes, 'items'):
                            lane_items = lanes.items()
                        elif isinstance(lanes, (list, tuple)):
                            lane_items = enumerate(lanes)
                        else:
                            continue

                        for next_lane_idx, next_lane in lane_items:
                            if next_lane and next_lane_idx == lane_idx:
                                return next_lane


                        for next_lane_idx, next_lane in lane_items:
                            if next_lane:
                                return next_lane

            return None

        except Exception as e:

            return None

    def check_and_fix_checkpoint_issue(self):



        try:
            agent = self.env.agent
            navigation = agent.navigation


            has_backward_checkpoint = self._check_backward_checkpoints(agent, navigation)


            route_completion = getattr(navigation, 'route_completion', 0)
            travelled_length = getattr(navigation, 'travelled_length', 0)





            if has_backward_checkpoint or route_completion < 0 or travelled_length < 0:

                self._fix_checkpoint_issues(navigation, agent)


                final_completion = getattr(navigation, 'route_completion', -1)


                # if final_completion >= 0:

                # else:

            else:
                print(f"No checkpoint guidance issue was detected")

        except Exception as e:
            print(f"Checkpoint inspection failed: {e}")

    def _check_backward_checkpoints(self, agent, navigation):
        has_backward_checkpoint = False

        try:
            checkpoint1, checkpoint2 = navigation.get_checkpoints()
            agent_pos = agent.position[:2]

            for i, checkpoint in enumerate([checkpoint1, checkpoint2]):
                ckpt_pos = checkpoint[:2]


                direction_vec = np.array(ckpt_pos) - np.array(agent_pos)


                heading = agent.heading_theta
                heading_vec = np.array([np.cos(heading), np.sin(heading)])


                dot_product = np.dot(direction_vec, heading_vec)
                is_forward = dot_product > 0

                distance = np.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
                direction_str = "ahead" if is_forward else "behind"




                if not is_forward:
                    has_backward_checkpoint = True


        except Exception as e:
            print(f"Unable to read checkpoint information: {e}")

        return has_backward_checkpoint

    def _fix_checkpoint_issues(self, navigation, agent):

        if hasattr(navigation, 'travelled_length'):
            old_travelled = navigation.travelled_length
            navigation.travelled_length = 0.0



        if hasattr(navigation, '_last_long_in_ref_lane') and hasattr(navigation, 'current_ref_lanes'):
            if navigation.current_ref_lanes:
                ref_lane = navigation.current_ref_lanes[0]
                current_long, _ = ref_lane.local_coordinates(agent.position)
                navigation._last_long_in_ref_lane = current_long



        new_completion = getattr(navigation, 'route_completion', -1)
        if new_completion < 0:

            try:
                success = self.fix_pg_map_navigation()
                if success:
                    print(f"  Navigation reset succeeded")
                else:
                    print(f" Navigation reset failed; using the basic fallback")

                    if hasattr(navigation, 'total_length') and navigation.total_length > 0:
                        navigation.travelled_length = 0.01 * navigation.total_length

            except Exception as e:
                print(f"  Navigation reset raised an error: {e}")

    def fix_navigation_route_internal(self):



        try:
            agent = self.env.agent
            navigation = agent.navigation
            current_map = self.env.engine.current_map
            road_network = current_map.road_network


            current_pos = agent.position
            if hasattr(self.env, 'custom_destination'):
                target_pos = self.env.custom_destination
            else:

                target_pos = [current_pos[0] + 500, current_pos[1]]





            current_lane = navigation.current_lane
            if not current_lane:

                return False




            route = self._build_navigation_route(current_lane, road_network, target_pos)

            if len(route) > 1:

                self._apply_navigation_route(navigation, route, road_network, current_lane)





                return True
            else:

                return False

        except Exception as e:

            import traceback
            traceback.print_exc()
            return False

    def _build_navigation_route(self, current_lane, road_network, target_pos):
        route = [current_lane.index]
        current_lane_obj = current_lane


        for _ in range(10):
            next_lane = self._find_next_connected_lane(current_lane_obj, road_network)
            if next_lane:
                route.append(next_lane.index)
                current_lane_obj = next_lane


                lane_end_pos = next_lane.position(next_lane.length, 0)
                distance_to_target = np.sqrt((lane_end_pos[0] - target_pos[0])**2 +
                                           (lane_end_pos[1] - target_pos[1])**2)

                if distance_to_target < 100:

                    break
            else:
                break

        return route

    def _apply_navigation_route(self, navigation, route, road_network, current_lane):

        navigation.route = route
        navigation._target_lane_index = route[-1]


        navigation._current_ref_lanes = [current_lane]
        navigation._next_ref_lanes = []


        total_length = 0
        for lane_idx in route:
            try:
                lane = road_network.get_lane(lane_idx)
                if lane:
                    total_length += lane.length
            except:
                pass

        navigation.total_length = total_length
        navigation.travelled_length = 0.0


    def _find_next_connected_lane(self, current_lane, road_network):
        try:
            current_index = current_lane.index


            if len(current_index) >= 3:
                current_start = current_index[0]
                current_end = current_index[1]
                lane_idx = current_index[2]


                if current_end in road_network.graph:
                    for next_end in road_network.graph[current_end].keys():
                        lanes = road_network.graph[current_end][next_end]


                        if hasattr(lanes, 'items'):
                            lane_items = lanes.items()
                        elif isinstance(lanes, (list, tuple)):
                            lane_items = enumerate(lanes)
                        else:
                            continue


                        for next_lane_idx, next_lane in lane_items:
                            if next_lane and next_lane_idx == lane_idx:
                                return next_lane


                        for next_lane_idx, next_lane in lane_items:
                            if next_lane:
                                return next_lane

            return None

        except Exception as e:

            return None

    def debug_navigation_info(self):

        if hasattr(self.env.agent, 'navigation') and self.env.agent.navigation:
            nav = self.env.agent.navigation


            route = getattr(nav, 'route', None)
            checkpoints = getattr(nav, 'checkpoints', None)
            established_route = route if route and len(route) > 1 else checkpoints
            if established_route and len(established_route) > 1:
                print(
                    "Navigation route established: "
                    f"{established_route[:3]}"
                    f"{'...' if len(established_route) > 3 else ''}"
                )
            else:
                success = self.fix_navigation_route_internal()
                if success:


                    if hasattr(nav, 'route') and nav.route and len(nav.route) > 1:
                        print(f"Route after repair: {nav.route[:3]}{'...' if len(nav.route) > 3 else ''}")
                else:
                    print(f"Navigation route repair failed; PPO may not work correctly")


        else:
            print(f"Error: Agent has no navigation module!")
