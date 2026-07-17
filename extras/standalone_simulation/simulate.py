#!/usr/bin/env python3
"""Optional standalone simulation entrypoint for a trained PPO checkpoint."""


import os
import sys

from common.headless import apply_headless_guard
apply_headless_guard()

import argparse
from pathlib import Path

from extras.standalone_simulation.simulator import PPOCheckpointSimulator

def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description="PPO checkpoint simulator")

    parser.add_argument("--checkpoint", type=str,
                       default="<PATH_TO_CHECKPOINT>",
                       help="Checkpoint file path")

    parser.add_argument("--config", type=str, default=None,
                       help="Config file path (optional)")

    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of simulation episodes")

    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum steps per episode")

    parser.add_argument("--no_render", action="store_true",
                       help="Disable rendering")

    parser.add_argument("--stochastic", action="store_true",
                       help="Use a stochastic policy (deterministic by default)")

    parser.add_argument("--evaluate", action="store_true",
                       help="Run model evaluation")

    parser.add_argument("--eval_episodes", type=int, default=20,
                       help="Number of evaluation episodes")

    # ===== Lane-change penalty configuration =====
    parser.add_argument("--w_lc", type=float, default=0.6,
                       help="Base lane-change cost (default: 0.6)")
    parser.add_argument("--k_speed", type=float, default=1.0,
                       help="High-speed amplification factor (default: 1.0)")
    parser.add_argument("--v_limit", type=float, default=15.0,
                       help="Speed limit used for normalization (default: 15.0)")
    parser.add_argument("--lc_cooldown_s", type=float, default=4.0,
                       help="Lane-change cooldown in seconds (default: 4.0)")
    parser.add_argument("--w_lc_cool", type=float, default=1,
                       help="Additional penalty during cooldown (default: 1)")

    # ===== Cognitive module configuration =====
    parser.add_argument("--use_cognitive_modules", action="store_true",
                       help="Enable cognitive modules (default: disabled)")

    parser.add_argument("--use_cognitive_bias", action="store_true",
                       help="Enable the cognitive bias module (default: disabled)")
    parser.add_argument("--bias_visual_aversion", action="store_true",
                       help="Enable visual aversion in the cognitive bias module (default: enabled)")
    parser.add_argument("--bias_visual_distance", type=float, default=300.0,
                       help="Cognitive bias module visual distance (default: 50.0)")
    parser.add_argument("--bias_inverse_tta_coef", type=float, default=1.5,
                       help="Cognitive bias looming-penalty coefficient c (default: 1.5)")
    parser.add_argument("--bias_tta_threshold", type=float, default=0.1,
                       help="Cognitive bias module TTA threshold (default: 0.1)")

    parser.add_argument("--use_cognitive_delay", action="store_true",
                       help="Enable the cognitive delay module (default: disabled)")
    parser.add_argument("--delay_steps", type=int, default=2,
                       help="Cognitive delay module delay steps (default: 2)")  # One step equals 0.1 s.

    parser.add_argument("--use_cognitive_perception", action="store_true",
                       help="Enable the cognitive perception module (default: disabled)")
    parser.add_argument("--perception_sigma0", type=float, default=0.1,
                       help="Raw sigma0 input for the cognitive perception network (default: 0.1)")
    parser.add_argument("--perception_sigma_max", type=float, default=0.8,
                       help="Cognitive perception input and noise cap sigma_max (default: 0.8)")
    parser.add_argument("--perception_noise_std", type=float, default=0.01,
                       help="Deprecated: legacy cognitive perception observation-noise standard deviation (default: 0.01)")
    parser.add_argument("--perception_attention_bias", type=float, default=0.1,
                       help="Cognitive perception attention bias (default: 0.1)")
    parser.add_argument("--perception_enable_lidar_noise", action="store_true",
                       help="Enable lidar noise in the cognitive perception module (default: enabled)")
    parser.add_argument("--perception_enable_state_noise", action="store_true",
                       help="Enable state noise in the cognitive perception module (default: enabled)")
    parser.add_argument("--perception_enable_attention_bias", action="store_true",
                       help="Enable attention bias in the cognitive perception module (default: enabled)")

    # ===== Speed-control reward configuration =====
    parser.add_argument("--use_speed_control_reward", action="store_true",
                       help="Enable the speed-control reward (default: disabled)")
    parser.add_argument("--speed_control_k", type=float, default=1.0,
                       help="Speed-tracking coefficient (default: 1.0)")
    parser.add_argument("--speed_control_kappa", type=float, default=0.5,
                       help="Soft speed-wall coefficient (default: 0.5)")
    parser.add_argument("--speed_control_mu", type=float, default=0.3,
                       help="Overspeed braking reward coefficient (default: 0.3)")
    parser.add_argument("--speed_control_nu", type=float, default=0.2,
                       help="Overspeed acceleration penalty coefficient (default: 0.2)")
    parser.add_argument("--speed_control_v_tolerance", type=float, default=1.0,
                       help="Speed-tracking tolerance (default: 1.0)")
    parser.add_argument("--speed_control_v_ref", type=float, default=15.0,
                       help="Target reference speed (default: 15.0)")
    parser.add_argument("--speed_control_enable_tracking", action="store_true",
                       help="Enable the speed-tracking submodule (default: disabled)")
    parser.add_argument("--speed_control_enable_soft_wall", action="store_true",
                       help="Enable the soft speed-wall submodule (default: disabled)")
    parser.add_argument("--speed_control_enable_behavior_guidance", action="store_true",
                       help="Enable the behavior-guidance submodule (default: disabled)")

    # ===== Cognitive visualization configuration =====
    parser.add_argument("--enable_cognitive_viz", action="store_true",
                       help="Enable cognitive-module visualization (default: disabled)")
    parser.add_argument("--enable_radar_beam_viz", action="store_true",
                       help="Enable radar-beam visualization (default: disabled)")

    parser.add_argument("--device", type=str, default="cpu",
                       choices=["auto", "cpu", "cuda"],
                       help="Compute device")

    args = parser.parse_args()

    # Validate the checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file does not exist: {args.checkpoint}")
        sys.exit(1)


    # Create the simulator
    simulator = PPOCheckpointSimulator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        args=args  # Pass command-line arguments through.
    )

    # Apply the requested speed-control configuration.
    if args.use_speed_control_reward:
        # Update only MetaDrive-compatible configuration keys.
        simulator.config.update({
            "use_speed_control_reward": True
        })
        print("Updated speed-control reward configuration")
        print(f"   Speed-tracking coefficient: {args.speed_control_k}")
        print(f"   Soft speed-wall coefficient: {args.speed_control_kappa}")
        print(f"   Behavior-guidance coefficients: μ={args.speed_control_mu}, ν={args.speed_control_nu}")
        print(f"   Target speed: {args.speed_control_v_ref} m/s")
        print(f"   Speed tolerance: {args.speed_control_v_tolerance} m/s")
        print(f"   Submodule status: tracking={args.speed_control_enable_tracking}, soft_wall={args.speed_control_enable_soft_wall}, behavior={args.speed_control_enable_behavior_guidance}")
        print("   Note: speed-control parameters are applied during environment creation through CLI arguments.")

    if args.evaluate:
        # Run model evaluation
        evaluation = simulator.evaluate_model(
            num_episodes=args.eval_episodes,
            render=not args.no_render
        )

        print(f"\nEvaluation summary:")
        print(f"Success rate: {evaluation['success_rate']:.1%}")
        print(f"Average reward: {evaluation['avg_reward']:.2f} ± {evaluation['std_reward']:.2f}")
        print(f"Average episode length: {evaluation['avg_episode_length']:.1f}")
        print(f"Average path completion: {evaluation['avg_path_completion']:.1%}")

        # Lane-change metrics
        print(f"Total lane changes: {evaluation['total_lane_changes']}")
        print(f"Average lane-change penalty: {evaluation['avg_lane_change_penalty']:.3f}")
        print(f"Average lane-change speed ratio: {evaluation['avg_lane_change_speed_ratio']:.3f}")
        print(f"Total cooldown violations: {evaluation['total_cooldown_violations']}")

    else:
        # Run the standard simulation
        simulator.run_simulation(
            num_episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps,
            deterministic=not args.stochastic
        )

    print("\nOPTIONAL_SIMULATION_OK")




if __name__ == "__main__":
    main()
