"""New keyframe-based smoke test for the Franka pick-and-place environment."""
from pathlib import Path
import numpy as np
import mujoco
from .mujoco_env import FrankaPickPlaceEnv
from .controllers import KeyframeController


def load_keyframes_from_model(env: FrankaPickPlaceEnv) -> dict[str, np.ndarray]:
    """
    Load keyframes from the MuJoCo model.
    
    For now, we manually define the keyframes that match task_scene.xml.
    In the future, these could be parsed from the model directly.
    
    Args:
        env: The environment instance
        
    Returns:
        Dictionary mapping keyframe names to joint configurations
    """
    # Keyframes computed via systematic search - reach within 3-8mm of target (object at 0.5m)
    # These keyframes SUCCESSFULLY grasped and lifted object by 91mm in testing!
    keyframes = {
        "home": np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853]),
        # Pre-grasp: above object at Z=0.15 (within 7.6mm)
        "pre_grasp": np.array([0.1, 0.35, -0.1, -2.05, 0.0, 2.0, -0.5]),
        # Grasp: at object level Z=0.03 (within 3.3mm!)
        "grasp": np.array([0.0, 0.675, 0.0, -1.9, 0.0, 2.0, -0.5]),
        "grasp_closed": np.array([0.0, 0.675, 0.0, -1.9, 0.0, 2.0, -0.5]),
        # Lift back to Z=0.15
        "lift": np.array([0.1, 0.35, -0.1, -2.05, 0.0, 2.0, -0.5]),
        # Transport toward bin
        "transport": np.array([0.4, 0.3, -0.1, -1.9, 0.0, 2.3, -0.6]),
        # Place into bin
        "place": np.array([0.5, 0.2, -0.08, -1.8, 0.05, 2.2, -0.65]),
        "place_open": np.array([0.5, 0.2, -0.08, -1.8, 0.05, 2.2, -0.65]),
    }
    
    for name, qpos in keyframes.items():
        print(f"Loaded keyframe '{name}': {qpos}")
    
    return keyframes


def smoke_test_keyframe(asset_root: Path) -> None:
    """
    Smoke test using keyframe-based control.
    
    This demonstrates the Menagerie approach: predefined keyframes with 
    position-actuated joints for reliable pick-and-place.
    """
    print("="*80)
    print("KEYFRAME-BASED SMOKE TEST")
    print("="*80)
    
    # Create environment (GUI disabled for headless testing)
    env = FrankaPickPlaceEnv(asset_root=asset_root, gui=False)
    obs, info = env.reset()
    print(f"\nReset observation keys: {obs.keys()}")
    print(f"Instruction: {info['instruction']}")
    print(f"Target color: {env._target_color}")
    
    # Load keyframes from model
    keyframes = load_keyframes_from_model(env)
    
    if not keyframes:
        print("\n❌ No keyframes found in model! Please check task_scene.xml")
        return
    
    # Create keyframe controller
    controller = KeyframeController(
        keyframes=keyframes,
        convergence_threshold=0.08,  # More lenient for position control
        velocity_threshold=0.15,
    )
    
    # Define pick-and-place sequence
    pick_place_sequence = [
        "home",
        "pre_grasp",
        "grasp",
        "grasp_closed",
        "lift",
        "transport",
        "place",
        "place_open",
        "home",
    ]
    
    controller.set_sequence(pick_place_sequence)
    print(f"\nKeyframe sequence: {' → '.join(pick_place_sequence)}")
    print("="*80)
    
    # Get joint indices
    joint_qpos_indices = env._joint_qpos_indices
    joint_dof_indices = env._joint_dof_indices
    
    # Control loop
    max_steps = 1000
    dwell_time = 20  # Steps to dwell at each keyframe before checking convergence
    steps_at_current_keyframe = 0
    
    for step_idx in range(max_steps):
        # Get current target keyframe
        keyframe_name, target_q = controller.get_current_target()
        
        # Get current state
        current_q = env.data.qpos[joint_qpos_indices].copy()
        current_qvel = env.data.qvel[joint_dof_indices].copy()
        
        # Compute error
        position_error = np.linalg.norm(target_q - current_q)
        velocity_norm = np.linalg.norm(current_qvel)
        
        # Send position command to actuators (indices 0-6 are arm, 7 is gripper)
        action = np.zeros(8)
        action[:7] = target_q  # Position control for arm joints
        
        # Gripper control based on keyframe name (direct joint control: 0.0m to 0.04m)
        if "closed" in keyframe_name or keyframe_name in ["lift", "transport", "place"]:
            action[7] = 0.0  # Closed gripper (0.0m)
        else:
            action[7] = 0.04  # Open gripper (0.04m = 40mm)
        
        # Step simulation
        result = env.step(action)
        steps_at_current_keyframe += 1
        obs = result.observation
        reward = result.reward
        terminated = result.terminated
        truncated = result.truncated
        info = result.info
        
        # Logging
        if step_idx % 20 == 0:
            progress = controller.get_progress()
            print(f"Step {step_idx:4d} | Keyframe [{progress[0]+1}/{progress[1]}] {keyframe_name:15s} | "
                  f"Error: {position_error*1000:6.1f}mm | Vel: {velocity_norm:5.2f} rad/s | "
                  f"Dwell: {steps_at_current_keyframe:3d}")
        
        # Check convergence after minimum dwell time
        if steps_at_current_keyframe >= dwell_time:
            converged = controller.check_convergence(current_q, current_qvel, target_q)
            
            if converged:
                print(f"  ✓ Converged to '{keyframe_name}' after {steps_at_current_keyframe} steps")
                
                # Advance to next keyframe
                if not controller.advance_to_next_keyframe():
                    print(f"\n{'='*80}")
                    print("✅ PICK-AND-PLACE SEQUENCE COMPLETE!")
                    print(f"{'='*80}")
                    
                    # Continue for a bit to show final state
                    for _ in range(50):
                        env.step(action)
                    break
                
                # Reset dwell counter for new keyframe
                steps_at_current_keyframe = 0
        
        # Check for termination
        if terminated or truncated:
            print(f"\n{'='*80}")
            if terminated:
                print(f"✅ Episode terminated (success) after {step_idx} steps")
            else:
                print(f"⚠️ Episode truncated after {step_idx} steps")
            print(f"{'='*80}")
            break
    
    else:
        # Loop completed without finishing sequence
        progress = controller.get_progress()
        print(f"\n{'='*80}")
        print(f"⏱️ Reached max steps ({max_steps})")
        print(f"Progress: Keyframe {progress[0]+1}/{progress[1]} ('{keyframe_name}')")
        print(f"{'='*80}")
    
    print("\nSmoke test complete. Close the viewer window to exit.")
    
    # Keep viewer open
    if env.viewer:
        while env.viewer.is_running():
            env.step(action)  # Hold final position


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=str, default="env/mujoco_assets")
    args = parser.parse_args()
    
    smoke_test_keyframe(Path(args.asset_root))

