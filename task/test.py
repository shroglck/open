import jax
import numpy as np
import cv2
import json
import os
from collections import deque
import time

# --- OpenPi Imports ---
from openpi.policies import PolicyFactory

# Ensure this imports your environment file correctly
from mujoco_env import FrankaPickPlaceEnv

class Pi0InferenceWrapper:
    def __init__(
        self, 
        checkpoint_path, 
        action_dim=8, 
        # Pi0 usually handles chunks internally, but we keep this for execution timing
        num_actions_to_execute=4  
    ):
        print(f"Loading Pi0 Policy from {checkpoint_path}...")
        
        # OpenPi automatically loads the base model + LoRA adapter and config
        # It also handles normalization internally.
        self.policy = PolicyFactory.load_policy(checkpoint_path)
        
        print("Model loaded successfully.")
        
        self.num_actions_to_execute = num_actions_to_execute
        
        # Action Queue for Chunking
        self.action_queue = deque(maxlen=num_actions_to_execute)
        
        # Storage for current observation
        self.current_obs = None
        self.instruction = ""

    def reset(self, instruction):
        self.action_queue.clear()
        self.current_obs = None
        self.instruction = instruction.lower()
        
        # If the policy has internal state (RNNs), reset it
        if hasattr(self.policy, "reset"):
            self.policy.reset()
            
        print(f"Reset with instruction: {self.instruction}")

    def update_observation(self, obs):
        """
        Formats the Mujoco env observation into the dictionary Pi0 expects.
        Note: You may need to adjust the dictionary keys ('image', 'state') 
        to match exactly what your LoRA was trained on.
        """
        # 1. Process Image (Pi0 typically uses 224x224)
        img = obs["rgb_static"]
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (224, 224))
        
        # 2. Process Proprioception
        proprio = obs["proprio"].astype(np.float32)
        
        # 3. Store in the format expected by policy.infer()
        # The key names here depend on your dataset config during fine-tuning.
        # Common keys: 'image', 'images', 'state', 'proprio'
        self.current_obs = {
            "image": img,           # Pi0 usually expects a single image or dict of images
            "state": proprio,       # Joint states
            "instruction": self.instruction
        }

    def needs_inference(self):
        """Check if we need to run inference (queue is empty)"""
        return len(self.action_queue) == 0

    def get_action(self, obs, use_deltas=False):
        """
        Get next action from queue. Queue should already be populated.
        """
        if len(self.action_queue) == 0:
            raise RuntimeError("Action queue is empty! Call run_inference() first.")
        
        raw_action = self.action_queue.popleft()

        # Process Action (Gripper/Deltas)
        # Pi0 outputs usually match the dataset format (e.g., 7 dim arm + 1 dim gripper)
        arm_action = raw_action[:7]
        gripper_action = raw_action[7:] if len(raw_action) > 7 else np.array([0.04])

        if use_deltas:
            current_joints = obs["proprio"][:7]
            target_arm = current_joints + arm_action
        else:
            target_arm = arm_action

        return np.concatenate([target_arm, gripper_action])

    def run_inference(self):
        """Runs the model and fills the action_queue with first N actions from chunk"""
        if self.current_obs is None:
            raise ValueError("Observation not set. Call update_observation first.")

        # Run Inference
        # OpenPi's infer method usually returns a dictionary containing 'actions'
        # or a result object with an .actions attribute
        result = self.policy.infer(self.current_obs)
        
        # Extract actions. Depending on version, it might be result['actions'] or result.actions
        if isinstance(result, dict):
            actions = result['actions']
        else:
            actions = result.actions
        
        # Actions shape is typically (batch, chunk_size, action_dim) or (chunk_size, action_dim)
        # We ensure it's just (chunk_size, action_dim)
        if actions.ndim == 3:
            actions = actions[0]
            
        # Only take the first num_actions_to_execute actions
        num_to_use = min(self.num_actions_to_execute, len(actions))
        
        for i in range(num_to_use):
            self.action_queue.append(actions[i])

def save_video(frames, filepath, fps=25):
    """Saves a list of frames to an MP4 video file."""
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to: {filepath}")

def run_eval(checkpoint_path, num_episodes=5, record_video=True):
    # Initialize Policy
    # Note: dataset_stats are not manually passed for OpenPi; they are loaded from checkpoint config
    policy = Pi0InferenceWrapper(
        checkpoint_path, 
        action_dim=8, 
        num_actions_to_execute=4 # Controls how often we query the model (temporal ensembling)
    )
    
    env = FrankaPickPlaceEnv(gui=False) 
    
    # Create video directory
    video_dir = "eval_videos_pi0"
    os.makedirs(video_dir, exist_ok=True)

    modes = {"Base": False, "Hindered": True}

    for mode_name, is_hindered in modes.items():
        print(f"\n--- Testing Mode: {mode_name} ---")
        
        success_count = 0
        
        for ep in range(num_episodes):
            obs, info = env.reset(hindered=is_hindered)
            policy.reset(info["instruction"])
            
            # Update observation
            policy.update_observation(obs)
            
            frames = []
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated):
                # 1. Check if we need inference
                # 2. Execute K actions from the chunk WITHOUT updating observations
                
                # If queue empty, refill it
                if policy.needs_inference():
                    policy.run_inference()
                    
                # Determine how many actions we can actually execute from current queue
                # (Should be handled by deque popping, but safety check)
                if len(policy.action_queue) == 0:
                     policy.run_inference()

                # Get action from queue
                # Note: Pi0 is often trained with Absolute actions, check use_deltas
                action = policy.get_action(obs, use_deltas=False) 
                
                # Capture frame
                if record_video:
                    frame = env.render(mode="rgb_array")
                    frame = (frame * 255).astype(np.uint8)
                    frames.append(frame)

                # Execute action in environment
                step_result = env.step(action)
                obs = step_result.observation
                done = step_result.terminated
                truncated = step_result.truncated
                step += 1
                
                if step > 250: 
                    truncated = True
                
                # IMPORTANT: Update the policy's observation buffer 
                # Note: In the Octo script, this was done AFTER the chunk execution loop.
                # However, Pi0 often acts in a closed loop or the wrapper handles the "hold".
                # To replicate the Octo behavior exactly: only update obs when we are about to infer.
                # But here we update it every step so the wrapper has the latest when it *does* decide to infer.
                policy.update_observation(obs)

                if done or truncated:
                    break

            # Track success
            if done:
                success_count += 1

            # Save Video
            if record_video:
                status = "SUCCESS" if done else "FAIL"
                filename = f"{mode_name}_ep{ep}_{status}.mp4"
                save_video(frames, os.path.join(video_dir, filename))

            print(f"Episode {ep}: {status} (Steps: {step})")
        
        # Print summary
        success_rate = (success_count / num_episodes) * 100
        print(f"\n{mode_name} Mode Summary: {success_count}/{num_episodes} successes ({success_rate:.1f}%)")

if __name__ == "__main__":
    # Point this to the directory containing params, config.json and the adapter
    CHECKPOINT_PATH = "/mnt/sphere/nvme-backups/yixing/shresth/data/octo/checkpoint/pi0_finetune/experiment_2025..." 
    
    # We don't need STATS_PATH anymore, Pi0 checkpoints contain their own config
    run_eval(CHECKPOINT_PATH, num_episodes=200)