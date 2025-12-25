import numpy as np
import cv2
import os
import tyro
from collections import deque
import logging

# --- OpenPi Imports ---
from openpi.policies import policy_config
from openpi.training import config as _config

# --- Environment Import ---
from mujoco_env import FrankaPickPlaceEnv

class Pi0InferenceWrapper:
    def __init__(
        self, 
        checkpoint_dir: str, 
        config_name: str,
        num_actions_to_execute: int = 10,
    ):
        """
        Args:
            checkpoint_dir: Path to the directory containing 'params' (e.g. checkpoints/my_run)
            config_name: The name of the TrainConfig you registered in config.py
            num_actions_to_execute: Chunk execution horizon
        """
        print(f"Loading Config: {config_name}...")
        try:
            # 1. Load the configuration used for training
            # This ensures all transforms (PadState, DeltaActions) are identical to training.
            self.config = _config.get_config(config_name)
        except KeyError:
            raise ValueError(f"Config '{config_name}' not found. Did you add it to src/openpi/training/config.py?")

        print(f"Loading Policy from {checkpoint_dir}...")
        
        # 2. Create the policy
        # This loads the base model + LoRA adapters + applies the data transforms
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)
        print("Model loaded successfully.")
        
        self.num_actions_to_execute = num_actions_to_execute
        self.action_queue = deque(maxlen=num_actions_to_execute)
        self.current_obs_dict = None
        self.instruction = ""

    def reset(self, instruction: str):
        self.action_queue.clear()
        self.current_obs_dict = None
        self.instruction = instruction.lower()
        
        if hasattr(self.policy, "reset"):
            self.policy.reset()
            
        print(f"Wrapper reset with instruction: '{self.instruction}'")

    def update_observation(self, obs):
        """
        Formats the Mujoco env observation.
        """
        # 1. Image: Resize to 224x224 and ensure uint8
        img = obs["rgb_static"]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
            
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))
        
        # 2. State: Pass RAW 7-dim state
        # The 'PadStateTransform' in your config will handle padding it to 8 dims internally.
        state = obs["proprio"].astype(np.float32)

        # 3. Construct Input Dictionary
        # Keys must match the 'RepackTransform' in your config
        self.current_obs_dict = {
            "image": img,  
            "state": state,
            "prompt": self.instruction
        }

    def needs_inference(self):
        return len(self.action_queue) == 0

    def run_inference(self):
        if self.current_obs_dict is None:
            raise ValueError("Observation not set. Call update_observation first.")

        # Run Inference
        # Returns a dict, usually containing "actions"
        result = self.policy.infer(self.current_obs_dict)
        
        # Extract actions (Horizon, Action_Dim)
        actions = result["actions"]
        
        # If batched (1, Horizon, Dim), remove batch dim
        if actions.ndim == 3:
            actions = actions[0]
        print(actions.shape)    
        # Enqueue actions
        # Your config used 'AbsoluteActions' in outputs, so these are absolute positions.
        num_to_use = min(self.num_actions_to_execute, len(actions))
        
        for i in range(num_to_use):
            self.action_queue.append(actions[i])

    def get_action(self):
        if len(self.action_queue) == 0:
            raise RuntimeError("Queue empty.")
        return self.action_queue.popleft()


def save_video(frames, filepath, fps=20):
    if not frames: return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved video: {filepath}")

def main(
    checkpoint_dir: str,
    config_name: str = "pi05_custom_absolute_pose_lora", # MATCH THIS to your Training Config Name
    num_episodes: int = 5,
):
    # Setup Policy
    policy_wrapper = Pi0InferenceWrapper(checkpoint_dir, config_name)
    
    # Setup Env
    env = FrankaPickPlaceEnv(gui=False)
    os.makedirs("eval_videos", exist_ok=True)

    success_count = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset(hindered=True)
        instruction = info.get("instruction", "pick up the object")
        
        policy_wrapper.reset(instruction)
        policy_wrapper.update_observation(obs)
        
        frames = []
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            if policy_wrapper.needs_inference():
                policy_wrapper.run_inference()

            action = policy_wrapper.get_action()
            print(action.shape)
            # Step Env
            
            step_result = env.step(action)#np.concatenate([np.array(obs["proprio"]+action[:7]),action[7:8]]))
            obs = step_result.observation
            done = step_result.terminated
            truncated = step_result.truncated
            
            # Render
            frames.append((env.render(mode="rgb_array") * 255).astype(np.uint8))
            
            # Update Policy State
            policy_wrapper.update_observation(obs)
            
            step += 1
            if step > 400: truncated = True
            if done: break

        if done: success_count += 1
        status = "SUCCESS" if done else "FAIL"
        print(f"Episode {ep}: {status}")
        
        save_video(frames, f"eval_videos/ep{ep}_{status}.mp4")

    print(f"Success Rate: {success_count}/{num_episodes}")

if __name__ == "__main__":
    tyro.cli(main)