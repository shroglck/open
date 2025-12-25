"""
Script for converting a custom RLDS/TFDS dataset (with absolute pose actions) to LeRobot format.

Usage:
uv run convert_custom_data_to_lerobot.py --data_dir /path/to/your/data

"""

import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

# UPDATE THIS: The name of your TFDS dataset as registered or located in data_dir
RAW_DATASET_NAMES = ["cubes_pick_place"] 
REPO_NAME = "shrg7/franka" 

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset based on the provided JSON schema
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="unknown", # Update this if you know the specific robot (e.g., "panda", "ur5")
        fps=10, # Ensure this matches your data collection FPS
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3), # Matches "rgb_static" in schema
                "names": ["height", "width", "channel"],
            },
            # "proprio" from schema maps to "state" here
            "state": {
                "dtype": "float32",
                "shape": (7,), # Matches "proprio" dimensions
                "names": ["state"],
            },
            # Actions are absolute pose positions (Shape 8 based on schema)
            "actions": {
                "dtype": "float32",
                "shape": (8,), # Matches "action" dimensions
                "names": ["actions"], 
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        # Load the dataset using TFDS
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                # Extract features according to the provided schema
                
                # Handle text instruction (decode bytes if necessary)
                instruction = step.get("language_instruction", b"")
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

                dataset.add_frame(
                    {
                        # Schema: observation -> rgb_static
                        "image": step["observation"]["rgb_static"],
                        
                        # Schema: observation -> proprio
                        "state": step["observation"]["proprio"],
                        
                        # Schema: action (Absolute Pose)
                        "actions": step["action"],
                        
                        "task": instruction,
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["custom", "rlds", "absolute_pose"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)