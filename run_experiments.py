import subprocess
import os
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Path to your python executable (using the one from the environment)
PYTHON_EXEC = sys.executable

# Path to the training script
TRAIN_SCRIPT = "custom_ppo_project/train_simple.py"

# WandB Settings
WANDB_PROJECT = "isaac_lab_experiments"
WANDB_ENTITY = "koghalai-uc-davis"  # Set this to your WandB username/team if needed, or pass via CLI

# Define the grid of hyperparameters to search over
algorithms = {
    "ppo": "custom_ppo_project/ppo_params.yaml",
    "sac": "custom_ppo_project/sac_params.yaml",
}
learning_rates = [0.0001]#[1e-3, 1e-4]
mlp_architectures = [
#    [64, 64],
    [256, 128, 64],
]
seeds = [42] # Add more seeds for robustness: [42, 100, 123]

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------

def run_experiments():
    experiment_count = 0
    total_experiments = len(algorithms) * len(learning_rates) * len(mlp_architectures) * len(seeds)

    print(f"Starting {total_experiments} experiments...")

    for algo_name, algo_config in algorithms.items():
        for lr in learning_rates:
            for units in mlp_architectures:
                for seed in seeds:
                    experiment_count += 1
                    
                    # Construct a unique name for this run
                    units_str = "_".join(map(str, units))
                    run_name = f"{algo_name}_lr{lr}_units{units_str}_seed{seed}"
                    
                    print(f"\n=== Running Experiment {experiment_count}/{total_experiments}: {run_name} ===")
                    
                    # Build the command
                    cmd = [
                        PYTHON_EXEC, TRAIN_SCRIPT,
                        "--track",  # Enable WandB
                        "--wandb-project-name", WANDB_PROJECT,
                        "--wandb-name", run_name,
                        "--agent_cfg_path", algo_config,
                        "--learning_rate", str(lr),
                        "--mlp_units", *map(str, units),
                        "--seed", str(seed),
                        "--headless", # Run without GUI
                        "--video", # Record video (optional, remove if not needed)
                        "--max_iterations", "1000" # Ensure short runs for testing
                    ]
                
                if WANDB_ENTITY:
                    cmd.extend(["--wandb-entity", WANDB_ENTITY])

                # Execute the command
                try:
                    # check=True raises CalledProcessError if the command fails
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running experiment {run_name}: {e}")
                    # Decide whether to continue or stop. Here we continue.
                    continue
                except KeyboardInterrupt:
                    print("\nExecution interrupted by user.")
                    return

    print("\nAll experiments completed.")

if __name__ == "__main__":
    # Ensure we are in the root directory
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Could not find {TRAIN_SCRIPT}. Please run this script from the workspace root.")
    else:
        run_experiments()
