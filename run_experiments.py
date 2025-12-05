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
    "ppo_baseline": "custom_ppo_project/ppo_params.yaml",
    "ppo_frozen_critic": "custom_ppo_project/ppo_frozen_critic.yaml",  # critic_coef=0, tau=1.0
    "ppo_lambda_zero": "custom_ppo_project/ppo_lambda_zero.yaml",      # tau=0 (pure TD)
    "ppo_gamma_zero": "custom_ppo_project/ppo_gamma_zero.yaml",        # gamma=0 (no future)
    "ppo_no_clipping": "custom_ppo_project/ppo_no_clipping.yaml",      # e_clip=1000
#    "sac": "custom_ppo_project/sac_params.yaml",
}
learning_rates = [0.0001]#[1e-3, 1e-4]
mlp_architectures = [
#    [64, 64],
    [256, 128, 64],
]
seeds = [42, 100, 123] # Multiple seeds for robustness
tasks = [
    #"Isaac-Stack-Cube-Franka-v0",
    "Isaac-Factory-PegInsert-Direct-v0",
#    "Isaac-Open-Drawer-Franka-v0",
#    "Isaac-Lift-Cube-Franka-v0",
]

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------

def run_experiments():
    experiment_count = 0
    total_experiments = len(tasks) * len(algorithms) * len(learning_rates) * len(mlp_architectures) * len(seeds)

    print("="*80)
    print("PPO ABLATION STUDY - EXPERIMENT RUNNER")
    print("="*80)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"  • Tasks: {len(tasks)}")
    print(f"  • Algorithm variants: {len(algorithms)}")
    print(f"  • Learning rates: {len(learning_rates)}")
    print(f"  • Architectures: {len(mlp_architectures)}")
    print(f"  • Seeds: {len(seeds)}")
    print(f"\nAlgorithm variants:")
    for algo_name in algorithms.keys():
        print(f"  • {algo_name}")
    print("\n" + "="*80 + "\n")

    for task in tasks:
        for algo_name, algo_config in algorithms.items():
            for lr in learning_rates:
                for units in mlp_architectures:
                    for seed in seeds:
                        experiment_count += 1
                        
                        # Construct a unique name for this run
                        units_str = "_".join(map(str, units))
                        run_name = f"{task}_{algo_name}_lr{lr}_units{units_str}_seed{seed}"
                        
                        print(f"\n{'='*80}")
                        print(f"Experiment {experiment_count}/{total_experiments}: {run_name}")
                        print(f"Config: {algo_config}")
                        print(f"{'='*80}\n")
                        
                        # Build the command
                        cmd = [
                            PYTHON_EXEC, TRAIN_SCRIPT,
                            "--task", task,
                            "--num_envs", "2048", # Reduce number of environments to avoid OOM
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
                            print(f"\n✓ Experiment {experiment_count} completed successfully\n")
                        except subprocess.CalledProcessError as e:
                            print(f"\n✗ Error running experiment {run_name}: {e}\n")
                            # Decide whether to continue or stop. Here we continue.
                            continue
                        except KeyboardInterrupt:
                            print("\n\nExecution interrupted by user.")
                            return

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"\nSuccessfully ran {experiment_count} experiments.")
    print(f"View results at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    print("\nNext steps:")
    print("  1. Compare learning curves across algorithm variants")
    print("  2. Look for instability in 'no_clipping' variant")
    print("  3. Check if 'frozen_critic' achieves similar performance")
    print("  4. Verify 'gamma_zero' fails (confirms importance of future planning)")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Ensure we are in the root directory
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Could not find {TRAIN_SCRIPT}. Please run this script from the workspace root.")
    else:
        run_experiments()
