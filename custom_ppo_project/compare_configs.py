#!/usr/bin/env python3
"""
Compare all ablation study configurations side-by-side.
Usage: python compare_configs.py
"""

import yaml
from pathlib import Path

configs = {
    "Baseline": "ppo_params.yaml",
    "Frozen Critic": "ppo_frozen_critic.yaml",
    "Lambda=0": "ppo_lambda_zero.yaml",
    "Gamma=0": "ppo_gamma_zero.yaml",
    "No Clipping": "ppo_no_clipping.yaml",
}

# Key parameters to compare
params_to_check = [
    ("gamma", ["params", "config", "gamma"]),
    ("tau (λ)", ["params", "config", "tau"]),
    ("critic_coef", ["params", "config", "critic_coef"]),
    ("e_clip", ["params", "config", "e_clip"]),
    ("clip_value", ["params", "config", "clip_value"]),
    ("normalize_value", ["params", "config", "normalize_value"]),
]

def get_nested_value(d, keys):
    """Get value from nested dict using list of keys."""
    for key in keys:
        d = d.get(key, {})
    return d

def load_config(filename):
    """Load YAML config file."""
    path = Path("custom_ppo_project") / filename
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("\n" + "="*100)
    print(" "*30 + "PPO ABLATION STUDY - CONFIG COMPARISON")
    print("="*100 + "\n")
    
    # Load all configs
    loaded_configs = {}
    for name, filename in configs.items():
        try:
            loaded_configs[name] = load_config(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return
    
    # Print header
    col_width = 16
    print(f"{'Parameter':<20}", end="")
    for name in configs.keys():
        print(f"{name:>{col_width}}", end="")
    print()
    print("-" * (20 + col_width * len(configs)))
    
    # Print each parameter
    for param_name, param_path in params_to_check:
        print(f"{param_name:<20}", end="")
        
        baseline_value = get_nested_value(loaded_configs["Baseline"], param_path)
        
        for name in configs.keys():
            value = get_nested_value(loaded_configs[name], param_path)
            
            # Format value
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            elif isinstance(value, bool):
                value_str = str(value)
            elif isinstance(value, int):
                value_str = str(value)
            else:
                value_str = str(value)
            
            # Highlight if different from baseline
            if name != "Baseline" and value != baseline_value:
                value_str = f"⚠️ {value_str}"
            
            print(f"{value_str:>{col_width}}", end="")
        print()
    
    print("-" * (20 + col_width * len(configs)))
    
    # Print interpretations
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100 + "\n")
    
    interpretations = [
        ("⚠️ = Modified from baseline", ""),
        ("", ""),
        ("Frozen Critic:", "critic_coef=0 (no updates), tau=1.0 (MC returns)"),
        ("Lambda=0:", "tau=0 (pure 1-step TD, high bias)"),
        ("Gamma=0:", "gamma=0 (myopic, no future planning)"),
        ("No Clipping:", "e_clip=1000 (no policy constraint)"),
    ]
    
    for line1, line2 in interpretations:
        if line1:
            print(f"  {line1:<40} {line2}")
    
    print("\n" + "="*100)
    print("EXPECTED OUTCOMES")
    print("="*100 + "\n")
    
    outcomes = [
        ("Baseline", "✓ Best performance - all components working"),
        ("No Clipping", "⚠️ Unstable training, possible divergence"),
        ("Frozen Critic", "⚠️ Higher variance, slower learning"),
        ("Lambda=0", "⚠️ Early struggles (biased advantages)"),
        ("Gamma=0", "✗ Complete failure (cannot plan ahead)"),
    ]
    
    for name, outcome in outcomes:
        print(f"  {name:<20} → {outcome}")
    
    print("\n" + "="*100)
    print(f"Total configurations: {len(configs)}")
    print(f"Seeds per config: 3")
    print(f"Total experiments: {len(configs) * 3}")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
