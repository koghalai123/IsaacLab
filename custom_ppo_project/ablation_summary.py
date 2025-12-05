"""
Quick reference for the ablation study experiments.

Run: python run_experiments.py

This will test 5 different configurations:
"""

experiments = {
    "ppo_baseline": {
        "gamma": 0.99,
        "tau": 0.95,
        "critic_coef": 4.0,
        "e_clip": 0.2,
        "clip_value": True,
        "description": "Standard PPO - all components active",
        "hypothesis": "Should achieve best performance"
    },
    
    "ppo_frozen_critic": {
        "gamma": 0.99,
        "tau": 1.0,      # Pure Monte Carlo
        "critic_coef": 0.0,  # No critic updates
        "e_clip": 0.2,
        "clip_value": False,
        "description": "Critic frozen - no value function updates",
        "hypothesis": "Higher variance, slower learning. Tests: Is critic necessary?"
    },
    
    "ppo_lambda_zero": {
        "gamma": 0.99,
        "tau": 0.0,      # Pure 1-step TD
        "critic_coef": 4.0,
        "e_clip": 0.2,
        "clip_value": True,
        "description": "Lambda=0 - maximum critic reliance (high bias)",
        "hypothesis": "Low variance but trusts potentially wrong value estimates"
    },
    
    "ppo_gamma_zero": {
        "gamma": 0.0,    # No future planning
        "tau": 0.95,
        "critic_coef": 4.0,
        "e_clip": 0.2,
        "clip_value": True,
        "description": "Gamma=0 - myopic agent, only immediate rewards",
        "hypothesis": "Will fail on tasks requiring multi-step reasoning"
    },
    
    "ppo_no_clipping": {
        "gamma": 0.99,
        "tau": 0.95,
        "critic_coef": 4.0,
        "e_clip": 1000.0,  # No clipping
        "clip_value": False,
        "description": "No clipping - vanilla policy gradient",
        "hypothesis": "Training instability, possible policy collapse"
    }
}

print("="*80)
print("PPO ABLATION STUDY - EXPERIMENT CONFIGURATIONS")
print("="*80)
print()

for exp_name, config in experiments.items():
    print(f"\n{exp_name.upper().replace('_', ' ')}")
    print("-" * 60)
    print(f"Description: {config['description']}")
    print(f"Hypothesis:  {config['hypothesis']}")
    print()
    print("Parameters:")
    print(f"  • gamma (γ):        {config['gamma']:6.2f}  {'← MODIFIED' if config['gamma'] != 0.99 else ''}")
    print(f"  • tau (λ):          {config['tau']:6.2f}  {'← MODIFIED' if config['tau'] != 0.95 else ''}")
    print(f"  • critic_coef:      {config['critic_coef']:6.1f}  {'← MODIFIED' if config['critic_coef'] != 4.0 else ''}")
    print(f"  • e_clip:           {config['e_clip']:6.1f}  {'← MODIFIED' if config['e_clip'] != 0.2 else ''}")
    print(f"  • clip_value:       {str(config['clip_value']):6s}  {'← MODIFIED' if config['clip_value'] != True else ''}")
    print()

print("="*80)
print("KEY INSIGHTS TO LOOK FOR:")
print("="*80)
print("""
1. BASELINE vs FROZEN_CRITIC:
   → Tests if the critic is necessary for learning
   → Look for: Higher variance, slower convergence in frozen_critic

2. BASELINE vs LAMBDA_ZERO:
   → Tests the bias-variance tradeoff in GAE
   → Look for: Faster but potentially incorrect learning in lambda_zero

3. BASELINE vs GAMMA_ZERO:
   → Tests necessity of long-term planning
   → Look for: Complete failure in gamma_zero (agent is myopic)

4. BASELINE vs NO_CLIPPING:
   → Tests PPO's key innovation (clipping)
   → Look for: Training instability, policy collapse in no_clipping

Expected Performance Ranking:
  1. Baseline PPO (best)
  2. No Clipping (unstable but might work)
  3. Frozen Critic (slower but can work)
  4. Lambda=0 (struggles early)
  5. Gamma=0 (likely fails completely)
""")

print("="*80)
print("ANALYSIS IN WANDB:")
print("="*80)
print("""
Compare these metrics across all experiments:
  • rewards/iter - Learning curve
  • episode_lengths/iter - Task completion
  • losses/actor_loss - Policy gradient magnitude
  • losses/critic_loss - Value function accuracy (where applicable)
  • kl_divergence - Policy change per update

Group by: experiment name
Average over: 3 seeds (42, 100, 123)
""")

print("="*80)
print(f"Total experiments to run: {len(experiments)} configs × 3 seeds = {len(experiments) * 3} runs")
print("="*80)
