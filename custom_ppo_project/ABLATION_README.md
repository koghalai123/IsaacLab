# PPO Ablation Study - Quick Start

## Overview

This ablation study tests which PPO components are essential for learning by systematically disabling different mechanisms through **hyperparameter changes only** (no code modifications).

## Experiments

### 1. Baseline PPO (`ppo_params.yaml`)
Standard configuration - all components active.

### 2. Frozen Critic (`ppo_frozen_critic.yaml`)
- **critic_coef = 0.0** → No critic updates
- **tau = 1.0** → Pure Monte Carlo returns
- **Tests:** Is the critic necessary?

### 3. Lambda = 0 (`ppo_lambda_zero.yaml`)
- **tau = 0.0** → Pure 1-step TD (high bias)
- **Tests:** Can we learn with maximum critic reliance?

### 4. Gamma = 0 (`ppo_gamma_zero.yaml`)
- **gamma = 0.0** → No future planning (myopic agent)
- **Tests:** Is long-term planning necessary?

### 5. No Clipping (`ppo_no_clipping.yaml`)
- **e_clip = 1000** → Effectively no clipping
- **Tests:** Is PPO's clipping mechanism necessary?

## Running the Study

```bash
# View experiment configurations
python custom_ppo_project/ablation_summary.py

# Run all experiments (15 total: 5 variants × 3 seeds)
python run_experiments.py
```

## Expected Results

**Performance Ranking (Best → Worst):**
1. ✅ Baseline PPO - Should perform best
2. ⚠️ No Clipping - May work but unstable
3. ⚠️ Frozen Critic - Slower/noisier but can work
4. ❌ Lambda=0 - Struggles with inaccurate critic
5. ❌ Gamma=0 - Should fail completely (no planning)

## Analysis

View results in WandB: `https://wandb.ai/YOUR_ENTITY/isaac_lab_experiments`

**Key Metrics:**
- `rewards/iter` - Learning curves
- `episode_lengths/iter` - Task completion
- `losses/actor_loss` - Policy gradient magnitude
- `losses/critic_loss` - Value accuracy (if applicable)

**What to Look For:**
- Does frozen_critic match baseline? (Is critic needed?)
- Does no_clipping diverge? (Is clipping necessary?)
- Does gamma_zero fail? (Is planning necessary?)
- Does lambda_zero learn? (How important is GAE?)

## Files Created

```
custom_ppo_project/
├── ppo_params.yaml              # Baseline
├── ppo_frozen_critic.yaml       # Experiment 1
├── ppo_lambda_zero.yaml         # Experiment 2  
├── ppo_gamma_zero.yaml          # Experiment 3
├── ppo_no_clipping.yaml         # Experiment 4
├── ABLATION_STUDY.md            # Detailed documentation
└── ablation_summary.py          # Quick reference

run_experiments.py                # Updated experiment runner
```

## Configuration Changes Summary

| Experiment | γ | λ | critic_coef | e_clip | What Changes |
|------------|---|---|-------------|--------|--------------|
| Baseline | 0.99 | 0.95 | 4.0 | 0.2 | Nothing (reference) |
| Frozen Critic | 0.99 | 1.0 | **0.0** | 0.2 | Critic frozen, MC returns |
| Lambda=0 | 0.99 | **0.0** | 4.0 | 0.2 | Pure TD (1-step) |
| Gamma=0 | **0.0** | 0.95 | 4.0 | 0.2 | No future planning |
| No Clipping | 0.99 | 0.95 | 4.0 | **1000** | No policy clipping |

## Scientific Questions Answered

1. **Is the critic necessary?** → Compare baseline vs frozen_critic
2. **Is GAE's bias-variance tradeoff important?** → Compare baseline vs lambda_zero
3. **Is long-term planning necessary?** → Compare baseline vs gamma_zero
4. **Is PPO clipping necessary?** → Compare baseline vs no_clipping

## Notes

- All experiments use **hyperparameter changes only**
- 3 seeds per configuration for statistical robustness
- Same network architecture across all experiments
- Videos recorded for qualitative analysis
- Early stopping after 250 epochs without improvement

---

**Ready to run?**
```bash
python run_experiments.py
```

Results will appear in WandB as they complete!
