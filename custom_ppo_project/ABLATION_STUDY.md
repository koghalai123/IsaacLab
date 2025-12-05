# PPO Ablation Study Experiments

This document explains the different experimental configurations for ablation studies on the PPO algorithm.

## Baseline Configuration
**File:** `ppo_params.yaml`

Standard PPO configuration with all components active:
- **gamma (γ)**: 0.99 - Discount factor for future rewards
- **tau (λ)**: 0.95 - GAE lambda for bias-variance tradeoff
- **critic_coef**: 4 - Weight for critic loss
- **e_clip**: 0.2 - PPO clipping parameter
- **clip_value**: True - Value function clipping enabled

**What it does:** Full PPO with all mechanisms working as designed.

---

## Experiment 1: Frozen Critic
**File:** `ppo_frozen_critic.yaml`

**Modified Hyperparameters:**
```yaml
critic_coef: 0.0        # Zero weight on critic loss (no updates)
tau: 1.0                # Pure Monte Carlo (advantages from actual returns)
normalize_value: False  # Don't normalize value targets
clip_value: False       # No value clipping
```

**What it tests:**
- Removes critic updates completely (critic_coef = 0)
- Uses pure Monte Carlo returns for advantages (tau = 1.0)
- Actor learns from actual observed rewards, not value estimates

**Mathematical effect:**
```python
# Critic loss has zero weight:
total_loss = actor_loss + 0.0 * critic_loss - entropy + bounds_loss

# Advantages become pure Monte Carlo:
A[t] = G[t] - V(s[t])
     = (r[t] + γr[t+1] + γ²r[t+2] + ...) - V(s[t])
     # V(s[t]) is a frozen constant baseline
```

**Expected outcome:**
- **Higher variance** in advantage estimates (no learned baseline)
- **Slower learning** due to noisier gradient estimates
- Tests: "Is the critic necessary for learning?"

---

## Experiment 2: Lambda = 0 (Pure TD)
**File:** `ppo_lambda_zero.yaml`

**Modified Hyperparameters:**
```yaml
tau: 0.0  # Lambda = 0: Pure 1-step Temporal Difference
```

**What it tests:**
- Maximum reliance on critic's value estimates
- Only uses 1-step bootstrapping
- High bias, low variance advantage estimates

**Mathematical effect:**
```python
# GAE with λ=0 collapses to 1-step TD:
A[t] = δ[t]
     = r[t] + γ * V(s[t+1]) - V(s[t])
     # Only looks 1 step ahead, then trusts critic
```

**Expected outcome:**
- **Low variance** but **high bias** (trusts possibly-wrong value estimates)
- **Fast updates** but potentially wrong direction if critic is inaccurate
- Tests: "Can we learn with maximum critic reliance?"

---

## Experiment 3: Gamma = 0 (No Future Planning)
**File:** `ppo_gamma_zero.yaml`

**Modified Hyperparameters:**
```yaml
gamma: 0.0  # No discounting: only immediate rewards matter
```

**What it tests:**
- Agent becomes completely myopic (short-sighted)
- Only cares about immediate rewards
- No credit assignment to past actions

**Mathematical effect:**
```python
# Returns collapse to immediate rewards:
G[t] = r[t] + 0.0 * r[t+1] + 0.0 * r[t+2] + ...
     = r[t]

# TD error becomes:
δ[t] = r[t] + 0.0 * V(s[t+1]) - V(s[t])
     = r[t] - V(s[t])
```

**Expected outcome:**
- **Cannot learn** tasks requiring multi-step reasoning
- Agent will fail on tasks where immediate rewards don't indicate success
- Tests: "Is future planning necessary for this task?"

---

## Experiment 4: No Clipping (Vanilla Policy Gradient)
**File:** `ppo_no_clipping.yaml`

**Modified Hyperparameters:**
```yaml
e_clip: 1000.0    # Effectively infinite (no clipping constraint)
clip_value: False # Also disable value clipping
```

**What it tests:**
- Removes PPO's key innovation (clipped surrogate objective)
- Allows arbitrarily large policy updates
- Reverts to vanilla policy gradient behavior

**Mathematical effect:**
```python
# Without clipping:
ratio = π_new(a|s) / π_old(a|s)
L_actor = ratio * advantage  # No min(ratio, clip(ratio))

# Policy can change drastically in one update:
ratio can be >> 1.2 or << 0.8 (normally clipped to [0.8, 1.2])
```

**Expected outcome:**
- **Training instability** - policy may collapse
- **Large policy swings** between updates
- **Possible divergence** if learning rate isn't very small
- Tests: "Is PPO clipping necessary for stable learning?"

---

## Comparison Matrix

| Experiment | γ | λ | critic_coef | e_clip | clip_value | Tests |
|------------|---|---|-------------|--------|------------|-------|
| **Baseline** | 0.99 | 0.95 | 4.0 | 0.2 | True | Full PPO |
| **Frozen Critic** | 0.99 | 1.0 | 0.0 | 0.2 | False | No critic updates |
| **Lambda=0** | 0.99 | 0.0 | 4.0 | 0.2 | True | Pure TD (high bias) |
| **Gamma=0** | 0.0 | 0.95 | 4.0 | 0.2 | True | No future (myopic) |
| **No Clipping** | 0.99 | 0.95 | 4.0 | 1000 | False | Vanilla PG |

---

## Expected Results

### Performance Ranking (Best to Worst):
1. **Baseline PPO** - Should perform best with all components working
2. **No Clipping** - Might work but with higher variance/instability
3. **Frozen Critic** - Can work but slower/noisier learning
4. **Lambda=0** - Will struggle if critic is inaccurate early in training
5. **Gamma=0** - Will likely fail on multi-step reasoning tasks

### Key Insights to Look For:

1. **Frozen Critic vs Baseline:**
   - If similar: Critic may not be crucial for this task
   - If worse: Critic's learned baseline reduces variance significantly

2. **Lambda=0 vs Baseline:**
   - If similar: Task is simple enough for 1-step TD
   - If worse: Need multi-step credit assignment (GAE helps)

3. **Gamma=0 vs Baseline:**
   - If similar: Task is purely reactive (rare!)
   - If worse: Long-term planning is essential

4. **No Clipping vs Baseline:**
   - If similar: Task is well-conditioned, policy gradient naturally stable
   - If worse: PPO clipping prevents catastrophic updates

---

## How to Run

The `run_experiments.py` script will automatically run all configurations:

```bash
python run_experiments.py
```

Results will be logged to WandB under project: `isaac_lab_experiments`

Each experiment will run with 3 seeds (42, 100, 123) for statistical robustness.

---

## Analysis Tips

### In WandB, compare:

1. **Learning curves**: rewards/iter across all experiments
2. **Variance**: Look at reward standard deviation over seeds
3. **Stability**: Check for divergence or policy collapse
4. **Sample efficiency**: Steps to reach target performance
5. **Final performance**: Maximum reward achieved

### Questions to Answer:

- Which component matters most for this task?
- Can we simplify PPO without losing performance?
- Where does PPO spend its "complexity budget"?
- Which ablation breaks learning completely?

---

## Implementation Notes

All experiments use **hyperparameter changes only** - no code modifications required.

The experiments test:
- ✅ Critic necessity (frozen_critic)
- ✅ Bias-variance tradeoff (lambda_zero)
- ✅ Future planning (gamma_zero)
- ✅ Update stability (no_clipping)

This comprehensive ablation study will reveal which PPO components are essential vs. optional for your specific robotics task.
