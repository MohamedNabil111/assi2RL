## Discussion

### 0.1 Experiments

We trained DQN and DDQN agents on four classical control environments: CartPole-v1, Acrobot-v1, MountainCar-v0, and Pendulum-v1. Each agent was trained for 400-1000 episodes depending on environment complexity, then evaluated over 100 test episodes using greedy action selection.

**Training Setup:**

- Network: 3-layer neural network with 128-256 hidden units
- Optimizer: Adam with learning rates 0.0005-0.001
- Experience replay buffer: 10,000-50,000 transitions
- Batch size: 64, Discount factor: 0.99

**Test Results (100 episodes):**

| Environment    | DQN Performance  | DDQN Performance | Winner |
| -------------- | ---------------- | ---------------- | ------ |
| CartPole-v1    | 213.15 ± 8.12    | 500.00 ± 0.00    | DDQN   |
| Acrobot-v1     | -88.20 ± 39.28   | -85.42 ± 13.96   | DDQN   |
| MountainCar-v0 | -105.08 ± 11.73  | -116.01 ± 21.52  | DQN    |
| Pendulum-v1    | -179.50 ± 224.76 | -166.47 ± 92.28  | DDQN   |

**Key Findings:**

The most important discovery was that certain hyperparameters are critical for training success:

1. **Gradient Clipping (5-10)**: Without this, training failed completely. In one CartPole experiment, loss exploded from 10 to 42,485, causing the agent to forget everything it learned.

2. **Q-Value Clipping (±500)**: Prevented numerical instability across all environments.

3. **Epsilon Decay Rate**: Had to match reward structure. MountainCar failed completely with fast decay (0.995) but succeeded with slow decay (0.9999), discovering the goal at episode 460.

4. **Learning Rate**: Continuous control environments needed lower rates (0.0005 vs 0.001) to prevent oscillations.

The experiments showed that DDQN eliminates catastrophic forgetting in CartPole (achieving perfect 500±0 score) and reduces variance significantly in Acrobot (64% reduction) and Pendulum (59% reduction). Training time was nearly identical between DQN and DDQN.

---

### 0.2 Question Answers

#### Question 1: What is the difference between DQN and DDQN in terms of training time and performance?

**Training Time:**

DQN and DDQN have essentially the same training time. Both use identical network architectures and perform the same number of training steps. The only difference is that DDQN uses one extra forward pass through the policy network when calculating target values, which adds less than 2% overhead.

Measured times on our system:

- CartPole: 8-10 minutes (500 episodes)
- Acrobot: 12-15 minutes (500-600 episodes)
- MountainCar: 20-25 minutes (1000 episodes)
- Pendulum: 10-12 minutes (400 episodes)

**Performance:**

DDQN significantly outperforms DQN on most environments:

**CartPole**: DDQN achieved perfect 500±0 performance while DQN only reached 213±8. DQN suffered from catastrophic forgetting where the agent would reach 500 reward then suddenly drop to 10-20 and never recover.

**Acrobot**: Both performed well, but DDQN was more stable (-85±14 vs -88±39). DQN had occasional failures taking over 400 steps, while DDQN consistently solved in under 125 steps.

**MountainCar**: DQN slightly better (-105±12 vs -116±22). Both algorithms struggled initially due to sparse rewards but eventually learned reliably.

**Pendulum**: DDQN had lower variance (-166±92 vs -179±225), though both struggled with the discretized action space.

**Why DDQN is Better:**

DQN has an overestimation problem. It uses the same network to both choose and evaluate the best action:

```
Q_target = r + γ * max Q_target(s', a')
```

The max operator always picks the highest value, which tends to amplify estimation errors. These overestimated values spread through training, causing unstable learning.

DDQN fixes this by using different networks for selection and evaluation:

```
best_action = argmax Q_policy(s', a')
Q_target = r + γ * Q_target(s', best_action)
```

This separation prevents overestimation from compounding, leading to more accurate Q-values and stable training.

---

#### Question 2: Do you think the trained agents are good? Show with test episode reward figures.

Yes, the agents performed well on 3 out of 4 environments.

**CartPole-v1 (DDQN): Excellent**

- Result: 500.00 ± 0.00 (perfect score on all 100 test episodes)
- This is the maximum possible reward. The agent learned to balance the pole indefinitely.

**Acrobot-v1 (DDQN): Near-Optimal**

- Result: -85.42 ± 13.96 steps
- The physical optimum is around 70-80 steps, so we're very close.
- The agent efficiently builds momentum to reach the goal height.

**MountainCar-v0 (DQN): Good**

- Result: -105.08 ± 11.73 steps
- Random policy fails at -200, so reaching -105 is reliable success.
- The agent learned the momentum-building strategy.
- Note: Required 500 episodes before first discovering the goal.

**Pendulum-v1 (DDQN): Moderate**

- Result: -166.47 ± 92.28
- High variance is a problem. Best episodes reached -0.76 (nearly perfect), but typical performance is inconsistent.
- The issue is discretization: we split continuous actions into 25 bins, losing the fine control needed for precise balance.
- Continuous control algorithms (DDPG, TD3) typically achieve -50 to -100.

**Overall Assessment:**

Three environments show excellent to near-optimal performance. The Pendulum limitation is expected given we forced a continuous control problem into a discrete action framework. The agents successfully demonstrate proper implementation of experience replay, target networks, and exploration strategies.

Key success factors:

- Q-value clipping prevented divergence
- Gradient clipping prevented catastrophic forgetting
- Environment-specific hyperparameter tuning
- DDQN algorithm choice for stability

---

#### Question 3: What is the effect of each hyperparameter value on the RL training and performance?

We tested nine different hyperparameters. Here are the most important findings:

**Critical Parameters (training fails without proper values):**

**1. Gradient Clipping (tested: none, 1.0, 5.0, 10.0)**

- Without clipping: Complete failure. Loss exploded to 42,485 in CartPole, performance collapsed from 500 to 10-20 reward.
- Too strict (1.0): Agent stuck at 221 reward, couldn't learn properly.
- Optimal (5-10): Balanced learning and stability.
- Conclusion: This parameter is non-negotiable. Use 5.0 for simple environments, 10.0 for complex ones.

**2. Q-Value Clipping (tested: none, ±500, ±1000)**

- Without clipping: Q-values grow unbounded, causing numerical instability.
- With ±500: Stable training across all environments.
- Conclusion: Always use this. It prevents cascading overestimation.

**High Impact Parameters:**

**3. Epsilon Decay (tested: 0.995, 0.998, 0.9995, 0.9999)**

- Fast decay (0.995): Works for CartPole and Acrobot (dense rewards).
- Slow decay (0.9999): Essential for MountainCar. With 0.995, agent stuck at -200 for all 1000 episodes. With 0.9999, breakthrough at episode 460.
- Conclusion: Match decay rate to reward sparsity. Sparse rewards need 10x slower decay.

**4. Epsilon Min (tested: 0.01, 0.02, 0.05, 0.10)**

- Low (0.01): Good for dense rewards where greedy policy is optimal.
- High (0.10): Needed for MountainCar to avoid local minima.
- Conclusion: Sparse reward environments benefit from never fully stopping exploration.

**5. Learning Rate (tested: 0.0001, 0.0005, 0.001, 0.005)**

- Too low (0.0001): Very slow, may not learn in time.
- Optimal (0.001): Good for discrete control.
- Lower (0.0005): Needed for Pendulum due to discretization sensitivity.
- Too high (0.005): Oscillations and instability.

**Moderate Impact Parameters:**

**6. Target Network Update Frequency (tested: 10, 50, 100 steps)**

- Frequent (10): Good for CartPole and Acrobot with dense feedback.
- Infrequent (100): Better for MountainCar, provides stability for rare successful experiences.

**7. Replay Buffer Size (tested: 10k, 50k, 100k)**

- Small (10k): Sufficient for CartPole and Acrobot.
- Large (50k): Helpful for MountainCar to store diverse exploration experiences.

**Low Impact Parameters:**

**8. Batch Size (tested: 32, 64, 128)**

- 64 worked well everywhere. Minimal performance differences.

**9. Network Size (tested: 128, 256 hidden units)**

- 128: Sufficient for CartPole, Acrobot, Pendulum.
- 256: Helped MountainCar learn sparse reward patterns.

**Summary Table:**

| Environment | LR     | ε-decay | ε-min | Grad Clip | Why                               |
| ----------- | ------ | ------- | ----- | --------- | --------------------------------- |
| CartPole    | 0.001  | 0.995   | 0.01  | 5.0       | Fast task, needs stability        |
| Acrobot     | 0.001  | 0.995   | 0.01  | 5.0       | Dense rewards                     |
| MountainCar | 0.001  | 0.9999  | 0.10  | 10.0      | Sparse rewards, needs exploration |
| Pendulum    | 0.0005 | 0.998   | 0.02  | 10.0      | Continuous control sensitivity    |

**Key Lesson:** There's no universal configuration. Hyperparameters must match the environment's reward structure and complexity.

---

#### Question 4: From your point of view, how well-suited is DQN/DDQN to solve the problem?

DDQN is highly suitable for discrete control problems, but has limitations for continuous control and very sparse rewards.

**Where DQN/DDQN Excel:**

**CartPole and Acrobot (Most suitability):**

- Perfect match. Discrete actions, reasonable state spaces, learnable in 200-500 episodes.
- DDQN achieved perfect performance on CartPole and near-optimal on Acrobot.
- These environments showcase the algorithm's strengths: sample efficiency through experience replay, stable learning with proper hyperparameters.

**Where DQN/DDQN Struggle:**

**MountainCar (Less suitability):**

- Works but inefficient. Sparse delayed rewards make exploration difficult.
- Wasted 460 episodes before first goal discovery.
- Required very careful hyperparameter tuning (slow epsilon decay, high minimum exploration).
- Better alternatives exist: curiosity-driven exploration or hierarchical RL would find the goal much faster.

**Pendulum (Least suitability):**

- Poor match. This is fundamentally a continuous control problem.
- Our workaround of discretizing into 25 actions lost the precision needed for perfect balance.
- High variance (-166±92) reflects this limitation.
- Proper continuous control algorithms (DDPG, TD3, SAC) would perform much better.

**Strengths of DQN/DDQN:**

1. **Sample Efficiency**: Experience replay allows reusing each experience many times. CartPole solved in ~200 episodes vs 1000+ for simple policy gradient methods.

2. **Stability (DDQN)**: Eliminates catastrophic forgetting through decoupled action selection and evaluation. CartPole achieved zero variance (500±0).

3. **Simplicity**: Straightforward to implement and understand. Easy to debug by monitoring Q-values and loss.

4. **Off-Policy Learning**: Can learn from any experience, even from different policies. Useful for learning from demonstrations.

**Limitations of DQN/DDQN:**

1. **Continuous Actions**: Must discretize, losing precision. This hurt Pendulum performance significantly.

2. **Sparse Rewards**: Slow learning when successful experiences are rare. MountainCar needed 460 episodes just to discover the goal once.

3. **Hyperparameter Sensitivity**: Requires careful tuning. Wrong values for gradient clipping or epsilon decay cause complete failure.

4. **Overestimation Bias (DQN)**: The max operator amplifies errors, causing instability. This is why DDQN should always be preferred.

**When to Use DQN/DDQN:**

Good fit:

- Discrete action spaces (2-20 actions)
- Low to moderate state dimensions
- Dense or moderately sparse rewards
- Need for sample efficiency

Poor fit:

- Continuous action spaces
- Extremely sparse rewards (goal found <1% of time)
- Very high-dimensional spaces without proper architecture

**Recommendation:** Use DDQN as the default for discrete control problems. It has virtually no additional cost compared to DQN but provides much better stability. For continuous control, use actor-critic methods (DDPG, TD3, SAC) instead. For very sparse rewards, consider adding curiosity-driven exploration or reward shaping.

Our results demonstrate both strengths and limitations clearly: perfect CartPole, near-optimal Acrobot (strengths), inefficient MountainCar, imprecise Pendulum (limitations).

---

## Appendix: Hyperparameter Configurations

| Parameter     | CartPole-v1 | Acrobot-v1 | MountainCar-v0 | Pendulum-v1 |
| ------------- | ----------- | ---------- | -------------- | ----------- |
| Learning Rate | 0.001       | 0.001      | 0.001          | 0.0005      |
| Epsilon Decay | 0.995       | 0.995      | 0.9999         | 0.998       |
| Epsilon End   | 0.01        | 0.01       | 0.10           | 0.02        |
| Gradient Clip | 5.0         | 5.0        | 10.0           | 10.0        |
| Q-Value Clip  | ±500        | ±500       | ±500           | ±500        |
| Hidden Units  | 128         | 128        | 256            | 128         |
| Episodes      | 500         | 600        | 1000           | 400         |
