# GRPO Negotiation Agent Blueprint

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Output Schema](#2-output-schema)
3. [Phase 0: SFT Pre-Training](#3-phase-0-sft-pre-training)
4. [Phase 1: Starting Task - Setup](#4-phase-1-starting-task---setup)
5. [Phase 2: GRPO During Task - Self-Play](#5-phase-2-grpo-during-task---self-play)
6. [Phase 3: End Task - Terminal Reward and Dialogue Rollouts](#6-phase-3-end-task---terminal-reward-and-dialogue-rollouts)
7. [How C1-C4 Map to the Agent](#7-how-c1-c4-map-to-the-agent)
8. [Known Failure Modes and Mitigations](#8-known-failure-modes-and-mitigations)
9. [Toolstack and Training Schedule](#9-toolstack-and-training-schedule)

---

## 1. Architecture Overview

The agent produces three outputs at every turn.
```
LLM-based Agent
    |
    |---> thought   (internal chain-of-thought, never shown to opponent)
    |---> talk      (natural language utterance shown to opponent)
    |---> action    (JSON offer or deal decision)
```

**Key design principle:** `thought` carries the C1-C3 reasoning load (comprehension, annotation, partner modeling). `talk` + `action` together carry C4 (generation).

---

## 2. Output Schema

Every agent turn produces the following JSON:

```json
{
  "thought": "My values: food=3pts, water=4pts, firewood=5pts. Max possible = 3+12+15=30pts. Partner just offered food:2, water:1, firewood:0 for me. That gives me 6+4+0=10pts out of 30. Weak offer. I should counter and push for firewood. Partner has mentioned needing food twice - they likely value it most.",
  "talk": "I appreciate the offer, but firewood is essential for me given the cold nights ahead. How about I take 2 firewood and 1 water, and you take the rest?",
  "action": {
    "type": "counter",
    "food": 0,
    "water": 1,
    "firewood": 2
  }
}
```

**`action.type` options:** `offer`, `counter`, `accept`, `reject`

**Allocation constraint (hard rule):** for each item, `agent_amount + partner_amount <= total_available (3)`. The verifier enforces this.

**`thought` field must always include:**
- Explicit arithmetic: `item x quantity x pts_per_item = subtotal`
- Current estimate of partner's highest priority item
- Justification for the chosen action type

---

## 3. Phase 0: SFT Pre-Training

### Goal

Teach the model the three-output JSON schema before any RL begins. This is not about making the model a good negotiator - only about making it output valid, schema-compliant JSON reliably. Target: format validity above 95% of outputs before moving to GRPO.

### Why This is Necessary

Cold-starting GRPO from base model produces malformed outputs in early episodes. The format reward component then dominates before any strategic learning can happen, wasting training budget.

### Dataset Construction (~1,000 turn-level samples)

Each sample is a **single agent turn**, not a full dialogue. A 10-turn dialogue produces roughly 5 agent-turn samples (the agent speaks on alternating turns).

| Source | Dialogues | Agent Turns | Notes |
|---|---|---|---|
| CaSiNo | ~50 dialogues | ~400 samples | Primary domain, richest annotations |
| DealOrNoDeal | ~30 dialogues | ~180 samples | Simpler arithmetic, good format diversity |
| CraigslistBargain | ~25 dialogues | ~150 samples | Richer `talk` language, persuasion variety |
| **Total** | **~105 dialogues** | **~730-1,000 samples** | Pad with more CaSiNo to reach 1k |

**Prioritize turns where:**
- The `action` is a concrete offer or counter (not an opening greeting or simple agreement)
- The dialogue has reached at least turn 2 so `thought` has meaningful context
- The `thought` field contains at least one arithmetic calculation

**NOTE:** Do not over-represent opening pleasantries and agreement turns - they are trivially easy for the model and waste SFT capacity.

### Constructing the `talk` and `action` Fields

These come directly from the source datasets:

- **CaSiNo:** `talk` from chat logs, `action` from deal outcome annotations and per-turn offer structure
- **DealOrNoDeal:** `talk` from dialogue, `action` from the structured offer representation already in the dataset
- **CraigslistBargain:** `talk` from dialogue, `action` inferred from the dialogue act annotation (propose/counter/agree)

### Constructing the `thought` Field

No human negotiation dataset contains internal reasoning. You must synthesize it. Use two methods:

**Rule-based synthesis (for arithmetic-heavy turns):**

Given the known `value2issue` point assignments from CaSiNo metadata, a script can produce deterministic thoughts:

```python
def build_thought(my_values, partner_offer, dialogue_history):
    max_pts = sum(qty * val for qty, val in my_values.items())
    offer_pts = sum(partner_offer[item] * my_values[item] for item in partner_offer)
    pct = offer_pts / max_pts
    partner_priority = infer_partner_priority(dialogue_history)
    return (
        f"My values: {format_values(my_values)}. Max possible = {max_pts}pts. "
        f"Partner offered {format_offer(partner_offer)} = {offer_pts}pts ({pct:.0%} of max). "
        f"Partner priority estimate: {partner_priority}. "
        f"Decision: {'accept' if pct > 0.6 else 'counter'}."
    )
```

**GPT synthesis (for strategic/persuasion turns):**

For the ~40% of CaSiNo turns with strategy annotations (self-need, elicit-preference, vouch-fairness, no-need), prompt GPT:

```
Given this CaSiNo negotiation context:
- My point values: {value2issue}
- Dialogue so far: {dialogue_history}
- My next utterance will use the strategy: {strategy_label}
- My next utterance is: {talk}

Write a concise internal thought (2-4 sentences) that explains the reasoning 
leading to this utterance. Include point arithmetic where relevant.
```

### SFT Training Config

Can be found in the `rl/sft.config.yaml` file for now. 

---

## 4. Phase 1: Starting Task - Setup

### Heuristic Verifiers (No RL Yet)

Before any training turn is accepted, run these rule-based checks:

| Verifier | Check | Failure Penalty |
|---|---|---|
| Allocation validity | `sum(agent + partner) <= 3` for each item | Reject sample entirely |
| Arithmetic correctness | `thought` contains correct point calculation | Flag for reward penalty |
| Format validity | Output parses as valid JSON with all required keys | Reject sample entirely |
| No-hallucination | `action` does not claim items outside scenario | Reject sample entirely |

These verifiers serve double duty: they filter bad SFT samples during Phase 0, and they become explicit reward signal components in Phase 2.

### CoT Prompt Template

Force the model to calculate values step-by-step before every action decision. Include this in the system prompt:

```
Before responding, always:
1. List your point value for each item (item x quantity x pts = subtotal)
2. Calculate the maximum points you could achieve
3. Estimate what your partner values most based on what they have said
4. Decide your action type (offer, counter, accept, reject)
5. Then produce your talk and action fields
```

---
Here is the updated Section 5 in full:

---

## 5. Phase 2: GRPO During Task - Self-Play

### Why GRPO Over PPO

GRPO is better suited for negotiation than standard PPO for two reasons:

1. **Sparse rewards.** Negotiation episodes are multi-turn with reward only at the end. GRPO handles this by sampling G=8 candidate responses per turn, scoring all of them, and updating to favor above-average candidates. This normalization reduces variance without a separate value network.

2. **No critic needed.** Training a separate critic model alongside a 7-8B policy is expensive. GRPO eliminates this.

The advantage estimate is:

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

### Self-Play Setup

The clone opponent is a copy of the learner model. The critical design decision is **never train against a purely cooperative clone**. The Be Selfish But Wisely paper shows that human-human negotiation data is ~80% agreements, so a clone trained on it becomes prosocial and accepts almost anything. The learner then learns to be maximally greedy without ever making concessions - a strategy that collapses immediately against real humans.

**Cycle through these clone personas during training:**

| Persona | Behavior | Purpose |
|---|---|---|
| Uncompromising | Insists on highest-priority items, rarely concedes | Forces learner to learn strategic concessions |
| Selfish | Claims 2-3 units of top-valued item in every offer | Forces learner to handle anchoring |
| Anchoring | Opens extreme, moves slowly | Trains patience and counter-anchoring |
| Cooperative | Occasionally used as a baseline | Prevents over-optimization for adversarial conditions |

**Persona pair structure:** assign asymmetric scenarios so there is always a non-zero zone of possible agreement. If the learner has firewood as highest priority (5pts), give the clone a scenario where water is highest priority. This prevents early training collapse from constant deadlocks.

### Multi-Source Reward Model

A single terminal reward is too sparse. Decompose into five components:

| Component | Signal | Weight | Notes |
|---|---|---|---|
| Terminal joint payoff | (own pts + partner pts) / max joint pts | 0.50 | Primary signal. Joint payoff prevents deadlock-seeking behavior |
| Format validity | JSON parse success + schema check | 0.15 | Training wheel - decays to 0 in GRPO Phase 2 |
| Arithmetic correctness | Verifier checks `thought` arithmetic | 0.15 | Permanent - measures factual correctness, not strategy |
| Strategy quality | Flan-T5-small strategy classifier on `talk` | 0.10 | Training wheel - decays to 0 in GRPO Phase 2 |
| Partner model accuracy | Inferred vs. actual partner priority | 0.10 | Permanent - measures factual correctness, not strategy |

### Reward Phasing

Intermediate rewards introduce a credit assignment conflict. A move that looks strategically poor in isolation (a generous concession at turn 5) might be exactly the right move for closing a deal at turn 8. Rewarding it negatively at turn 5 penalizes the action that caused the good terminal outcome.

To resolve this, rewards are split into two categories and phased across training:

| Component | Type | GRPO Phase 1 | GRPO Phase 2 | Rationale |
|---|---|---|---|---|
| Terminal joint payoff | Terminal | Active | Active | Always the primary signal |
| Format validity | Intermediate | Active | Decayed to 0 | Training wheel - SFT should have solved this |
| Arithmetic correctness | Intermediate (thought-level) | Active | Active | Factual correctness - does not conflict with terminal signal |
| Strategy quality | Intermediate | Active | Decayed to 0 | Strategic quality should emerge from terminal reward, not be prescribed |
| Partner model accuracy | Intermediate (thought-level) | Active | Active | Factual correctness - does not conflict with terminal signal |

**The distinction:** arithmetic correctness and partner model accuracy reward verifiable facts about the `thought` field - the calculation is either right or wrong regardless of whether a deal is reached. Format and strategy rewards are prescriptive - they tell the model how to behave rather than letting the terminal signal teach it. Phase these out once the model has the basics stable.

**Decay schedule:** linearly decay format and strategy reward weights from their Phase 1 values to 0 over the first 1,000 GRPO episodes of Phase 2. Do not decay abruptly - a hard cutoff creates a sudden shift in the reward landscape that destabilizes training.

**Terminal deal score formula:**

```
R_terminal = (own_pts / max_own_pts) + lambda * (partner_pts / max_partner_pts)
```

Set `lambda = 0.3-0.5`. This reflects the finding from Be Selfish But Wisely that a selfish-but-not-completely-selfish agent outperforms both fully selfish and fully cooperative agents in human trials.

**Walkaway penalty:** assign `-0.2` (not `0`) when no deal is reached. A zero reward for walkaway gives the agent no incentive to avoid stalemates. A small negative reward teaches it that any deal is better than deadlock.

**Strategy quality reward:** train a Flan-T5-small classifier on CaSiNo strategy annotations to detect self-need, elicit-preference, vouch-fairness, and no-need in the `talk` field. Reward the agent when it uses information-gathering strategies (elicit-preference) in early turns and self-need strategies in mid-turns, before making hard proposals.

**Partner model reward:** the `thought` field must contain an explicit estimate of partner priority. Compare this estimate against the ground-truth `value2issue` metadata from CaSiNo. This reward is only available during training (when you have ground truth) and directly trains the Theory-of-Mind capability that the paper identifies as the weakest area for all open-source LLMs.

### GRPO Training Loop

```
For each episode:
  1. Sample a CaSiNo scenario (item counts: 3,3,3; random point values)
  2. Assign a persona to the clone from the persona pool
  3. Run negotiation for up to 10 turns
     - At each learner turn: sample G=8 candidates
     - Score each with the multi-source reward model
     - Compute GRPO advantage across the group of 8
     - Update policy to increase log-prob of positive-advantage candidates
  4. Every N episodes: sync clone weights from learner
```

**Training config:**

```
LoRA rank:               16-32
LoRA targets:            q_proj, v_proj, k_proj, o_proj
Quantization:            4-bit during inference, full precision for LoRA update
Candidates per turn (G): 8
Max turns per episode:   10
Clone sync interval:     Every 200-500 episodes
KL penalty coefficient:  0.1 (prevents policy from drifting too far from SFT checkpoint)
Reward decay window:     First 1,000 episodes of Phase 2 for format and strategy weights
```

---

## 6. Phase 3: End Task - Terminal Reward and Dialogue Rollouts

### Terminal Reward

Calculated once at dialogue end. Uses joint payoff (see formula above) to ensure the model learns to reach agreements rather than win aggressively.

The `end_deal_total_ca` CaSiNo task directly measures whether the model can calculate its own final points. The `thought` field arithmetic requirement is the fix for this - requiring explicit step-by-step calculation at every turn means the model never loses track of the running score.

### Dialogue Rollouts for Final Turns

At turns T-2 and T-1 (second-to-last and last turns of the negotiation), switch to rollout mode:

```
1. Generate K=5 candidate responses from the learner
2. For each candidate, run the clone forward until dialogue concludes
   (deal, rejection, or 10-turn timeout)
3. Score each complete trajectory with the terminal reward function
4. Select the candidate that leads to highest expected terminal reward
```

This is a one-step lookahead. It is computationally expensive but limited to the final 2-3 turns, so overhead is manageable. This addresses the core failure mode identified in the Are LLMs Effective Negotiators paper: models generate plausible individual responses but fail to reason about how those responses affect the final deal outcome.

---

## 7. How C1-C4 Map to the Agent

C1-C4 are the paper's evaluation capability categories, not a sequential pipeline. They all happen within a single turn:

| Capability | Where it lives in the agent | When it is most active |
|---|---|---|
| C1 - Comprehension | `thought` field | Every turn. Reads scenario, tracks item counts and point values |
| C2 - Annotation | `thought` field | Every turn. Interprets incoming opponent utterance (dialogue act, strategy) |
| C3 - Partner Modeling | `thought` field | Heaviest in turns 1-4. Updates partner priority estimate from dialogue |
| C4 - Generation | `talk` + `action` fields | Every turn. Produces the response and structured offer |

**Implication for early vs. late turns:**

Early turns (1-3) should be C3-heavy in the `thought` field and produce elicit-preference utterances in `talk` ("What do you need most?"). The `action` field in early turns may be a soft opening offer. Late turns shift to C4-dominant behavior where `thought` is shorter, `action` carries more weight, and the dialogue rollout kicks in.

---

## 8. Known Failure Modes and Mitigations

### Overagreeable Behavior
The paper identifies that Mistral-7B accepts unfair offers and fails to push back - a consequence of prosocial human-human training data.

**Mitigation:** Heavy use of adversarial clone personas. Add a small reward bonus for successfully countering an offer that gives the agent less than 40% of maximum possible points.

### Arithmetic Failure on End Tasks
Mistral-7B achieves only 36.4% accuracy on End-stage comprehension tasks zero-shot, primarily `end_deal_total_ca`.

**Mitigation:** The CoT requirement in the `thought` field forces explicit `item x qty x pts = subtotal` arithmetic at every turn. CoT prompting on arithmetic tasks brings accuracy near 100% for models that otherwise fail (shown for GPT-4 in the paper; same principle applies).

### Degenerate Self-Play Collapse
Two greedy RL agents get stuck repeating their opening offer with no movement - documented in Be Selfish But Wisely.

**Mitigation:** (1) Penalize repeated `action` fields across consecutive turns. (2) Require `action` to differ from the previous turn's action (hard constraint in generation prompt). (3) Add a diversity bonus for novel offer structures.

### Subjective Task Weakness
Even GPT-4 achieves only 0.30 PCC on satisfaction prediction. The model sometimes predicts the opposite of the true label.

**Mitigation:** Do not include subjective task performance in the training reward. Track it as a held-out evaluation metric only.

---

## 9. Toolstack and Training Schedule

### Recommended Stack

| Component | Tool |
|---|---|
| Base model | Mistral-7B-Instruct-v0.3 |
| SFT + GRPO training | TRL library with GRPO trainer + PEFT for LoRA |
| Clone inference (self-play) | vLLM for fast rollouts |
| Strategy classifier (reward) | Flan-T5-small fine-tuned on CaSiNo strategy annotations |
| `thought` field synthesis (SFT data) | Rule-based scripts + GPT-4 API |