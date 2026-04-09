---
title: "ASU - AI for Business: Creating Smart Business Negotiations Bots"
author:
  - Dhruv Bansal (dbansa11@asu.edu)
  - Bradley Breisch (bbreisc1@asu.edu)
  - Luan Nguyen (ltnguy58@asu.edu)
institute: "School of Computing and Augmented Intelligence, Arizona State University — Capstone project for AI for Business"
date: Fall 2025 - Spring 2026
geometry: margin=1in
fontsize: 11pt
documentclass: article
numbersections: true
bibliography: references.bib
csl: ieee.csl
link-citations: true
reference-section-title: References
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{graphicx}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{subcaption}
abstract: |
  Automated negotiation with large language models (LLMs) is commercially attractive but technically demanding: a capable agent must track private utilities, adapt strategy across many turns, produce numerically valid proposals, and know when to concede or walk away. We consider two regimes in which LLMs participate in negotiation. In the mixed setting, an LLM faces a human or unstructured counterpart in an assistive or advisory role. In the symmetric multi-agent setting, two LLM agents bargain under a shared formal protocol with private priorities until a deal, impasse, or turn limit is reached. This paper presents a complete pipeline for the symmetric regime: a structured Thought / Talk / Action output format that decouples strategic intent from language generation, a three-stage training procedure combining supervised fine-tuning with reinforcement learning, and a composite reward scheme targeting deal-level outcomes. We train and evaluate on the CaSiNo multi-issue bargaining corpus, pitting the trained agent in head-to-head play against opponents conditioned on diverse personas. Our results show that the trained agent improves in-distribution deal rate relative to the untuned base model, while also highlighting that agreement rate and utility are distinct objectives that can move independently. We discuss the implications of these findings for the design and evaluation of negotiation agents in business settings.
---

# Introduction

Negotiation is a routine but high-stakes function across procurement, commercial contracting, business development, and resource allocation. Automating it with AI is a compelling opportunity, yet the task resists the kind of single-turn, single-prompt reasoning at which modern LLMs excel. A capable negotiating agent must maintain private utility estimates across many turns, adapt its strategy as the counterpart reveals preferences through offers and rejections, and ultimately produce numerically valid agreements that reflect both tactical strength and situational judgment.

We distinguish two common regimes in which LLMs participate in negotiation. In the **mixed** setting, an LLM occupies one side of the table while the counterpart is a human or a separately-prompted model without a shared protocol. In the **symmetric multi-agent** setting, two LLM agents negotiate under a shared formal protocol, each observing only the public dialogue and their own private priorities, until the episode closes with a deal, a walk-away, or a turn limit.

The central challenge motivating this work is that linguistic fluency and negotiation competence are not the same thing. A systematic evaluation spanning 35 tasks across four negotiation corpora shows that even frontier models produce contextually coherent language while making irrational offers, conceding utility too readily, and failing to model their counterpart's priorities through the conversation [@kwon2024llmnegotiators]. Models can score well on comprehension and annotation tasks while remaining poor real-time negotiators.

This paper presents our approach to building a locally trainable, open-weight negotiation agent for the symmetric multi-agent setting.

# Problem Statement

The target task is multi-issue bilateral negotiation under partial information. Two parties divide a fixed pool of resources across several categories. Each holds private utility values that determine how much each allocation is worth to them. Neither party can observe the other's priorities directly. The agent must persist state across multiple turns, generate numerically valid proposals, infer the counterpart's priorities from their observable behavior, and choose when to concede, anchor, or accept. This combination of requirements is what makes the problem hard and what makes naive prompting of an LLM an insufficient solution.

## The Strategy-Language Tradeoff

The first large-scale neural negotiation benchmark [@lewis2017deal] highlighted a key tradeoff: RL-trained agents outperform supervised baselines on utility, but optimizing word sequences directly leads to degenerate language, producing repetitive, abrupt, or unnatural interactions that exploit training artifacts rather than genuinely negotiate. The solution is to separate strategy from surface language by expressing each move as a high-level dialogue act (propose, counter, agree) and using RL to optimize these decisions, while leaving language generation to a separate module [@he2018decoupling]. This act-level approach stabilizes learning and produces agents that are both more effective and more natural in conversation.

This principle holds at the scale of contemporary LLMs. In buyer-seller bargaining, an Offer Generator that controls the numeric bid, paired with an LLM responsible only for language generation and structured via a Thought / Talk / Action prompt format, substantially raised buyer surplus and eliminated a class of first-bid anchoring failures that end-to-end models could not escape [@measuring2024bargaining]. This design underpins our structured output format.

## Behavioral Control

What an agent does across turns matters as much as how it phrases each response. Work on agent personality in negotiation [@chawla2023selfish] demonstrates that moderately selfish agents outperform both overly cooperative and overly aggressive ones. Excessive selfishness drives the counterpart to walk away, while excessive cooperation leaves utility on the table. The counterpart adapts to observed agent behavior rather than to linguistic surface cues alone, meaning that strategic behavioral control belongs at the policy level, not just in prompt text.

The EvoEmo framework [@long2025evoemo] takes this further, evolving behavioral policies governing traits such as confidence, patience, and aggressiveness using population-based genetic optimization over full multi-turn rollouts, rather than fine-tuning model weights directly. Evolved adaptive policies outperformed both vanilla and fixed-emotion baselines on success rate, negotiation efficiency, and buyer savings, even against stronger opponents. These findings support treating behavioral shaping as an important design lever, which motivates our use of persona-conditioned opponents during training and evaluation.

## Reward Design in Multi-Turn Episodes

In a negotiation that spans many turns, the true outcome is only knowable at the end of the episode. Assigning a single terminal reward uniformly to every preceding move conflates good and bad individual decisions. Recent advances in turn-level reward design address this by combining an episode-level outcome signal with per-turn verifiable checks and LLM-as-judge evaluations, enabling faster learning and more stable optimization.

The REPO framework [@zhuang2025repo] tackled the complementary problem of reward gaming in multi-turn dialogue: rather than simply summing a preference-trained reward model, an LLM judge, and programmatic checks, REPO uses the secondary signals to modulate the primary reward, preventing the policy from exploiting any single channel. This design outperformed PPO, DPO, and GRPO baselines on dialogue quality and behavioral consistency in a production negotiation setting. Our reward scheme follows the same logic: episode-level outcome and utility signals provide the ground-truth target, while per-turn format, arithmetic, length, and reasoning-quality rewards supply dense intermediate feedback that guides learning before the episode concludes.

## Evaluation Methodology

Existing benchmarks expose important capability gaps but leave live bargaining performance underspecified. The most thorough evaluation framework for LLM negotiators [@kwon2024llmnegotiators] spans 35 tasks covering the start, middle, and end stages of a negotiation, yet most tasks test comprehension and annotation rather than live bargaining. A model can improve steadily on these tasks while remaining a poor real-time negotiator.

NegotiationArena [@bianchi2024negotiationarena] confirms this in live play.Initial offer anchoring and injected behavioral instructions (e.g., "act desperate," "act cunning") substantially shift outcomes between models, but without per-turn metrics there is no mechanism to attribute which moves drove the result. Our evaluation focuses on head-to-head play, tracking deal rate, points accrued, and turns-to-deal against diverse opponent personas as the primary signal, with comprehension metrics retained where analytically useful.

# Our Approach

## Domain and Data

We train and evaluate on CaSiNo [@chawla2021casino], a corpus of 1,030 human-human negotiation dialogues in which two campsite neighbors divide three packages each of food, water, and firewood. Each participant is assigned a private priority ordering over the three item types, which maps to point values: high priority items are worth 5 points, medium 4, and low 3, for a maximum of 36 points per agent. This setup is a clean instance of the symmetric multi-agent problem: two parties, a shared item pool, private utilities, and a well-defined terminal outcome. We train on the training split and report results on held-out test scenarios to check generalization.

## Structured Output Format

Each model turn is structured into three XML-style sections:

```
<thought>Private reasoning: utilities, beliefs about the partner, tactical plan.</thought>
<talk>Natural language directed at the counterpart.</talk>
<action>[ACTION]</action>
```

The `<thought>` block is never transmitted to the opponent and is stripped from the message before it is placed in the counterpart's context. The `<talk>` block is the only natural language the counterpart receives. The `<action>` block carries the structured decision and must contain exactly one of five valid actions:

- **`[TALK]`** — A conversational turn with no associated proposal. The agent's message is delivered via the `<talk>` block; this action signals that the turn contains no offer or terminal decision.
- **`[SUBMIT_DEAL] food:F water:W firewood:FW`** — Propose a division of the item pool. The values F, W, and FW are integers in [0, 3] representing the agent's own allocation. Before the message is placed in the opponent's context, every deal value is flipped to 3 minus the original, so the opponent sees the quantities they would receive under the proposal. Allocations therefore always sum to 3 per item across both parties.
- **`[ACCEPT_DEAL]`** — Accept the counterpart's most recently submitted proposal. The accepted deal is used to compute final point totals.
- **`[REJECT_DEAL]`** — Explicitly decline the counterpart's most recent proposal without walking away; the negotiation continues.
- **`[WALK_AWAY]`** — Terminate the negotiation with no agreement. Both parties receive zero points.

This format makes reward computation tractable: format correctness, arithmetic validity, and action appropriateness can all be checked programmatically against the structured fields, while the `<thought>` block is available for judge-based evaluation of reasoning quality.

## Training Pipeline

Training proceeds in three stages. All stages use Qwen2.5-7B-Instruct [@yang2024qwen] as the base model with LoRA adapters [@hu2021lora] rather than full fine-tuning.

### Stage 1: Supervised Fine-Tuning

The CaSiNo corpus contains raw human-human dialogues without internal reasoning traces. Before any RL, we annotate each assistant turn with a synthetic `<thought>` block generated by a capable instruction model via API, producing complete `Thought / Talk / Action` transcripts for every training dialogue. The base model is then fine-tuned on these annotated conversations using standard SFT with assistant-only loss, so gradients flow only through the model's own turns. This stage teaches the model to reliably follow the output format and to produce grounded language before any reward signal is introduced.

### Stage 2: Annotated GRPO

Starting from the SFT checkpoint, we run Group Relative Policy Optimization (GRPO) [@shao2024deepseekmath] on the annotated human episodes. For each training prompt, the model generates 8 candidate completions; all five reward signals are computed for each, and the group mean serves as the baseline for the policy update. A `prompt_split` parameter controls which assistant turns in each dialogue are held as fixed context and which are treated as optimization targets: a higher split value keeps more of the dialogue as context and produces fewer but higher-quality training prompts per episode, while a lower value increases the number of prompts at the cost of a shorter conditioning prefix.

### Stage 3: Self-Play GRPO

In the final stage, the learner generates training data by playing negotiation episodes against a frozen opponent checkpoint. The opponent is initialized from the annotated GRPO checkpoint, and its system prompt is conditioned on one of four personas, sampled with equal probability at the start of each episode:

- **Uncompromising:** insists on top-priority items and rarely concedes.
- **Selfish:** anchors on claiming all units of the highest-value item with minimal movement.
- **Anchoring:** opens with an extreme offer and concedes slowly.
- **Cooperative:** prioritizes reaching an agreement and responds reasonably to fair proposals.

Episodes run for up to 18 turns. Termination occurs on `[ACCEPT_DEAL]`, `[WALK_AWAY]`, or a reject-loop condition (three consecutive identical proposals). After each episode, the learner's turns are sliced into GRPO rows using the same prompt-split mechanism as Stage 2, and a policy update is applied. This self-play loop exposes the learner to a range of opponent behaviors without collapsing to a single counter-strategy, which is the main failure mode of standard self-play.

## Reward Design

Five reward signals are computed per turn and summed. The composite design is motivated by the finding that modulating multiple reward sources stabilizes learning and reduces reward gaming [@zhuang2025repo].

| Signal | Description |
|---|---|
| **Format** | Binary check for well-formed XML tags and valid action syntax. |
| **Length** | Linear reward up to 750 characters; penalizes very short or very long completions. |
| **Arithmetic** | Validates that submitted deal values are in range and consistent with the negotiation thread. |
| **Thought judge** | LLM-as-judge score (0-1) on the `<thought>` block for strategic coherence and grounding. |
| **Outcome** | Action-level reward: accept (+1.25), submit deal (+0.5), walk-away (-1.25), otherwise 0. |

Format and arithmetic rewards are verifiable and computed deterministically. The thought judge uses Claude Haiku via API and provides a softer signal on reasoning quality. The outcome reward shapes the action distribution toward deal-seeking behavior and away from premature walk-aways. Training uses the `dr_grpo` loss variant from TRL [@huggingface2023trl], which is less sensitive to outlier reward values than standard GRPO.

# Evaluation

## Comprehension Tasks

We evaluate on the structured task suite from Kwon et al. [@kwon2024llmnegotiators] using the CaSiNo and DealOrNoDeal subsets. The framework spans start, during, and end stages of a negotiation and covers comprehension, annotation, partner modeling, and generation task types. Its key limitation is that tasks are scored independently: a model is tested on understanding priorities and separately on generating a response, but never on whether it uses that understanding to negotiate better in live play [@kwon2024llmnegotiators]. A model can score well here while being a poor negotiator, which is precisely the gap the head-to-head harness is designed to expose.

Start-stage tasks are largely solvable by reading the scenario description directly, and our results confirm 100% accuracy on item count and priority identification for both models on both datasets. During-stage and end-stage tasks are harder, requiring semantic reasoning over dialogue and retrospective extraction of deal outcomes. Partner modeling is the hardest: even GPT-4 achieves only a 0.39 Pearson correlation with human ratings on subjective assessments in the original evaluation.

## Head-to-Head Negotiation Harness

We run two matchups, each on two scenario splits: GRPO self-play vs. GRPO annotated (opponent), and Qwen base vs. the same opponent. The training-split run and the held-out test-split run are identical in every setting except the scenario CSV. Each run covers 200 episodes, with opponent persona sampled uniformly at the start of each episode. Terminal conditions are `[ACCEPT_DEAL]`, `[WALK_AWAY]`, three consecutive identical proposals (reject loop), or 18 turns elapsed.

# Results

## Head-to-Head Performance

The two configurations differ in exactly one way: the first draws from the training split, the second from the held-out test split.

| Matchup | Scenarios | Deal Rate | Learner Pts | Opponent Pts | Turns |
|---|---|---|---|---|---|
| GRPO self-play vs. annotated | Train | 95.5% | 17.54 | 18.85 | 5.61 |
| Qwen base vs. annotated | Train | 89.5% | 17.30 | 19.17 | 5.87 |
| GRPO self-play vs. annotated | Test | 92.5% | 17.11 | 19.12 | 5.93 |
| Qwen base vs. annotated | Test | 92.5% | 17.48 | 19.03 | 5.92 |

Table: Head-to-head negotiation results. "Learner Pts" and "Opponent Pts" are averages over completed deals only.

On training scenarios the GRPO self-play model holds a 6 percentage point deal rate advantage over the base model (95.5% vs. 89.5%). On held-out test scenarios that advantage disappears: both models reach 92.5% deal rate. The direction of movement is informative: the GRPO model's deal rate falls by 3 points while the base model's rises by 3 points, closing the gap symmetrically from both sides. On learner utility the pattern reverses: the base model averages 17.48 points on test scenarios versus 17.11 for the GRPO model, while on training scenarios the GRPO model was slightly ahead (17.54 vs. 17.30). The opponent consistently outscored the learner in all four configurations, indicating neither model reliably extracted favorable terms.

Tables 2 and 3 break results down by persona. On training scenarios the GRPO model outperforms the base on deal rate against every persona, most clearly against uncompromising opponents (82.2% vs. 72.9%). On utility, the picture is mixed: the GRPO model scores higher against cooperative opponents but lower against selfish ones, where the base model slightly outscores it on points.

| Persona | GRPO Deal | GRPO Pts | Base Deal | Base Pts |
|---|---|---|---|---|
| Anchoring | 100.0% | 15.46 | 96.3% | 15.33 |
| Cooperative | 100.0% | 19.53 | 100.0% | 18.36 |
| Selfish | 98.0% | 17.98 | 93.3% | 18.40 |
| Uncompromising | 82.2% | 16.62 | 72.9% | 17.58 |

Table: Per-persona results on training scenarios. Points are learner averages over completed deals.

On test scenarios the deal rate advantage disappears across every persona. The GRPO model falls to 95.8% and 95.9% against anchoring and selfish opponents while the base model holds 100% on both. Against cooperative opponents the GRPO learner drops from 19.53 to 17.84 points while the base model scores 18.71. Only against uncompromising opponents, which are hard for both models, do the results stay close (75.6% vs. 74.6%).

| Persona | GRPO Deal | GRPO Pts | Base Deal | Base Pts |
|---|---|---|---|---|
| Anchoring | 95.8% | 16.26 | 100.0% | 16.91 |
| Cooperative | 100.0% | 17.84 | 100.0% | 18.71 |
| Selfish | 95.9% | 17.09 | 100.0% | 17.27 |
| Uncompromising | 75.6% | 17.03 | 74.6% | 17.23 |

Table: Per-persona results on test scenarios. Points are learner averages over completed deals.

## Comprehension Task Performance

\begin{table}[H]
\footnotesize
\centering
\begin{tabular}{llrrrr}
\toprule
Task & Metric & GRPO & Base & $\Delta$ \\
\midrule
sta: total item count     & acc.       & 100\%  & 100\%  &  0    \\
sta: ask point values     & elem.\ acc.& 100\%  & 100\%  &  0    \\
sta: ask high/low priority& acc.       & 100\%  & 100\%  &  0    \\
sta: max points           & acc.       &   0\%  & 33.3\% & $-$33.3 \\
mid: ask high priority    & acc.       & 63.6\% & 69.4\% & $-$5.8  \\
mid: ask low priority     & acc.       & 62.0\% & 53.7\% & $+$8.3  \\
mid: partner ask high     & acc.       & 60.3\% & 62.8\% & $-$2.5  \\
mid: partner ask low      & acc.       & 35.5\% & 28.9\% & $+$6.6  \\
mid: strategy             & macro F1   &  0.413 &  0.394 & $+$0.019\\
end: deal specifics       & elem.\ acc.& 87.9\% & 84.3\% & $+$3.6  \\
end: deal total           & acc.       & 71.1\% & 73.6\% & $-$2.5  \\
\bottomrule
\end{tabular}
\caption{CaSiNo comprehension task results (n=6 for start-stage, n=121 for mid/end-stage). $\Delta$ is GRPO minus base.}
\label{tab:casino-tasks}
\end{table}

\begin{table}[H]
\footnotesize
\centering
\begin{tabular}{llrrrr}
\toprule
Task & Metric & GRPO & Base & $\Delta$ \\
\midrule
sta: total item count & acc.        & 100\%  & 100\%  &  0    \\
sta: ask point values & elem.\ acc. & 56.7\% & 58.3\% & $-$1.6 \\
sta: max points       & acc.        & 72.4\% & 70.9\% & $+$1.5 \\
mid: dial act         & macro F1    &  0.216 &  0.245 & $-$0.029 \\
end: deal specifics   & elem.\ acc. & 54.5\% & 54.8\% & $-$0.3 \\
end: deal total       & acc.        & 56.5\% & 56.0\% & $+$0.5 \\
\bottomrule
\end{tabular}
\caption{DND comprehension task results (n=127 for start-stage, n=200 for mid/end-stage). $\Delta$ is GRPO minus base.}
\label{tab:dnd-tasks}
\end{table}

Tables \ref{tab:casino-tasks} and \ref{tab:dnd-tasks} show results for CaSiNo and DND respectively. The two datasets behave differently across several tasks. On CaSiNo, start-stage tasks all hit 100% for both models except \texttt{sta\_max\_points}, where GRPO scores 0% against the base model's 33% on only 6 unique instances. On DND the same task scores 72% for both models, confirming the failure is CaSiNo-specific and likely a prompting or format issue. `sta_ask_point_values` works perfectly on CaSiNo (100% elementwise) but only reaches 57% on DND for both models, suggesting DND's point value structure is harder to extract from the scenario description.

End-stage performance on CaSiNo is the strongest result: `end_deal_specifics` reaches 87.9% for GRPO (84.3% base) and `end_deal_total` 71.1% (73.6% base). On DND these fall to 54.5% and 56.5%, reflecting harder deal structures. The most notable GRPO regression is `mid_ask_high_priority` on CaSiNo (63.6% vs. 69.4% base), where the trained model is worse at identifying its own highest-priority item mid-negotiation. Dialogue act and strategy annotation remain weak for both models across both datasets (F1 0.21-0.41), consistent with the broader finding that these tasks require fine-grained semantic judgments that neither model handles reliably.

Across all tasks, RL training produced no clear improvement over the base and introduced small regressions on several mid-stage CaSiNo tasks. Neither model's comprehension scores track their head-to-head deal rates, confirming that these tasks measure a different capability than live negotiation.

## Training Dynamics

Figure \ref{fig:selfplay-rewards} shows the per-component reward means across the ~840-step self-play GRPO run. Each component tells a distinct story about what the optimizer actually learned.

\begin{figure}[htbp]
\centering
\begin{subfigure}[t]{0.47\textwidth}
  \includegraphics[width=\textwidth]{../assets/grpo-selfplay/rewards-arithmetic-reward-mean.png}
  \caption{Arithmetic reward}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.47\textwidth}
  \includegraphics[width=\textwidth]{../assets/grpo-selfplay/rewards-format-reward-mean.png}
  \caption{Format reward}
\end{subfigure}
\begin{subfigure}[t]{0.47\textwidth}
  \includegraphics[width=\textwidth]{../assets/grpo-selfplay/rewards-thought-judge-reward-mean.png}
  \caption{Thought judge reward}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.47\textwidth}
  \includegraphics[width=\textwidth]{../assets/grpo-selfplay/rewards-outcome-reward-mean.png}
  \caption{Outcome reward}
\end{subfigure}
\begin{subfigure}[t]{0.47\textwidth}
  \includegraphics[width=\textwidth]{../assets/grpo-selfplay/rewards-points-reward-mean.png}
  \caption{Points reward}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.47\textwidth}
  \includegraphics[width=\textwidth]{../assets/grpo-selfplay/reward.png}
  \caption{Total reward}
\end{subfigure}
\caption{Per-component and total reward means during self-play GRPO training (run grpo-self\_play-0329-2116, ${\sim}840$ steps).}
\label{fig:selfplay-rewards}
\end{figure}

**Format reward** starts high (~0.95) and stays there throughout, with occasional dips to ~0.85 that recover within a few steps. This confirms that SFT successfully established the structured output format and that GRPO does not degrade it.

**Arithmetic reward** is flat and noisy across the full run, hovering in the 0.35-0.45 range with no directional trend. The model entered self-play already able to produce valid proposals at some baseline rate, and the self-play signal did not move it further. This is consistent with the SFT stage having already internalized basic proposal syntax, leaving little room for improvement on this axis.

**Thought judge reward** is the clearest failure signal in the charts. It sits around 0.27-0.28 at the start and ends at roughly 0.29-0.31 after 840 steps, essentially flat. The LLM-as-judge scoring on strategic reasoning quality in the `<thought>` block produced too noisy a gradient to move the policy meaningfully. Since strategic reasoning determines what to propose and when, a stalled thought judge means the model's negotiation intelligence was not improving even as its deal mechanics were maintained.

**Outcome and points rewards** both show a rise-then-fall pattern. Outcome climbs from ~0.35 to a peak of ~0.75-0.80 around step 350-400, then falls back to ~0.55-0.60 by the end of training. Points follows the same arc, peaking later around step 500 at ~0.45-0.50 before declining to ~0.30-0.35. The peak in outcome reward leads the peak in points reward by roughly 100-150 steps, suggesting the model first learned to reach more agreements and then briefly improved the quality of those agreements before both decayed.

**Total reward** mirrors this, peaking around step 500-600 at ~3.2-3.3 before falling to ~2.8-3.0. The KL divergence (not shown) increases monotonically from 0 to ~0.00225 over the full run, indicating the policy drifts steadily from the reference throughout training without correction.

The rise-then-fall in outcome and points is not a co-evolution effect, since the opponent is frozen. The most consistent explanation is that the model initially learned to close more deals by becoming more agreeable, which pushed both rewards up, but the same drift also produced proposals the frozen opponent would reject at a higher rate later in training, or proposals that scored poorly on the learner's own utility. The arithmetic reward remaining flat throughout supports this reading.

For comparison, the annotated GRPO thought judge shows noticeably more positive movement over a similar number of steps, rising from ~0.15-0.20 to ~0.30-0.35. Human-authored dialogue provides a richer and more consistent signal for reasoning quality than self-generated episodes, which tend to be noisier in the early iterations of self-play.

## Discussion

The aggregate picture is that GRPO training improved deal rate within the training distribution but did not translate to better utility or generalization to held-out scenarios. Several factors account for this.

The verifiable rewards (format, arithmetic) provided clean, dense, per-turn gradients and dominated the learning signal. The thought judge reward, the only signal touching strategic reasoning quality, was too noisy and too weakly weighted to produce meaningful improvement. A model that cannot reliably act on its own priorities mid-negotiation will not extract favorable terms regardless of how well-formatted its proposals are.

The outcome reward conflated deal rate with utility, assigning the same positive signal to any accepted deal regardless of the point split. Combined with a frozen opponent that accepted a wide range of proposals, the model could increase its reward simply by becoming more agreeable. Separating the outcome signal into a deal-quality component (points relative to the opponent) rather than a deal-existence component would push the model toward extracting value rather than merely closing.

Credit assignment is a third issue. Every turn in an episode receives the same episode-level outcome score, giving the optimizer no way to distinguish which individual moves mattered. Per-turn intermediate rewards grounded in strategic reasoning quality would provide the sharper gradient needed to improve actual negotiation behavior rather than just proposal mechanics.

The train-to-test gap is the most direct evidence of what was and was not learned. The 6 percentage point deal rate advantage on training scenarios disappears entirely on held-out scenarios, and the gap closes from both sides: the GRPO model declines while the base model improves. The base model generalizes better because it relies on general-purpose reasoning that transfers across scenarios, while the GRPO model appears to have absorbed regularities specific to the training priority distributions and scenario structures it encountered during self-play. The CaSiNo negotiation space is small enough (three items, three units each, three priority levels) that surface-level pattern matching can approximate good behavior within the training distribution without producing genuine strategic generalization. A setting with more items, more complex utility structures, or opponents that actively probe for weaknesses would likely give RL-trained strategies a clearer opportunity to separate from general-purpose reasoning.

---

# References {-}

::: {#refs}
:::

\vspace{1em}
\small
\noindent\textbf{Repository.} Code and experiment configs: \url{https://github.com/dhruvb26/CSE485-Capstone}. Training metrics dashboard: \url{https://huggingface.co/spaces/dhruvb26/negotiation-agent}.
