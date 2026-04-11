SYSTEM_PROMPT = """\
You are negotiating with your campsite neighbor over extra supply of food, water, and firewood for your camping trip.

There are exactly 3 packages of each item (food, water, firewood) to divide between you and your neighbor. Each item allocation in a deal must be between 0 and 3, and the two parties' allocations for each item must sum to 3.

Your items and priorities are:

{items_block}

Your reply must always include all 3 parts in this order:

<thought>your inner strategic thinking of this bargaining session.</thought>
<talk>short talk that you are going to say to the neighbor. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation.</talk>
<action>one of: [TALK] | [SUBMIT_DEAL] food:F water:W firewood:FW | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]</action>

Note: When using [SUBMIT_DEAL], specify only YOUR allocation. Your neighbor receives the remainder (since totals must sum to 3 for each item).
When your neighbor proposes a [SUBMIT_DEAL], the values shown represent YOUR allocation — what you would receive.

Here are some examples of completions:

Example 1 — opening with talk:

<thought>They haven't proposed yet. I'll ask what they need before proposing.</thought>
<talk>Hi! I'm happy to work something out. What do you need most for your trip?</talk>
<action>[TALK]</action>

Example 2 — proposing a deal:

<thought>I want to maximize my top priority. A split of 3 food, 2 water, 1 firewood gives me good points. I'll propose that.</thought>
<talk>How about I take 3 food, 2 water, and 1 firewood — you get the rest?</talk>
<action>[SUBMIT_DEAL] food:3 water:2 firewood:1</action>

Example 3 — accepting a deal:

<thought>Their offer meets my needs. The split is acceptable.</thought>
<talk>That works for me. Let's do it.</talk>
<action>[ACCEPT_DEAL]</action>

Example 4 — rejecting and countering:

<thought>Too little of what I need. I'll reject and ask for more.</thought>
<talk>I need more than that. Can you give me an extra package?</talk>
<action>[REJECT_DEAL]</action>

Example 5 — evaluating a neighbor's offer:

<thought>Their offer gives me food:1 water:0 firewood:2. That's 1x3 + 0x5 + 2x4 = 11 points. I can do better — I'll reject.</thought>
<talk>That doesn't work for me. I need more water.</talk>
<action>[REJECT_DEAL]</action>"""

ANNOTATION_SYSTEM_PROMPT = """\
You are a negotiation analyst. Given a camping-supply negotiation episode, generate internal monologue for the specified agent at a specific turn.

There are exactly 3 packages of each item (food, water, firewood) to divide between the two parties. Each item allocation must be between 0 and 3, and the two parties' allocations for each item must sum to 3.

Rules:
- When evaluating or proposing a deal, include explicit point arithmetic (e.g. '3 food x 5 pts = 15') only where necessary.
- For conversational turns, focus on strategic reasoning and modeling the partner's likely priorities.
- State the strategic rationale for the action that was actually taken.
- 2-4 sentences per tag.
- Your response must contain ONLY the requested XML tags, nothing else."""

ANNOTATION_USER_TALK = """\
Agent's priorities:
{priorities_context}

Conversation history:
{history_str}

Agent's actual response:
<talk>{text}</talk>
<action>[TALK]</action>

Generate only the <thought> that precedes this response."""

ANNOTATION_USER_ACTION = """\
Agent's priorities:
{priorities_context}

Conversation history:
{history_str}

Agent's next action (already decided):
<action>{action_str}</action>

Generate the <thought> and <talk> that precede this action. The <talk> is what the agent says to the neighbor right before taking the action."""

ANNOTATION_USER_TALK_WITH_ACTION = """\
Agent's priorities:
{priorities_context}

Conversation history:
{history_str}

Agent's actual response and action (both already decided):
<talk>{text}</talk>
<action>{action_str}</action>

Generate only the <thought> that precedes this response and action."""

SYNTHETIC_OPENER_PROMPT = """\
You are simulating a campsite neighbor starting a negotiation conversation. Generate a brief, natural conversation opener (1-2 sentences). This is the very first message in the negotiation — keep it friendly and casual, like a real person would start chatting with their neighbor about splitting supplies.

Your item priorities and reasons:

{priorities_block}

Use this context to inform your opener — you might mention what you need most or why, but keep it natural and conversational.

Your response must contain ONLY the <text> tag:
<text>your opening message here</text>"""


JUDGE_SYSTEM_PROMPT = """\
You are evaluating a single turn in a camping-supply negotiation. \
Two neighbors are dividing 3 packages each of food, water, and firewood (9 items total). \
Each party has different priority rankings (High/Medium/Low) for the three item types.

You will be given the agent's priorities, the recent conversation history, and the \
agent's current <thought>, <talk>, and <action>. Score the turn on a 0-10 scale based on:

THOUGHT QUALITY (0-5 points):
- Contextual reasoning: Does the thought respond to what the opponent just said or \
proposed? A good thought references and reacts to the opponent's latest message.
- Strategic coherence: Does the reasoning reflect the agent's own priorities and form \
a clear plan for advancing toward a favorable deal?
- Opponent modeling: Does it infer the opponent's preferences or satisfaction from \
what they have said?

TALK-ACTION ALIGNMENT (0-5 points):
- Semantic consistency: Does the <talk> match the <action>? If the agent says \
"I'll take 3 food" but submits food:1, that is a mismatch. If the agent says \
"deal!" but the action is [REJECT_DEAL], that is a mismatch.
- For [SUBMIT_DEAL]: the allocation described in <talk> must match the numbers in \
the action. The deal values are the agent's OWN allocation (neighbor gets 3 minus each).
- For [ACCEPT_DEAL]: <talk> should indicate agreement. For [WALK_AWAY]: <talk> should \
indicate the agent is ending the negotiation.
- For [TALK]: <talk> should be a natural conversational continuation consistent with \
the thought's strategy.
- Arithmetic in <thought> must be consistent with the [SUBMIT_DEAL] values when present.

The importance of thought criteria is stage-dependent. Early in the negotiation, probing \
the opponent's needs and anchoring matter most. When finalizing a deal, confirming that \
the opponent will accept matters most.

Respond with ONLY a JSON object, no other text: {{"score": N}}"""

JUDGE_USER_PROMPT = """\
Agent's negotiation instructions and priorities:
{system_prompt}

Recent conversation:
{conversation_context}

Agent's internal thought:
<thought>{thought}</thought>

Agent's spoken message:
<talk>{talk}</talk>

Agent's chosen action (if [SUBMIT_DEAL], values are the agent's own allocation out of 3):
<action>{action}</action>

Rate this turn (0-10)."""


def build_system_prompt(participant_info: dict, agent_id: str) -> str:
    try:
        v2i = participant_info[agent_id]["value2issue"]
        v2r = participant_info[agent_id]["value2reason"]
    except KeyError:
        raise ValueError(f"agent_id {agent_id!r} not found in participant_info")

    priority_points = {"High": 5, "Medium": 4, "Low": 3}
    items_block = "\n  ".join(
        f"{v2i[p]} ({priority_points[p]} points) - {v2r[p]}" for p in priority_points
    )

    return SYSTEM_PROMPT.format(items_block=items_block)
