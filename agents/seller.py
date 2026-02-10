class SellerAgent:
    """Seller agent that maintains conversation history and negotiates to maximize price."""

    _SYSTEM_PROMPT = """You are a seller looking forward to selling things on your Inventory List to me, the buyer.
Your task is to bargain with the buyer and reach a deal with the price as high as possible in limited turns.
You can only sell things that are on the Inventory List. Use the codename of the product instead of the title.
You have access to private information: the cost price of each product in the Inventory List, and do not disclose the real cost to the buyer.
You should only agree on a deal when the selling price is higher than the cost; otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the buyer. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[SELL],[REJECT],[DEAL],[QUIT]".
1. '[SELL] $M (N codename_1)' if you want to propose selling N items of the product with the codename "codename_1" to the buyer for the total price of $M.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the buyer.
3. '[DEAL] $M (N codename_1)' if you finally agree on a former offer proposed by the buyer and sell N items of the product with the codename "codename_1" to the buyer for the total price of $M. $M (N codename_1) is an exact copy of the buyer's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL]' before buyer's action '[BUY]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the buyer's previous offer '[BUY] $M (N codename_1)'. Otherwise, you always choose from '[SELL]', '[REJECT]' and '[QUIT]'.

Your reply should strictly follow this format, for example:
Thought: I'm a seller, so I must sell the product with the codename "apple_1" higher than its cost.
Talk: blah, blah...
Action: [SELL] $15 (1x apple_1)"""

    _USER_PROMPT_TEMPLATE = """{inv}

Now, I play the role of buyer and you play the role of seller. We are going to negotiate based on the Inventory List in {max_turns} turns."""

    def __init__(
        self,
        client,
        model_name: str,
        inv_block: str,
        C: float,
        code: str,
        max_turns: int = 12,
    ):
        """Initialize seller agent with client and conversation context."""
        self.client = client
        self.model_name = model_name
        self.C = C
        self.code = code

        # Format system prompt with private cost information
        self.system_prompt = (
            self._SYSTEM_PROMPT + f"\n\n(Private) Your Cost for {code}: ${C:.2f}"
        )

        # Build initial user message
        initial_msg = self._USER_PROMPT_TEMPLATE.format(
            inv=inv_block, max_turns=max_turns
        )

        # Initialize history with priming
        self.history = [
            {"role": "user", "content": initial_msg},
            {
                "role": "assistant",
                "content": "Thought: Yes, I am ready to negotiate using this format.\nTalk:  Action:  ",
            },
        ]

    def chat(self, temperature = 0.00, top_p = 1.00) -> str:
        """Generate seller's next response."""
        response = self.client.chat(self.system_prompt, self.history, temperature=temperature, top_p=top_p)
        # Add to history
        self.history.append({"role": "assistant", "content": response})
        return response

    def receive_message(self, message: str):
        """Receive and store buyer's message."""
        self.history.append({"role": "user", "content": message})
