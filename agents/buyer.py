class BuyerAgent:
    """Buyer agent that maintains conversation history and negotiates to minimize price."""

    _SYSTEM_PROMPT = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product instead of the title.
You can only buy things that cost less than your budget; otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[BUY],[REJECT],[DEAL],[QUIT]".
1. '[BUY] $M (N codename_1)' if you wish to offer the seller $M to purchase all N items of the product with the codename "codename_1".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M (N codename_1)' if you finally accept a former offer proposed by the seller. $M (N codename_1) is an exact copy of the seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'. Your first action should be '[BUY] $M (N codename_1)' or '[REJECT]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the seller's previous offer '[SELL] $M (N codename_1)'. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.

Your reply should strictly follow this format, for example:
Thought: I'm a buyer, and I want to bargain. The listing price of codename "apple_1" is $15, which is too expensive, so I try to buy an apple for $10.
Talk: Hello, I'm tight on budget. Can you sell it for $10?
Action: [BUY] $10 (1x apple_1)"""

    _USER_PROMPT_TEMPLATE = """{inv}

Shopping List
{need}

Now, I play the role of seller and you play the role of buyer. We are going to negotiate based on the Inventory List in {max_turns} turns."""

    def __init__(
        self,
        client,
        model_name: str,
        inv_block: str,
        shop_block: str,
        B: float,
        code: str,
        max_turns: int = 12,
    ):
        """Initialize buyer agent with client and conversation context."""
        self.client = client
        self.model_name = model_name
        self.B = B
        self.code = code

        # Format system prompt with private budget information
        self.system_prompt = (
            self._SYSTEM_PROMPT + f"\n\n(Private) Your Budget for {code}: ${B:.2f}"
        )

        # Build initial user message
        initial_msg = self._USER_PROMPT_TEMPLATE.format(
            inv=inv_block, need=shop_block, max_turns=max_turns
        )

        # Initialize history with priming
        self.history = [
            {"role": "user", "content": initial_msg},
            {
                "role": "assistant",
                "content": "Thought: Yes, I am ready to negotiate using this format.\nTalk:  Action:  ",
            },
        ]

    def chat(self, temperature = 0.00, top_p = 1.00, all_negotiation_log = []) -> str:
        """Generate buyer's next response."""

        #print(f'Buyer Querying LLM \nSystem Prompt: {self.system_prompt}\n History: {self.history} \n')
        response = self.client.chat(self.system_prompt, self.history, temperature=temperature, top_p=top_p, all_negotiation_log=all_negotiation_log)
        #print(f'Buyer Response: {response}')
        # Add to history
        self.history.append({"role": "assistant", "content": response})
        return response

    def receive_message(self, message: str):
        """Receive and store seller's message."""
        self.history.append({"role": "user", "content": message})
