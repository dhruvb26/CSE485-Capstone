class BuyerAgent:
    """Stateful negotiation agent that plays the buyer role.

    Maintains a conversation history and uses a chat client to generate
    responses following a structured Thought/Talk/Action format. The buyer's
    goal is to purchase a product at the lowest possible price without
    exceeding its private budget.
    """

    system_prompt = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
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

    user_prompt_template = """{inv}

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
        """Set up the buyer with its client, product context, and private budget.

        Constructs the system prompt (with the private budget appended) and
        primes the conversation history with the inventory/shopping context
        and a ready-to-negotiate assistant message.

        Args:
            client: A chat client instance implementing
                ``chat(instructions, messages) -> str``.
            model_name: Identifier of the model backing this agent.
            inv_block: Seller's inventory listing text (from
                :func:`~benchmark.utils.inventory_list`).
            shop_block: Buyer's shopping list text (from
                :func:`~benchmark.utils.shopping_list`).
            B: The buyer's private maximum budget for this product.
            code: Product codename (e.g. ``"electronics_3"``).
            max_turns: Number of negotiation rounds communicated to the agent.
        """
        self.client = client
        self.model_name = model_name
        self.B = B
        self.code = code

        self.system_prompt = (
            self.system_prompt + f"\n\n(Private) Your Budget for {code}: ${B:.2f}"
        )

        initial_msg = self.user_prompt_template.format(
            inv=inv_block, need=shop_block, max_turns=max_turns
        )

        self.history = [
            {"role": "user", "content": initial_msg},
            {
                "role": "assistant",
                "content": "Thought: Yes, I am ready to negotiate using this format.\nTalk:  Action:  ",
            },
        ]

    def chat(self) -> str:
        """Generate the buyer's next negotiation response.

        Sends the full conversation history to the underlying chat client,
        appends the generated response to history, and returns it.

        Returns:
            The raw text response from the model, expected to contain
            Thought/Talk/Action sections.
        """
        response = self.client.chat(self.system_prompt, self.history)
        self.history.append({"role": "assistant", "content": response})
        return response

    def receive_message(self, message: str):
        """Append the seller's latest message to the conversation history.

        Args:
            message: The seller agent's raw text response.
        """
        self.history.append({"role": "user", "content": message})
