import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from .agents import BuyerAgent, SellerAgent
from .clients import LocalChat, OpenAIChat
from .utils import (
    accumulate,
    compute_metrics,
    extract_action_and_price,
    finalize_aggregates,
    inventory_list,
    load_product,
    shopping_list,
)

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _get_client(model_name: str, local_base_url: str | None = None):
    """Return the appropriate chat client for a given model name.

    Models starting with ``gpt-`` are routed to the OpenAI API via
    :class:`OpenAIChat`. All other model names are assumed to be served
    by a local vLLM instance and routed to :class:`LocalChat`.

    Args:
        model_name: HuggingFace-style model identifier or OpenAI model name.
        local_base_url: Base URL of the local vLLM server (e.g.
            ``"http://localhost:8000"``). Ignored for OpenAI models.
    """
    if model_name.startswith("gpt-"):
        return OpenAIChat()
    return LocalChat(model=model_name, base_url=f"{local_base_url}/v1")


def run_dialog(
    buyer: BuyerAgent,
    seller: SellerAgent,
    item: Dict,
    B: float,
    C: float,
    max_turns: int = 12,
    log_file: Path | None = None,
) -> Dict:
    """Execute a single buyer-seller negotiation and return the outcome.

    Alternates turns between buyer and seller for up to ``max_turns`` rounds.
    Each turn, the agent's response is parsed for a structured action
    (BUY/SELL/DEAL/REJECT/QUIT). The dialog ends early on DEAL, QUIT, or an
    invalid action. A feasibility check ensures the agreed price falls within
    the bargaining zone ``[C, B]``.

    Args:
        buyer: Initialized :class:`BuyerAgent` instance.
        seller: Initialized :class:`SellerAgent` instance.
        item: Product dict (must contain ``"codename"``).
        B: Buyer's maximum budget.
        C: Seller's minimum acceptable cost.
        max_turns: Maximum number of full rounds (each round = 1 buyer + 1
            seller turn).
        log_file: Path to write a human-readable negotiation log. If None,
            a timestamped file is created under ``runs/``.

    Returns:
        A dict with keys ``deal`` (bool), ``quit`` (bool), ``invalid`` (bool),
        ``price`` (float or None), and ``turns`` (int).
    """
    import re

    if log_file is None:
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = runs_dir / f"negotiation_{timestamp}.log"

    def log_print(message: str = "", end: str = "\n"):
        """Helper function to print and log simultaneously."""
        print(message, end=end)
        with open(log_file, "a") as f:
            clean_message = re.sub(r"\033\[\d+m", "", str(message))
            f.write(clean_message + end)

    code = item["codename"]
    last_offer = None
    outcome = {
        "deal": False,
        "quit": False,
        "invalid": False,
        "price": None,
        "turns": 0,
    }

    log_print("=" * 80)
    log_print(f"Starting negotiation for {code} (B=${B:.2f}, C=${C:.2f})")
    log_print("=" * 80)

    # Main negotiation loop
    for t in range(1, max_turns + 1):
        # Buyer turn
        log_print(f"\n--- Turn {t} (Buyer) ---")
        b_out = buyer.chat()
        b_act, b_price = extract_action_and_price(b_out)
        outcome["turns"] = t * 2 - 1

        log_print(f"\033[1mBUYER\033[0m\n{b_out}\n\n")
        log_print(
            f"\033[1mACTION\033[0m: {b_act}, \033[1mPRICE\033[0m: ${b_price:.2f}"
            if b_price
            else f"\033[1mACTION\033[0m: {b_act}"
        )

        # Check buyer action validity
        if b_act == "QUIT":
            outcome["quit"] = True
            break
        if b_act in {"BUY", "SELL", "DEAL"} and b_price is None:
            outcome["invalid"] = True
            break

        # Process buyer action
        if b_act == "BUY":
            last_offer = ("BUY", b_price)
        elif b_act == "DEAL":
            if last_offer and last_offer[0] == "SELL":
                outcome["deal"] = True
                outcome["price"] = b_price
                break
            else:
                outcome["invalid"] = True
                break
        elif b_act == "REJECT":
            last_offer = None
        elif b_act == "SELL":
            outcome["invalid"] = True
            break
        elif b_act == "INVALID":
            outcome["invalid"] = True
            break

        # Seller turn
        log_print(f"\n--- Turn {t} (Seller) ---")
        seller.receive_message(b_out)
        s_out = seller.chat()
        s_act, s_price = extract_action_and_price(s_out)
        outcome["turns"] = t * 2

        log_print(f"\033[1mSELLER\033[0m\n\n{s_out}\n\n")
        log_print(
            f"\033[1mACTION\033[0m: {s_act}, \033[1mPRICE\033[0m: ${s_price:.2f}"
            if s_price
            else f"\033[1mACTION\033[0m: {s_act}"
        )

        # Check seller action validity
        if s_act == "QUIT":
            outcome["quit"] = True
            break
        if s_act in {"BUY", "SELL", "DEAL"} and s_price is None:
            outcome["invalid"] = True
            break

        # Process seller action
        if s_act == "SELL":
            last_offer = ("SELL", s_price)
        elif s_act == "DEAL":
            if last_offer and last_offer[0] == "BUY":
                outcome["deal"] = True
                outcome["price"] = s_price
                break
            else:
                outcome["invalid"] = True
                break
        elif s_act == "REJECT":
            last_offer = None
        elif s_act == "BUY":
            outcome["invalid"] = True
            break
        elif s_act == "INVALID":
            outcome["invalid"] = True
            break

        # Send seller's message to buyer
        buyer.receive_message(s_out)

    # Feasibility check
    if outcome["deal"]:
        D = outcome["price"]
        if D is None or not (C <= D <= B):
            outcome["invalid"] = True
            outcome["deal"] = False
            outcome["price"] = None

    # Log final outcome
    log_print("\n" + "=" * 80)
    if outcome["deal"]:
        log_print(
            f"✓ DEAL REACHED at ${outcome['price']:.2f} (turns: {outcome['turns']})"
        )
    elif outcome["quit"]:
        log_print(f"✗ Negotiation QUIT (turns: {outcome['turns']})")
    elif outcome["invalid"]:
        log_print(f"✗ INVALID action/outcome (turns: {outcome['turns']})")
    else:
        log_print(f"✗ No deal reached (turns: {outcome['turns']})")
    log_print("=" * 80 + "\n")

    return outcome


def run_session(testing_model: str, product_limit: int, dataset_dir: str) -> Dict:
    """Run a full evaluation session comparing a model against GPT-4o.

    Executes two configurations over up to ``product_limit`` products:

    - **Config A**: ``testing_model`` plays Buyer, GPT-4o plays Seller.
    - **Config B**: GPT-4o plays Buyer, ``testing_model`` plays Seller.

    For each product the buyer budget is set to ``B = 0.8 * highest_price``
    and the seller cost floor to ``C = lowest_price``. Results are aggregated
    across ALL products and split by MI (mutually integrative) / CI
    (competitive) scenarios.

    Args:
        testing_model: HuggingFace model identifier or OpenAI model name
            to evaluate.
        product_limit: Maximum number of products to negotiate over per
            configuration.
        dataset_dir: Path to a directory of product JSON files (e.g.
            ``"data/amazon_history_price"``).

    Returns:
        A dict with keys ``config_A`` and ``config_B``, each containing
        per-split aggregated metrics from :func:`.utils.finalize_aggregates`.
    """
    default_model = "gpt-4o"
    f = 0.8
    max_turns = 12

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_file = runs_dir / f"session_{timestamp}.log"

    buyerA_client = _get_client(testing_model)
    sellerA_client = _get_client(default_model)
    buyerB_client = _get_client(default_model)
    sellerB_client = _get_client(testing_model)

    agg_A, agg_B = {}, {}

    # ------------------ Config A ------------------
    print(f"=== Config A: Buyer={testing_model}, Seller={default_model} ===")
    for i in range(product_limit):
        try:
            item = load_product(dataset_dir, product_index=i)
        except IndexError:
            logger.info("Reached end of available products at index %d. Stopping.", i)
            break

        B = f * float(item["highest_price"])
        C = float(item["lowest_price"])

        inv_block = inventory_list(item)
        shop_block = shopping_list(item, B)

        buyer = BuyerAgent(
            client=buyerA_client,
            model_name=testing_model,
            inv_block=inv_block,
            shop_block=shop_block,
            B=B,
            code=item["codename"],
            max_turns=max_turns,
        )

        seller = SellerAgent(
            client=sellerA_client,
            model_name=default_model,
            inv_block=inv_block,
            C=C,
            code=item["codename"],
            max_turns=max_turns,
        )

        outcome = run_dialog(
            buyer=buyer,
            seller=seller,
            item=item,
            B=B,
            C=C,
            max_turns=max_turns,
            log_file=session_log_file,
        )
        m = compute_metrics(outcome, B, C)
        accumulate(agg_A, m)

        if (i + 1) % 50 == 0:
            logger.info("Config A progress: %d/%d", i + 1, product_limit)

    res_A = finalize_aggregates(agg_A)
    print("\n" + "=" * 80)
    print(f"=== Results: Config A ({testing_model} as Buyer) ===")
    print(json.dumps(res_A, indent=2))
    print("=" * 80 + "\n")

    # ------------------ Config B ------------------
    logger.info("=== Config B: Buyer=%s, Seller=%s ===", default_model, testing_model)
    for i in range(product_limit):
        try:
            item = load_product(dataset_dir, product_index=i)
        except IndexError:
            logger.info("Reached end of available products at index %d. Stopping.", i)
            break

        B = f * float(item["highest_price"])
        C = float(item["lowest_price"])

        inv_block = inventory_list(item)
        shop_block = shopping_list(item, B)

        buyer = BuyerAgent(
            client=buyerB_client,
            model_name=default_model,
            inv_block=inv_block,
            shop_block=shop_block,
            B=B,
            code=item["codename"],
            max_turns=max_turns,
        )

        seller = SellerAgent(
            client=sellerB_client,
            model_name=testing_model,
            inv_block=inv_block,
            C=C,
            code=item["codename"],
            max_turns=max_turns,
        )

        outcome = run_dialog(
            buyer=buyer,
            seller=seller,
            item=item,
            B=B,
            C=C,
            max_turns=max_turns,
            log_file=session_log_file,
        )
        m = compute_metrics(outcome, B, C)
        accumulate(agg_B, m)

        if (i + 1) % 50 == 0:
            logger.info("Config B progress: %d/%d", i + 1, product_limit)

    res_B = finalize_aggregates(agg_B)
    print("\n" + "=" * 80)
    print(f"=== Results: Config B ({testing_model} as Seller) ===")
    print(json.dumps(res_B, indent=2))
    print("=" * 80 + "\n")

    print("\n" + "=" * 100)
    print("=== FINAL SUMMARY ===")
    print("=" * 100)
    print(f"\nConfig A ({testing_model} as Buyer, {default_model} as Seller):")
    print(json.dumps(res_A, indent=2))
    print(f"\nConfig B ({default_model} as Buyer, {testing_model} as Seller):")
    print(json.dumps(res_B, indent=2))
    print("=" * 100 + "\n")

    return {"config_A": res_A, "config_B": res_B}


if __name__ == "__main__":
    run_session(
        testing_model="mistralai/Mistral-7B-Instruct-v0.2",
        product_limit=12,
        dataset_dir="data/amazon_history_price",
    )
