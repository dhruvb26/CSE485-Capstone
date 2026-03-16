import glob
import json
import os
import re
from typing import Dict, Optional, Tuple


def load_product(dataset_dir: str, product_index: int = 0) -> Dict:
    """Load a single product by its global index across all JSON files in a directory.

    Iterates through sorted JSON files in ``dataset_dir``, treating each file as
    a list of products. The ``product_index`` counts across file boundaries, so
    index 0 is the first item in the first file, and if file A has 10 items,
    index 10 is the first item in file B.

    Args:
        dataset_dir: Path to a directory containing one or more JSON files,
            each holding a list of product dicts with keys like ``title``,
            ``description``, ``highest_price``, ``lowest_price``, ``category``.
        product_index: Zero-based global index of the product to retrieve.

    Returns:
        A dict with keys ``title``, ``description``, ``highest_price`` (float),
        ``lowest_price`` (float), ``category``, and ``codename``
        (e.g. ``"electronics_3"``).

    Raises:
        FileNotFoundError: If no JSON files exist in ``dataset_dir``.
        IndexError: If ``product_index`` exceeds the total number of products.
    """
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {dataset_dir}")

    remaining = product_index
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            continue

        if remaining < len(data):
            item = data[remaining]
            filename = os.path.splitext(os.path.basename(file_path))[0]
            codename = f"{filename}_{remaining}"

            def parse_price(p):
                if isinstance(p, (int, float)):
                    return float(p)
                if isinstance(p, str):
                    return float(p.replace("$", "").replace(",", ""))
                return 0.0

            highest = parse_price(item.get("highest_price"))
            lowest = parse_price(item.get("lowest_price"))

            return {
                "title": item.get("title"),
                "description": item.get("description"),
                "highest_price": highest,
                "lowest_price": lowest,
                "category": item.get("category"),
                "codename": codename,
            }
        else:
            remaining -= len(data)

    raise IndexError(
        f"Product index {product_index} exceeds total products across all JSON files."
    )


def inventory_list(item: dict) -> str:
    """Format a product dict into the seller's inventory listing text block.

    This text is injected into both the buyer and seller system prompts so each
    agent can see the product details and listing price.

    Args:
        item: Product dict as returned by :func:`load_product`.

    Returns:
        A multi-line string describing the product, its listing price, and
        available quantity.
    """
    return (
        "Inventory List:\n"
        f"Product (codename: {item['codename']})\n"
        f'Title: "{item["title"]}"\n'
        f'Description: "{item["description"]}"\n'
        f"Listing Price: ${item['highest_price']:.2f} per item\n"
        "Available Quantity: 1\n"
    )


def shopping_list(item: dict, budget: float) -> str:
    """Format a product dict into the buyer's shopping list text block.

    This text is injected into the buyer's prompt so it knows which product
    to negotiate for and its maximum budget.

    Args:
        item: Product dict as returned by :func:`load_product`.
        budget: The buyer's maximum willingness-to-pay for this product.

    Returns:
        A multi-line string with the product codename, quantity, and budget.
    """
    return (
        "Shopping List\n"
        f"codename: {item['codename']}\n"
        f"quantity: 1\n"
        f"budget: ${budget:.2f}"
    )


def extract_action_and_price(text: str) -> Tuple[str, Optional[float]]:
    """Parse a negotiation action and optional price from an agent's response.

    Scans the response text bottom-up, line by line, looking for the first line
    matching the ``[ACTION] $price`` pattern (e.g. ``[BUY] $42.50``). Only the
    price on the same line as the action tag is extracted — prices mentioned
    elsewhere in the response are ignored.

    Args:
        text: The full text output from a buyer or seller agent.

    Returns:
        A ``(action, price)`` tuple where ``action`` is one of ``"BUY"``,
        ``"SELL"``, ``"DEAL"``, ``"REJECT"``, ``"QUIT"``, or ``"INVALID"``,
        and ``price`` is a float (for BUY/SELL/DEAL) or None.
    """
    if not text:
        return "INVALID", None

    action, price = "INVALID", None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.compile(
            r"^(?:\s*Action:\s*)?\[(BUY|SELL|DEAL|REJECT|QUIT)\]\s*(?:\$\s*([0-9,]+(?:\.[0-9]{1,2})?))?",
            re.I,
        ).match(ln)
        if not m:
            continue
        action = m.group(1).upper()
        if action in {"BUY", "SELL", "DEAL"}:
            p = m.group(2)
            price = float(p.replace(",", "")) if p is not None else None
        else:
            price = None
        return action, price
    return action, price


def compute_metrics(outcome: Dict, B: float, C: float) -> Dict:
    """Compute per-dialog performance metrics from a negotiation outcome.

    Calculates buyer profit (Pb), seller profit (Ps), and their normalized
    variants (NPb, NPs) relative to the bargaining zone ``|B - C|``.  Also
    classifies the scenario as MI (mutually integrative, ``C <= B``) or
    CI (competitive, ``C > B``).

    Args:
        outcome: Dict returned by ``run_dialog`` with keys ``deal``, ``invalid``,
            and ``price``.
        B: Buyer's maximum budget.
        C: Seller's minimum acceptable cost.

    Returns:
        A dict with keys ``valid``, ``deal``, ``scenario``, ``Pb``, ``Ps``,
        ``NPb``, ``NPs``.
    """
    res = {
        "valid": not outcome.get("invalid", False),
        "deal": outcome.get("deal", False),
        "scenario": "MI" if C <= B else "CI",
    }
    if res["deal"]:
        D = outcome["price"]
        denom = abs(B - C) if abs(B - C) > 1e-9 else 1.0
        res["Pb"] = B - D
        res["Ps"] = D - C
        res["NPb"] = (B - D) / denom
        res["NPs"] = (D - C) / denom
    else:
        res["Pb"] = res["Ps"] = res["NPb"] = res["NPs"] = 0.0
    return res


def accumulate(agg: Dict, one: Dict) -> None:
    """Add a single dialog's metrics into running aggregates by scenario split.

    Maintains separate running totals for ``"ALL"`` and for the specific
    scenario (``"MI"`` or ``"CI"``). Each split tracks counts (n, valid, deal)
    and cumulative profit sums (SPb, SPs, SNPb, SNPs).

    Args:
        agg: Mutable aggregation dict, modified in-place. Starts as ``{}``
            and is populated with split keys on first call.
        one: Single-dialog metrics dict as returned by :func:`compute_metrics`.
    """
    for split in ["ALL", one["scenario"]]:
        a = agg.setdefault(
            split,
            {
                "n": 0,
                "valid": 0,
                "deal": 0,
                "SPb": 0.0,
                "SPs": 0.0,
                "SNPb": 0.0,
                "SNPs": 0.0,
            },
        )
        a["n"] += 1
        a["valid"] += int(one["valid"])
        a["deal"] += int(one["deal"])
        a["SPb"] += one["Pb"]
        a["SPs"] += one["Ps"]
        a["SNPb"] += one["NPb"]
        a["SNPs"] += one["NPs"]


def finalize_aggregates(agg: Dict) -> Dict:
    """Convert running totals from :func:`accumulate` into final summary stats.

    Args:
        agg: Aggregation dict populated by :func:`accumulate`, keyed by
            scenario split (``"ALL"``, ``"MI"``, ``"CI"``).

    Returns:
        A dict keyed by split, each containing ``n``, ``valid_rate``,
        ``deal_rate``, ``SP_b``, ``SP_s``, ``SNP_b``, ``SNP_s``.
    """
    out = {}
    for split, a in agg.items():
        n = max(1, a["n"])
        out[split] = {
            "n": a["n"],
            "valid_rate": a["valid"] / n,
            "deal_rate": a["deal"] / n,
            "SP_b": a["SPb"],
            "SP_s": a["SPs"],
            "SNP_b": a["SNPb"],
            "SNP_s": a["SNPs"],
        }
    return out
