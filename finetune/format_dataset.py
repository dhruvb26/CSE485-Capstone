import argparse
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class PromptTemplates:
    def __init__(self, platform_name: str):
        self.platform_name = platform_name

    def create_system_prompt(
        self,
        role: str,
        item_title: str,
        item_price: float,
        item_description: str,
        category: str,
    ) -> str:
        base_context = f"""
        You are participating in a negotiation on {self.platform_name} for the following item:

        **Item:** {item_title}
        **Category:** {category.title()}
        **Listed Price:** ${item_price}
        **Description:** {item_description}
        """

        if role == "buyer":
            return (
                base_context
                + """**Your Role:** You are a buyer interested in purchasing this item. Your goals are to:

                    - Get the best possible price
                    - Ask relevant questions about the item's condition, features, etc.
                    - Negotiate respectfully but firmly
                    - Be willing to walk away if the price is too high
                    - Consider factors like pickup/delivery, payment method, etc.

                    Respond naturally and conversationally while trying to achieve a good deal."""
            )
        else:
            return (
                base_context
                + """**Your Role:** You are the seller of this item. Your goals are to:

                    - Get a fair price for your item (ideally close to the listed price)
                    - Answer questions about the item honestly
                    - Negotiate while maintaining the value of your item
                    - Be willing to make reasonable concessions to close the deal
                    - Consider factors like pickup/delivery, urgency to sell, etc.

                    Respond naturally and conversationally while trying to get a good price for your item."""
            )

    def create_alpaca_instruction(
        self,
        perspective: str,
        item_title: str,
        item_price: float,
        category: str,
        item_description: str,
    ) -> str:
        return f"""You are a {perspective} in a price negotiation on {self.platform_name}
        for: {item_title} (${item_price}, {category}).

        Item description: {item_description}

        Respond to the following conversation as the {perspective} would.
        Always reason in three parts:

        Thought: your inner reasoning about price and strategy.
        Talk: your natural response.
        Action: choose one of [BUY], [SELL], [DEAL], [REJECT], or [QUIT].
        """

    def create_sharegpt_context(
        self, item_title: str, item_price: float, category: str, item_description: str
    ) -> str:
        return f"""This is a negotiation conversation on {self.platform_name} for: {item_title}

                    Price: ${item_price} | Category: {category}
                    Description: {item_description}

                    The conversation shows how buyer and seller negotiate."""


class MessageData(BaseModel):
    role: str = Field(..., description="Role of the speaker (buyer/seller)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default="", description="Message timestamp")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ["buyer", "seller"]:
            raise ValueError('Role must be either "buyer" or "seller"')
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class ConversationData(BaseModel):
    uuid: str = Field(..., description="Unique conversation identifier")
    category: str = Field(..., description="Item category")
    item_title: str = Field(..., description="Item title")
    item_price: float = Field(
        ..., ge=0, description="Item price (must be non-negative)"
    )
    item_description: str = Field(..., description="Item description")
    messages: List[MessageData] = Field(
        ..., min_length=1, description="Conversation messages"
    )
    outcome: Dict[str, Any] = Field(
        default_factory=dict, description="Negotiation outcome"
    )
    successful: bool = Field(..., description="Whether negotiation was successful")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Conversation must have at least one message")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasetProcessor(ABC):
    """Abstract base class for dataset-specific processing logic."""

    @abstractmethod
    def extract_messages(
        self, events_or_data: List[Dict[str, Any]]
    ) -> List[MessageData]:
        """Extract messages from dataset-specific format."""
        pass

    @abstractmethod
    def get_item_info(self, record: Dict[str, Any]) -> Tuple[str, float, str]:
        """Extract item information from dataset-specific format."""
        pass

    @abstractmethod
    def is_successful(self, record: Dict[str, Any]) -> bool:
        """Determine if negotiation was successful from dataset-specific format."""
        pass

    @abstractmethod
    def get_conversation_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract conversation metadata (uuid, category, outcome) from dataset-specific format."""
        pass


class CraigslistBargainsProcessor(DatasetProcessor):
    """Processor for Craigslist Bargains dataset format."""

    def extract_messages(self, events: List[Dict[str, Any]]) -> List[MessageData]:
        messages = []

        for event in events:
            if event.get("action") == "message" and event.get("data"):
                agent_id = event.get("agent", 0)
                role = "buyer" if agent_id == 0 else "seller"
                content = event["data"].strip()

                if content:
                    try:
                        message = MessageData(
                            role=role,
                            content=content,
                            timestamp=str(event.get("time", "")),
                        )
                        messages.append(message)
                    except Exception as e:
                        logger.info(f"Warning: Invalid message data: {e}")
                        continue

        return messages

    def get_item_info(self, record: Dict[str, Any]) -> Tuple[str, float, str]:
        """Extract item information from Craigslist Bargains scenario format."""
        try:
            scenario = record.get("scenario", {})
            kbs = scenario.get("kbs", [{}])
            if kbs:
                item = kbs[0].get("item", {})
                title = item.get("Title", "Unknown Item")
                price = float(item.get("Price", 0))

                description = item.get("Description", [])
                if isinstance(description, list):
                    description = " ".join(description)
                elif not isinstance(description, str):
                    description = str(description)

                return title, price, description
        except Exception as e:
            logger.info(f"Warning: Error extracting item info: {e}")

        return "Unknown Item", 0.0, "No description available"

    def is_successful(self, record: Dict[str, Any]) -> bool:
        """Check if negotiation was successful in Craigslist Bargains format."""
        outcome = record.get("outcome", {})
        return outcome.get("reward", 0) == 1

    def get_conversation_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from Craigslist Bargains format."""
        scenario = record.get("scenario", {})
        return {
            "uuid": record.get("uuid", ""),
            "category": scenario.get("category", "unknown"),
            "outcome": record.get("outcome", {}),
            "events": record.get("events", []),
        }


class ProcessorFactory:
    """Factory class to create appropriate dataset processors."""

    _processors = {
        "craigslist_bargains": CraigslistBargainsProcessor,
    }

    @classmethod
    def create_processor(cls, dataset_name: str) -> DatasetProcessor:
        """Create a processor instance for the given dataset."""
        if dataset_name not in cls._processors:
            available = ", ".join(cls._processors.keys())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available processors: {available}"
            )

        return cls._processors[dataset_name]()

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of supported dataset names."""
        return list(cls._processors.keys())


class DatasetFormatter:
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        platform_name: str,
        processor: DatasetProcessor = None,
    ):
        self.data_dir = Path(data_dir) / dataset_name
        self.dataset_name = dataset_name
        self.conversations: List[ConversationData] = []
        self.prompt_templates = PromptTemplates(platform_name)

        if processor is None:
            self.processor = ProcessorFactory.create_processor(dataset_name)
        else:
            self.processor = processor

    def load_data(self, split: str = "all") -> List[Dict[str, Any]]:
        data = []

        if split == "all":
            files = ["train.json", "validation.json", "test.json"]
        else:
            files = [f"{split}.json"]

        for filename in files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        data.extend(file_data)
                        logger.info(
                            f"  Loaded {len(file_data)} conversations from {filename}"
                        )
                except Exception as e:
                    logger.info(f"Error loading {filename}: {e}")
            else:
                logger.info(f"Warning: {filepath} not found")

        return data

    def process_conversations(
        self,
        raw_data: List[Dict[str, Any]],
    ) -> None:
        """Process raw data into structured conversations using dataset-specific processor."""
        processed = 0
        filtered = 0

        for record in raw_data:
            try:
                metadata = self.processor.get_conversation_metadata(record)
                uuid = metadata.get("uuid", "")
                category = metadata.get("category", "unknown")
                outcome = metadata.get("outcome", {})
                events_or_data = metadata.get("events", [])

                title, price, description = self.processor.get_item_info(record)

                messages = self.processor.extract_messages(events_or_data)

                successful = self.processor.is_successful(record)

                try:
                    conv_data = ConversationData(
                        uuid=uuid,
                        category=category,
                        item_title=title,
                        item_price=price,
                        item_description=description,
                        messages=messages,
                        outcome=outcome,
                        successful=successful,
                    )
                    self.conversations.append(conv_data)
                    processed += 1
                except Exception as validation_error:
                    logger.info(
                        f"Validation error for conversation {uuid}: {validation_error}"
                    )
                    filtered += 1
                    continue

            except Exception as e:
                logger.info(
                    f"Error processing conversation {record.get('uuid', 'unknown')}: {e}"
                )
                filtered += 1

        if self.conversations:
            success_rate = (
                sum(1 for c in self.conversations if c.successful)
                / len(self.conversations)
                * 100
            )
            logger.info(
                f"Processed: {processed}, Filtered: {filtered}, Success: {success_rate:.1f}%"
            )
        else:
            logger.info(
                f"Processed: {processed}, Filtered: {filtered}, Success: N/A (no conversations)"
            )

    def format_chatml(
        self, conversation: ConversationData, perspective: str = "buyer"
    ) -> Dict[str, Any]:
        messages = []

        system_prompt = self.prompt_templates.create_system_prompt(
            perspective,
            conversation.item_title,
            conversation.item_price,
            conversation.item_description,
            conversation.category,
        )
        messages.append({"role": "system", "content": system_prompt})

        for msg in conversation.messages:
            if msg.role == perspective:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                messages.append({"role": "user", "content": msg.content})

        return {
            "messages": messages,
            "metadata": {
                "uuid": conversation.uuid,
                "category": conversation.category,
                "item_title": conversation.item_title,
                "item_price": conversation.item_price,
                "successful": conversation.successful,
                "perspective": perspective,
                "final_offer": conversation.outcome.get("offer", {}).get("price")
                if conversation.successful
                else None,
            },
        }

    def format_alpaca(
        self, conversation: ConversationData, perspective: str = "buyer"
    ) -> Dict[str, Any]:
        instruction = self.prompt_templates.create_alpaca_instruction(
            perspective,
            conversation.item_title,
            conversation.item_price,
            conversation.category,
            conversation.item_description,
        )

        input_parts = []
        output_parts = []

        for msg in conversation.messages:
            if msg.role == perspective:
                output_parts.append(
                    f"Thought: Reason about your next move based on context.\n"
                    f"Talk: {msg.content}\n"
                    f"Action: Choose one valid action: [BUY], [SELL], [DEAL], [REJECT], or [QUIT]."
                )
            else:
                other_role = "seller" if perspective == "buyer" else "buyer"
                input_parts.append(f"{other_role.title()}: {msg.content}")

        return {
            "instruction": instruction,
            "input": "\n".join(input_parts),
            "output": "\n".join(output_parts),
            "metadata": {
                "uuid": conversation.uuid,
                "category": conversation.category,
                "successful": conversation.successful,
                "perspective": perspective,
                "price": conversation.item_price
            }
        }


    def format_sharegpt(self, conversation: ConversationData) -> Dict[str, Any]:
        messages = []

        system_prompt = self.prompt_templates.create_sharegpt_context(
            conversation.item_title,
            conversation.item_price,
            conversation.category,
            conversation.item_description,
        )

        messages.append({"from": "system", "value": system_prompt})

        for msg in conversation.messages:
            role_map = {"buyer": "human", "seller": "gpt"}
            messages.append(
                {
                    "from": role_map.get(msg.role, msg.role),
                    "value": f"[{msg.role.title()}] {msg.content}",
                }
            )

        return {
            "conversations": messages,
            "metadata": {
                "uuid": conversation.uuid,
                "category": conversation.category,
                "successful": conversation.successful,
                "outcome": conversation.outcome,
            },
        }

    def export_data(
    self,
    output_dir: str = "formatted_data",
    formats: List[str] = None,
    max_samples: int = None,
    ) -> None:
        if formats is None:
            formats = ["chatml", "alpaca", "sharegpt"]

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Only use successful conversations
        conversations_to_use = [c for c in self.conversations if c.successful]

        if max_samples is not None:
            conversations_to_use = conversations_to_use[:max_samples]
            logger.info(
                f"Limiting to {len(conversations_to_use)} conversations (max_samples={max_samples})"
            )

        logger.info(f"Exporting data to {output_path}...")

        def sanitize_text(value: Any) -> Any:
            """Remove or escape problematic control characters and normalize whitespace."""
            if isinstance(value, str):
                # Replace backslashes, carriage returns, and raw newlines
                cleaned = value.replace("\\", " ").replace("\r", "").replace("\n", " ")
                
                # Normalize multiple spaces into one and trim leading/trailing spaces
                cleaned = " ".join(cleaned.split())
                return cleaned.strip()
            return value

        for format_name in formats:
            formatted_data = []

            if format_name == "chatml":
                for conv in conversations_to_use:
                    formatted_data.append(self.format_chatml(conv, "buyer"))
                    formatted_data.append(self.format_chatml(conv, "seller"))

            elif format_name == "alpaca":
                for conv in conversations_to_use:
                    formatted_data.append(self.format_alpaca(conv, "buyer"))
                    formatted_data.append(self.format_alpaca(conv, "seller"))

            elif format_name == "sharegpt":
                for conv in conversations_to_use:
                    formatted_data.append(self.format_sharegpt(conv))

            output_file = output_path / f"{self.dataset_name}_{format_name}.jsonl"

            with open(output_file, "w", encoding="utf-8") as f:
                for item in formatted_data:
                    cleaned_item = {
                        k: sanitize_text(v)
                        if isinstance(v, str)
                        else (
                            {kk: sanitize_text(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
                        )
                        for k, v in item.items()
                    }
                    json.dump(cleaned_item, f, ensure_ascii=False)
                    f.write("\n")

            logger.info(f"Saved {len(formatted_data)} valid examples to {output_file}")



def main():
    available_datasets = ProcessorFactory.get_available_datasets()

    parser = argparse.ArgumentParser(
        description="Format negotiation dataset for instruction tuning. Supports multiple dataset formats through pluggable processors."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--dataset-name",
        default="craigslist_bargains",
        choices=available_datasets,
        help=f"Name of the dataset (must match a folder name inside the --data-dir). Available: {', '.join(available_datasets)}",
    )
    parser.add_argument(
        "--platform-name",
        default="online marketplace",
        help="Name of the platform/marketplace for prompts",
    )
    parser.add_argument(
        "--output-dir",
        default="formatted_data",
        help="Output directory for formatted files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
        help="Which data split to process",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["chatml", "alpaca", "sharegpt"],
        default=["alpaca"],
        help="Output formats to generate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of conversations to include in the final dataset",
    )

    args = parser.parse_args()

    try:
        processor = ProcessorFactory.create_processor(args.dataset_name)
        logger.info(f"Using processor: {processor.__class__.__name__}")
    except ValueError as e:
        logger.error(f"Error: {e}")
        return

    formatter = DatasetFormatter(
        args.data_dir, args.dataset_name, args.platform_name, processor
    )

    raw_data = formatter.load_data(args.split)
    if not raw_data:
        logger.error("No data loaded. Please check the data directory path.")
        return

    formatter.process_conversations(raw_data)

    formatter.export_data(args.output_dir, args.formats, args.max_samples)


if __name__ == "__main__":
    main()
