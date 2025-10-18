import json
import logging
import sys
from pathlib import Path
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import check_gpu, start_vllm_server

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def _start_vllm_and_client(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    port: int = 8000,
    host: str = "0.0.0.0",
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
) -> OpenAI:
    """Start vLLM server and return OpenAI client."""
    try:
        cache_dir: str = "/scratch/dbansa11/hf_models"
        
        # Start the vLLM server
        start_vllm_server(
            model=model_name,
            port=port,
            host=host,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            cache_dir=cache_dir,
        )
        
        client = OpenAI(
            api_key="", 
            base_url=f"http://localhost:{port}/v1",
        )
        
        return client
    except Exception as e:
        logger.error(f"Error starting vLLM server: {e}")
        exit(1)


def run_model(
    client: OpenAI,
    model_name: str,
    messages: List[ChatCompletionMessageParam],
    stream: bool = False,
) -> str:
    """Run the model via vLLM OpenAI API with optional streaming output.

    Args:
        client: OpenAI client connected to vLLM server
        model_name: Name of the model being used
        messages: List of conversation messages (system, user, assistant)
        stream: If True, prints tokens as generated; if False, returns complete text

    Returns:
        Complete generated text as string
    """
    try:
        if stream:
            return _generate_streaming(client, model_name, messages)
        else:
            # Non-streaming mode: return complete text
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""

    except Exception as e:
        logger.error(f"Error running model: {e}")
        return f"Error: {e}"


def _generate_streaming(
    client: OpenAI,
    model_name: str,
    messages: List[ChatCompletionMessageParam],
) -> str:
    """Helper function for streaming generation with real-time printing."""
    try:
        full_response = ""
        
        # Create streaming request
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )
        
        # Print tokens as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n")
        return full_response

    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        error_msg = f"Error: {e}"
        print(error_msg, end="", flush=True)
        return error_msg


def main() -> None:
    """Main entrypoint: interactive conversation with the model."""
    check_gpu()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    client = _start_vllm_and_client(model_name=model_name)

    with open("prompt.json", "r") as f:
        prompt_data = json.load(f)

    messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": prompt_data["system"]}]  # type: ignore[list-item]
    logger.info("Type your messages and press Enter to chat. Ctrl+C to exit.")

    try:
        while True:
            try:
                user_input = input("\n[User] - ").strip()
            except EOFError:
                logger.info("Goodbye!")
                break

            if not user_input:
                logger.info("Please enter a message.")
                continue

            messages.append({"role": "user", "content": user_input})

            print("[Assistant] - ", end="", flush=True)

            response = run_model(
                client,
                model_name,
                messages,
                stream=True,
            )

            messages.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        logger.info("Goodbye!")
        exit(0)


if __name__ == "__main__":
    main()
