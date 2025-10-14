import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run inference on Cambridge MCQ test data with a Hugging Face causal model.")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-32B",
                        help='The model to use for inference, Qwen/Qwen3-32B, Qwen/QwQ-32B, microsoft/Phi-4-reasoning')
    parser.add_argument('--cache_dir', type=str, default='/nfshomes/minglii/scratch/cache/hub/',
                        help='Directory to cache the model (default: HF default cache location)')
    parser.add_argument('--input_file', type=str, default='EduAgent_related/Cambridge_mcq_test_pub_Qformat.json',
                        help='Path to the input JSON file (Cambridge MCQ test data)')
    parser.add_argument('--output_file', type=str, default='EduAgent_related/model_results/Cambridge_mcq_results.jsonl',
                        help='Path to save the updated data as JSONL')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run the model on (default: auto-detect)')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate for each prompt (default: 2048)')
    parser.add_argument('--start_ratio', type=float, default=0.0,
                        help='Start ratio in [0,1). Defaults to 0.0')
    parser.add_argument('--end_ratio', type=float, default=1.0,
                        help='End ratio in (0,1]. Defaults to 1.0 (process to end).')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='System prompt to use for the conversation (optional)')
    parser.add_argument('--prefix_prompt', type=str, default=None,
                        help='Prefix prompt to prepend to the user prompt (optional)')
    parser.add_argument('--reasoning', action='store_true',
                        help='Enable reasoning mode - retry inference until response contains </think> tag')

    args = parser.parse_args()

    # Determine device (GPU vs CPU)
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # Function to run inference on a single prompt
    def run_inference(prompt: str) -> str:
        """
        Generate a response for the given prompt using the loaded model.
        In reasoning mode, retry until response contains </think> tag.
        """
        max_retries = 10  # Maximum number of retries for reasoning mode
        retry_count = 0
        
        while retry_count < max_retries:
            # Build the user prompt with prefix if provided
            user_content = prompt
            if args.prefix_prompt:
                user_content = args.prefix_prompt + prompt
            
            # Build messages array
            messages = []
            
            # Add system prompt if provided
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": user_content})
            
            if 'Qwen3' in args.model_name:
                if args.reasoning:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                else:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                )

            # Extract only the newly generated tokens (beyond the prompt)
            new_tokens = generated_ids[0][len(inputs["input_ids"][0]):]

            # Decode
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Check if reasoning mode is enabled and response contains </think> tag
            if args.reasoning:
                if '</think>' in response_text:
                    print(f"Reasoning mode: Successfully generated response with </think> tag (attempt {retry_count + 1})")
                    return response_text
                else:
                    retry_count += 1
                    print(f"Reasoning mode: Response missing </think> tag, retrying... (attempt {retry_count}/{max_retries})")
                    continue
            else:
                # Non-reasoning mode, return immediately
                return response_text
        
        # If we've exhausted all retries in reasoning mode
        if args.reasoning and retry_count >= max_retries:
            print(f"Warning: Reasoning mode failed to generate response with </think> tag after {max_retries} attempts. Returning last response.")
        
        return response_text

    # Ensure output has the correct .jsonl extension
    if not args.output_file.endswith('.jsonl'):
        args.output_file += '.jsonl'

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # -----------------------------
    # Determine items already processed
    # -----------------------------
    processed_items = set()
    response_key = f"model_response"
    if os.path.exists(args.output_file):
        print(f"Reading already processed items from {args.output_file} to avoid re‑processing …")
        with open(args.output_file, 'r', encoding='utf-8') as prev_f:
            for line in prev_f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines
                if isinstance(obj, dict) and 'processed_text' in obj and response_key in obj:
                    # Use processed_text as unique identifier for each item
                    processed_items.add(obj['processed_text'])
        print(f"Found {len(processed_items)} previously processed items.")

    # Read the input JSON (list of items)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input JSON must contain a list of items.")

    print(f"Loaded {len(data)} items from {args.input_file}")

    # Determine slicing purely based on ratios
    if not (0.0 <= args.start_ratio < args.end_ratio <= 1.0):
        raise ValueError("start_ratio must be >=0 and < end_ratio <=1.")

    start_idx = int(len(data) * args.start_ratio)
    end_idx = int(len(data) * args.end_ratio)

    # Clamp indices (in case end_idx equals len(data))
    end_idx = min(end_idx, len(data))

    data = data[start_idx:end_idx]
    print(f"Processing items ratio range: [{args.start_ratio}, {args.end_ratio}) => index [{start_idx}, {end_idx})  -> {len(data)} items")

    # Decide write mode: append if file exists, else write
    out_mode = 'a' if os.path.exists(args.output_file) else 'w'

    # Open the output .jsonl file
    with open(args.output_file, out_mode, encoding='utf-8') as out_f:
        if out_mode == 'a':
            out_f.seek(0, os.SEEK_END)  # ensure we are at end for appending

        for idx, item in enumerate(tqdm(data, desc="Processing items")):
            # Validate item structure
            if not isinstance(item, dict):
                print(f"Warning: Not a dictionary item, skipping: {item}")
                continue
            if 'processed_text' not in item:
                print(f"Warning: Missing 'processed_text' field in item: {item}")
                continue

            prompt_text = item['processed_text']
            if not isinstance(prompt_text, str):
                print(f"Warning: processed_text field is not string, skipping: {prompt_text}")
                continue

            # Skip if we've already processed this item
            if prompt_text in processed_items:
                # Item is already in the output file with a response – skip re-processing
                continue

            # Run inference
            try:
                response_text = run_inference(prompt_text)
            except RuntimeError as e:
                # Handle CUDA Out Of Memory (OOM) or other runtime errors gracefully
                err_msg = str(e)
                if 'out of memory' in err_msg.lower():
                    print(f"[OOM] Skipping item index {idx}: CUDA out-of-memory. Clearing cache and continuing …")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue  # move on to next item
                else:
                    # For other runtime errors, re-raise to halt execution
                    raise
            except Exception as e:
                # Catch-all for any other unexpected errors, log and continue
                print(f"[ERROR] Skipping item index {idx}: {e}")
                continue
            
            # Add model response to the item
            item[response_key] = response_text

            # Write updated item to .jsonl
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()

            # Mark as processed to avoid duplicates within the same run
            processed_items.add(prompt_text)

    print(f"\nProcessing complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
