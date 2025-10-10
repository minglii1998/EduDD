#!/usr/bin/env python3
"""
EduRL Model Inference and Evaluation Script

This script reads Cambridge MCQ test data, performs inference using HuggingFace models
with their native chat templates, extracts difficulty predictions from \boxed{} format, 
and evaluates against ground truth.
"""

import os
import json
import re
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

# Disable DeepSpeed CUDA checks for CPU inference
os.environ["DS_BUILD_OPS"] = "0"
os.environ["DS_BUILD_FUSED_ADAM"] = "0"
os.environ["DS_BUILD_FUSED_LAMB"] = "0"
os.environ["DS_BUILD_TRANSFORMER"] = "0"
os.environ["DS_BUILD_TRANSFORMER_INFERENCE"] = "0"
os.environ["DS_BUILD_STOCHASTIC_TRANSFORMER"] = "0"
os.environ["DS_BUILD_UTILS"] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_data(json_path):
    """Load and parse the Cambridge MCQ test data."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_texts = []
    difficulties = []
    
    for item in data:
        processed_texts.append(item['processed_text'])
        difficulties.append(item['difficulty'])
    
    print(f"Loaded {len(processed_texts)} samples")
    return processed_texts, difficulties


def setup_model(model_name, device='auto'):
    """Setup the HuggingFace model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Determine appropriate dtype and loading parameters based on device
    if device == 'cpu':
        torch_dtype = torch.float32
        print("Using CPU mode with float32")
        
        # Load model for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        torch_dtype = torch.float16
        print("Using GPU mode with float16")
        
        # Load model for GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if model has chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        print(f"Using model's chat template")
    else:
        print(f"Model does not have chat template, using fallback format")
    
    print(f"Model loaded successfully on device: {model.device}")
    return model, tokenizer


def extract_boxed_value(text):
    """Extract numerical value from \boxed{} format in the response.
    If no boxed value is found, extract the last number from the response."""
    # Look for \boxed{number} pattern
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        try:
            # Extract the last match (most likely the final answer)
            value_str = matches[-1].strip()
            # Try to extract number from the string
            number_match = re.search(r'-?\d+\.?\d*', value_str)
            if number_match:
                return float(number_match.group())
        except (ValueError, AttributeError):
            pass
    
    # If no boxed value found, try to extract the last number from the entire response
    try:
        # Find all numbers in the text
        number_pattern = r'-?\d+\.?\d*'
        all_numbers = re.findall(number_pattern, text)
        
        if all_numbers:
            # Return the last number found
            return float(all_numbers[-1])
    except (ValueError, AttributeError):
        pass
    
    return None


def generate_response(model, tokenizer, prompt, system_prompt, max_length=2048, temperature=0.6):
    """Generate response using the model's chat template."""
    # Prepare messages for chat template
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": f"Please analyze the difficulty of this reading comprehension question:\n\n{prompt}"
        }
    ]
    
    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback for models without chat template
        formatted_prompt = f"System: {system_prompt}\n\nUser: Please analyze the difficulty of this reading comprehension question:\n\n{prompt}\n\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_predictions(predictions, ground_truth):
    """Calculate evaluation metrics."""
    # Filter out None predictions
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truth = [ground_truth[i] for i in valid_indices]
    
    if len(valid_predictions) == 0:
        print("No valid predictions found!")
        return None
    
    print(f"Valid predictions: {len(valid_predictions)}/{len(predictions)}")
    
    # Calculate metrics
    mae = mean_absolute_error(valid_ground_truth, valid_predictions)
    mse = mean_squared_error(valid_ground_truth, valid_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(valid_ground_truth, valid_predictions)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'valid_samples': len(valid_predictions),
        'total_samples': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description='EduRL Model Inference and Evaluation')
    parser.add_argument('--data_path', type=str, 
                       default='split_pub/Cambridge_mcq_test_pub_Processed.json',
                       help='Path to the JSON data file')
    parser.add_argument('--model_name', type=str, 
                       default='EduRL_related/verl_edu_checkpoint/Cambridge_difficulty/try_Cambridge_difficulty/global_step_340/actor/huggingface',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--system_prompt', type=str, 
                       default="You are an expert to analyze the reading comprehension question difficulty levels. There are 10 difficulty levels, from 0 to 9, where 0 is the easiest and 9 is the hardest. Analyze the difficulty of the given question step by step and put your final answer (the classification digit) within \\boxed{}.")
    
    args = parser.parse_args()
    
    # Load data
    processed_texts, difficulties = load_data(args.data_path)
    
    # Limit samples if specified
    if args.max_samples:
        processed_texts = processed_texts[:args.max_samples]
        difficulties = difficulties[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for testing")
    
    # Setup model
    model, tokenizer = setup_model(args.model_name, args.device)
    
    # Perform inference
    print("Starting inference...")
    predictions = []
    
    for i, text in enumerate(tqdm(processed_texts, desc="Processing")):
        try:
            # Generate response
            response = generate_response(model, tokenizer, text, args.system_prompt)
            
            # Extract boxed value
            boxed_value = extract_boxed_value(response)
            
            if boxed_value is not None:
                # Multiply by 10 as requested
                predicted_difficulty = boxed_value * 10
                predictions.append(predicted_difficulty)
                print(f"Sample {i+1}: Predicted {predicted_difficulty:.2f}, Ground truth {difficulties[i]:.2f}")
            else:
                predictions.append(None)
                print(f"Sample {i+1}: No valid prediction found in response")
                
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            predictions.append(None)

        pass
    
    # Evaluate results
    print("\nEvaluating results...")
    results = evaluate_predictions(predictions, difficulties)
    
    if results:
        print(f"\nEvaluation Results:")
        print(f"MAE: {results['MAE']:.4f}")
        print(f"MSE: {results['MSE']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"R²: {results['R2']:.4f}")
        print(f"Valid samples: {results['valid_samples']}/{results['total_samples']}")
        
        # Save results
        output_data = {
            'model_name': args.model_name,
            'data_path': args.data_path,
            'system_prompt': args.system_prompt,
            'metrics': results,
            'predictions': predictions,
            'ground_truth': difficulties
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output_file}")
    else:
        print("No valid results to save")


if __name__ == "__main__":
    main()
