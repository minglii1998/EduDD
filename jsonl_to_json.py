#!/usr/bin/env python3
"""
Convert JSONL file to JSON with indentation
"""

import json
import sys
import os

def jsonl_to_json(input_file, output_file=None, indent=2):
    """
    Convert JSONL file to JSON with indentation
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSON file (optional)
        indent (int): Number of spaces for indentation (default: 2)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.json"
    
    try:
        # Read JSONL file and convert to list of JSON objects
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        
        # Write to JSON file with indentation
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        print(f"Successfully converted {len(data)} records from '{input_file}' to '{output_file}'")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) < 2:
        print("Usage: python jsonl_to_json.py <input_file> [output_file] [indent]")
        print("Example: python jsonl_to_json.py data.jsonl data.json 4")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    indent = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    success = jsonl_to_json(input_file, output_file, indent)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
