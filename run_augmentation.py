#!/usr/bin/env python3
"""
Script to apply multiple Random Code Transformations to JSONL datasets.
Mimics run_rename.py structure but selects 1 of 5 transformations per sample
in a balanced random manner (try until success).

Usage:
    python run_augmentation.py --input_dir data/raw --output_dir data/augmented --mode c
    python run_augmentation.py --input_dir data/raw --output_dir data/augmented --mode java
"""

import os
import sys
import json
import logging
import argparse
import multiprocessing
import random
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Type
from functools import partial
from tqdm import tqdm

# Add project root to sys.path to import src modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import required modules
try:
    from src.data_preprocessors.transformations.block_swap_transformations import BlockSwap
    from src.data_preprocessors.transformations.confusion_remove import ConfusionRemover
    from src.data_preprocessors.transformations.dead_code_inserter import DeadCodeInserter
    from src.data_preprocessors.transformations.for_while_transformation import ForWhileTransformer
    from src.data_preprocessors.transformations.operand_swap_transformations import OperandSwap
except ImportError as e:
    print(f"Error: Could not import transformations. Make sure you are running from the project root. Detail: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
PARSER_LIB_PATH = PROJECT_ROOT / "parser" / "languages.so"

TRANSFORMERS: List[Type] = [
    BlockSwap,
    ConfusionRemover,
    DeadCodeInserter,
    ForWhileTransformer,
    OperandSwap
]

def get_parser_lang(mode: str, file_name: str) -> str:
    """
    Determine the correct Tree-sitter language identifier based on mode and filename.
    
    Args:
        mode: 'c' or 'java'
        file_name: The source filename (e.g., "test.c", "main.cpp")
        
    Returns:
        The language string ('c', 'cpp', 'java')
    """
    if mode == "java":
        return "java"
    
    if mode == "c":
        # Check extensions for C++
        if file_name.endswith(".cpp") or file_name.endswith(".cc") or file_name.endswith(".cxx"):
            return "cpp"
        # Default to c for .c or others
        return "c"
        
    return mode


def process_line(args: Tuple[str, str, str]) -> Optional[str]:
    """
    Worker function to process a single JSON line.
    
    Args:
        args: Tuple containing (json_line_string, parser_path, mode)
        
    Returns:
        Processed JSON string or None if failed.
    """
    line, parser_path, mode = args
    if not line.strip():
        return None

    try:
        data: Dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        return None

    code_content = data.get("func")
    if not code_content:
        return line # Return original if no code found

    file_name = data.get("file", "")
    
    # Determine language
    lang = get_parser_lang(mode, file_name)

    # Strategy: Balanced Random Selection
    # 1. Shuffle the list of transformers for this random sample
    # 2. Try each one in order until one succeeds
    
    transformers_shuffled = list(TRANSFORMERS)
    random.shuffle(transformers_shuffled)
    
    transformation_success = False
    
    # Try transformations
    for TransformerClass in transformers_shuffled:
        try:
            # Instantiate per line to be safe with multiprocessing and parser state
            transformer = TransformerClass(parser_path=parser_path, language=lang)
            
            # transform_code returns: (new_code_string, meta_dict)
            new_code, meta = transformer.transform_code(code_content)
            
            # Check success flag from metadata
            success = meta.get("success", False)
            
            # Double check: sometimes success is True but code is identical (guard)
            if success and new_code.strip() != code_content.strip():
                data["func"] = new_code
                data["augmented"] = True
                data["transformation_used"] = TransformerClass.__name__
                transformation_success = True
                # User requirement: "chỉ cần 1 biến thể success là được"
                break 
                
        except Exception:
            # If one transformer fails (crashes), continue to the next one
            continue

    # If all failed, we still return the original data (can be filtered later if needed, 
    # but run_rename.py returns original too). 
    # The 'transformed_code' field will be missing if no transformation succeeded.
    
    return json.dumps(data)


def process_file(
    input_file: Path, 
    output_file: Path, 
    mode: str, 
    num_workers: int
) -> None:
    """Read input file, process lines in parallel, and write to output."""
    
    # 1. Read all lines
    logger.info(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    parser_path_str = str(PARSER_LIB_PATH)
    
    # Pack arguments for worker
    worker_args = [(line, parser_path_str, mode) for line in lines]

    # 2. Process in parallel
    logger.info(f"Processing {total_lines} samples with {num_workers} workers...")
    
    # We use a lower chunksize for augmentation because it might be slower than renaming
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_line, worker_args, chunksize=50),
            total=total_lines,
            unit="sample"
        ))
    
    # Filter Nones
    processed_lines = [r for r in results if r is not None]

    # Calculate stats
    augmented_count = sum(1 for line in processed_lines if '"augmented": true' in line)
    logger.info(f"Augmentation Success Rate: {augmented_count}/{len(processed_lines)} ({augmented_count/len(processed_lines)*100:.2f}%)")

    # 3. Write Output
    logger.info(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
            
    logger.info(f"Done. Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Random Code Augmentation Runner (5 Types)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw jsonl folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save augmented jsonl")
    parser.add_argument("--mode", type=str, choices=["c", "java"], required=True, help="Language mode: 'c' (auto-detects cpp) or 'java'")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    
    args = parser.parse_args()
    
    in_path = Path(args.input_dir)
    out_path = Path(args.output_dir)
    
    if not in_path.exists():
        logger.error(f"Input directory not found: {in_path}")
        return

    # Check parser existence
    if not PARSER_LIB_PATH.exists():
        logger.error(f"Parser library not found at: {PARSER_LIB_PATH}")
        logger.error("Please run 'bash setup.sh' first.")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    # Process all jsonl files
    jsonl_files = list(in_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {in_path}")
        return

    for file_path in jsonl_files:
        filename = file_path.name.lower()
        
        # Simple filename filtering based on mode (same as run_rename.py)
        if args.mode == 'java' and 'java' not in filename:
            logger.info(f"Skipping {file_path.name} (Not a Java dataset)")
            continue
            
        if args.mode == 'c' and 'java' in filename:
            logger.info(f"Skipping {file_path.name} (Found 'java' in name, but mode is 'c')")
            continue

        # Output filename: add _augmented suffix
        # User requested: "lưu vào dataset đúng cái row đó nhưng thêm 1 column là transformed_code"
        # but also implied creating a new output similar to run_rename.py logic.
        output_file = out_path / f"{file_path.stem}_augmented{file_path.suffix}"
        
        logger.info(f"Processing {file_path.name} -> {output_file.name}")
        process_file(file_path, output_file, args.mode, args.workers)

if __name__ == "__main__":
    main()
