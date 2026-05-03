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
    from src.data_preprocessors.transformations.simple_perturbations import (
        CommentInserter, SpacingNormalizer, IdentifierRenamer, LineCommenter
    )
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
    OperandSwap,
    CommentInserter,
    SpacingNormalizer,
    IdentifierRenamer,
    LineCommenter
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


def process_line(args: Tuple[Dict[str, Any], str, str]) -> Tuple[Dict[str, Any], int]:
    """
    Worker function to process a single JSON object (from an array).
    
    Args:
        args: Tuple containing (data_dict, parser_path, mode)
        
    Returns:
        Tuple of (Modified data dict, Number of successful transformations)
    """
    data, parser_path, mode = args
    
    code_content = data.get("func")
    if not code_content:
        print("No func in: ", data)
        return data, 0

    file_name = data.get("file", "")
    
    # Determine language
    lang = get_parser_lang(mode, file_name)

    success_count = 0
    
    # Try all transformations
    for TransformerClass in TRANSFORMERS:
        try:
            # Instantiate per line to be safe with multiprocessing and parser state
            transformer = TransformerClass(parser_path=parser_path, language=lang)
            
            # transform_code returns: (new_code_string, meta_dict)
            new_code, meta = transformer.transform_code(code_content)
            
            # Check success flag from metadata
            success = meta.get("success", False)
            
            # Double check: sometimes success is True but code is identical (guard)
            if success and new_code.strip() != code_content.strip():
                # Store with the transformer class name as key
                data[TransformerClass.__name__] = new_code
                success_count += 1
                
        except Exception:
            # If one transformer fails (crashes), continue to the next one
            continue

    return data, success_count


def process_file(
    input_file: Path, 
    output_file: Path, 
    mode: str, 
    num_workers: int
) -> None:
    """Read input file (JSON array), process objects in parallel, and write to output."""
    
    # 1. Read JSON array
    logger.info(f"Reading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            data_list = data_list[:10]
    except Exception as e:
        logger.error(f"Failed to load {input_file}: {e}")
        return

    if not isinstance(data_list, list):
        logger.error(f"Input file {input_file} is not a JSON array (expected list at root).")
        return

    total_samples = len(data_list)
    parser_path_str = str(PARSER_LIB_PATH)
    
    # Pack arguments for worker: (data_dict, parser_path, mode)
    worker_args = [(item, parser_path_str, mode) for item in data_list]

    # 2. Process in parallel
    logger.info(f"Processing {total_samples} samples with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Results will be a list of (modified_data, success_count)
        results = list(tqdm(
            pool.imap(process_line, worker_args, chunksize=20),
            total=total_samples,
            unit="sample"
        ))
    
    # Separate data and stats
    processed_data = [r[0] for r in results]
    success_counts = [r[1] for r in results]

    # Calculate stats
    total_augmentations = sum(success_counts)
    avg_augmentations = total_augmentations / total_samples if total_samples > 0 else 0
    
    augmented_samples = sum(1 for c in success_counts if c > 0)
    
    logger.info(f"Augmented Samples: {augmented_samples}/{total_samples} ({augmented_samples/total_samples*100:.2f}%)")
    logger.info(f"Total Transformations: {total_augmentations}")
    logger.info(f"Average Transformations per Function: {avg_augmentations:.2f}")

    # 3. Write Output (JSON array)
    logger.info(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
            
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

    # Process all json files
    json_files = list(in_path.glob("*.json"))
    if not json_files:
        logger.warning(f"No .json files found in {in_path}")
        return

    for file_path in json_files:
        # Use the same filename as input
        output_file = out_path / file_path.name
        
        logger.info(f"Processing {file_path.name} -> {output_file.name}")
        process_file(file_path, output_file, args.mode, args.workers)

if __name__ == "__main__":
    main()
