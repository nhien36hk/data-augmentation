#!/usr/bin/env python3
"""
Script to apply multiple Random Code Transformations to C/C++ datasets for CoCa paper reproduction.
Applies transformations probabilistically (lottery with threshold p > 0.5) in a chained manner.

Usage:
    python run_augmentation.py --input_dir data/raw/devign_split --output_dir data/augmented/devign_split
"""

import os
import sys
import json
import logging
import argparse
import multiprocessing
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Type
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
    from src.data_preprocessors.transformations.var_renaming_transformation import VarRenamer
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
    VarRenamer
]


def process_sample(args: Tuple[Dict[str, Any], str, float]) -> Dict[str, Any]:
    data, parser_path, prob_threshold = args
    if not isinstance(data, dict):
        return data

    code_content = data.get("func")
    if not code_content:
        data["func_mutated"] = ""
        data["transformations_used"] = []
        data["augmented"] = False
        return data

    current_code = code_content
    transformations_used: List[str] = []

    # Roll lottery for all transformers and chain successful transformations
    for TransformerClass in TRANSFORMERS:
        # Lottery check: only apply transformer if random draw > prob_threshold (0.5)
        if random.random() <= prob_threshold:
            continue

        try:
            transformer = TransformerClass(parser_path=parser_path, language="cpp")
            new_code, meta = transformer.transform_code(current_code)
            
            success = meta.get("success", False)
            if success and new_code and new_code.strip() != current_code.strip():
                current_code = new_code
                transformations_used.append(TransformerClass.__name__)
        except Exception:
            # If a transformer fails on this sample, safely proceed to next
            continue

    # Update sample dictionary with original code in 'func' and augmented in 'func_mutated'
    data["func_mutated"] = current_code
    data["transformations_used"] = transformations_used
    data["augmented"] = len(transformations_used) > 0

    return data


def process_file(
    input_file: Path, 
    output_file: Path, 
    num_workers: int,
    prob_threshold: float = 0.5
) -> None:
    """Read input JSON/JSONL file, process samples in parallel, and write to output."""
    
    logger.info(f"Reading {input_file}...")
    
    is_jsonl = input_file.suffix.lower() == ".jsonl"
    
    if is_jsonl:
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f if line.strip()]
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

    if not isinstance(samples, list):
        logger.error(f"Expected a list of records in {input_file}, got {type(samples)}")
        return

    total_samples = len(samples)
    parser_path_str = str(PARSER_LIB_PATH)
    
    # Pack arguments for worker
    worker_args = [(sample, parser_path_str, prob_threshold) for sample in samples]

    # Process in parallel
    logger.info(f"Processing {total_samples} samples with {num_workers} workers (p_threshold={prob_threshold})...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        processed_samples = list(tqdm(
            pool.imap(process_sample, worker_args, chunksize=50),
            total=total_samples,
            unit="sample"
        ))

    # Filter out samples where augmentation failed (augmented is False)
    final_samples = [sample for sample in processed_samples if sample.get("augmented", False)]
    augmented_count = len(final_samples)
    logger.info(f"Augmentation Success Rate: {augmented_count}/{total_samples} ({augmented_count/total_samples*100:.2f}%)")

    # Write Output (only augmented samples)
    logger.info(f"Writing {augmented_count} augmented samples to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if is_jsonl:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in final_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_samples, f, indent=2, ensure_ascii=False)
            
    logger.info(f"Done. Saved {augmented_count} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Code Augmentation Runner for CoCa Paper (Chained Transformations)")
    parser.add_argument("--input_dir", type=str, default="data/raw/devign_split", help="Path to raw json/jsonl folder")
    parser.add_argument("--output_dir", type=str, default="data/augmented/devign_split", help="Path to save augmented json/jsonl")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Lottery probability threshold (> 0.5 triggers transformation)")
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

    # Process all json and jsonl files in input_dir
    data_files = sorted(list(in_path.glob("*.json")) + list(in_path.glob("*.jsonl")))
    if not data_files:
        logger.warning(f"No .json or .jsonl files found in {in_path}")
        return

    for file_path in data_files:
        output_file = out_path / file_path.name
        
        logger.info(f"Processing {file_path.name} -> {output_file.name}")
        process_file(file_path, output_file, args.workers, args.prob_threshold)

if __name__ == "__main__":
    main()
