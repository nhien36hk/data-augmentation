#!/usr/bin/env python3
"""
Script to apply Variable Renaming to JSONL datasets.
Optimized for performance using multiprocessing.

Usage:
    python run_rename.py --input_dir data/raw --output_dir data/renamed --mode c
    python run_rename.py --input_dir data/raw --output_dir data/renamed --mode java
"""

import os
import sys
import json
import logging
import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from functools import partial
from tqdm import tqdm

# Add project root to sys.path to import src modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import required modules
try:
    from src.data_preprocessors.transformations.var_renaming_transformation import VarRenamer
except ImportError:
    print("Error: Could not import 'src'. Make sure you are running from the project root.")
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

    try:
        # Initialize Renamer (Instantiating per line is slightly overhead but safe for multiprocessing)
        # Note: Language(parser_path, lang) loads the .so file. 
        # In extremely high throughput, we might want to initialize this once per process,
        # but for this scale, it's acceptable and safer.
        renamer = VarRenamer(parser_path=parser_path, language=lang)
        
        # Apply transformation
        # var_renaming returns: (root_node, new_code_string, success_bool)
        _, new_code, success = renamer.var_renaming(code_content)
        
        if success:
            # Update the code content
            data["func"] = new_code
            
            # Mark metadata
            data["renamed"] = True
            data["transformation"] = "var_renaming"
        
        return json.dumps(data)

    except Exception as e:
        # If augmentation fails, return original data but log warning if needed
        # Often better to return original than crash
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
    processed_lines = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for slight speedup if order strictly perfectly doesn't matter,
        # but imap preserves order which is usually nicer for datasets.
        results = list(tqdm(
            pool.imap(process_line, worker_args, chunksize=100),
            total=total_lines,
            unit="sample"
        ))
    
    # Filter Nones
    processed_lines = [r for r in results if r is not None]

    # 3. Write Output
    logger.info(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
            
    logger.info(f"Done. Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Clean Data Augmentation Runner")
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
        
        # Simple filename filtering based on mode
        if args.mode == 'java' and 'java' not in filename:
            logger.info(f"Skipping {file_path.name} (Not a Java dataset)")
            continue
            
        if args.mode == 'c' and 'java' in filename:
            logger.info(f"Skipping {file_path.name} (Found 'java' in name, but mode is 'c')")
            continue

        output_file = out_path / file_path.name 
        process_file(file_path, output_file, args.mode, args.workers)

if __name__ == "__main__":
    main()
