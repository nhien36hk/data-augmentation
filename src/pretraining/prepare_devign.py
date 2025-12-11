import argparse
import json
import multiprocessing
import os
import random
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from tqdm import tqdm

from src.data_preprocessors.transformations import (
    NoTransformation, SemanticPreservingTransformation,
    BlockSwap, ConfusionRemover, DeadCodeInserter,
    ForWhileTransformer, OperandSwap, VarRenamer
)
from src.pretraining.utils import loads, load_split_datasets


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_transformers_from_conf_file():
    # Ignore weighting/config; use one instance per transform, exclude noising.
    return {
        BlockSwap: 1,
        ConfusionRemover: 1,
        DeadCodeInserter: 1,
        ForWhileTransformer: 1,
        OperandSwap: 1,
        VarRenamer: 1
    }

# --------------------------------------------------------------------------- #
# Language detection (C vs C++) using compiled regex patterns
# --------------------------------------------------------------------------- #
_CPP_PATTERNS = [
    re.compile(r'\w+\s*::\s*\w+'),
    re.compile(r'\bclass\s+\w+'),
    re.compile(r'\bnamespace\s+\w+'),
    re.compile(r'\btemplate\s*<'),
    re.compile(r'\bvirtual\s+'),
    re.compile(r'\boverride\b'),
    re.compile(r'\bnullptr\b'),
    re.compile(r'\bpublic\s*:'),
    re.compile(r'\bprivate\s*:'),
    re.compile(r'\bprotected\s*:'),
    re.compile(r'\bnew\s+\w+'),
    re.compile(r'\bdelete\s+'),
    re.compile(r'\bdelete\s*\['),
    re.compile(r'\bstd::'),
    re.compile(r'\bconstexpr\b'),
    re.compile(r'\bauto\s+\w+\s*='),
    re.compile(r'\btry\s*\{'),
    re.compile(r'\bcatch\s*\('),
    re.compile(r'\bthrow\s+'),
    re.compile(r'\busing\s+namespace\b'),
    re.compile(r'\bfriend\s+'),
    re.compile(r'\bexplicit\s+'),
    re.compile(r'\binline\s+'),
    re.compile(r'\bconst_cast\s*<'),
    re.compile(r'\bstatic_cast\s*<'),
    re.compile(r'\bdynamic_cast\s*<'),
    re.compile(r'\breinterpret_cast\s*<'),
    re.compile(r'\btypename\s+'),
    re.compile(r'\boperator\s*[+\-*/%=<>!&|^~\[\]()]+'),
    re.compile(r'::\s*~\w+'),
    re.compile(r'\w+\s*<[^>]+>\s*\w+'),
]


def detect_language(code: str) -> str:
    """Return 'cpp' if any C++-specific pattern matches; otherwise 'c'."""
    for p in _CPP_PATTERNS:
        if p.search(code):
            return "cpp"
    return "c"


class ExampleProcessor:
    def __init__(
            self,
            language,
            parser_path,
            transformation_config,
            max_function_length=400
    ):
        self.language = language
        self.parser_path = parser_path
        self.transformation_config = transformation_config
        self.max_function_length = max_function_length

    def initialize(self):
        global example_transformer
        transformers = create_transformers_from_conf_file()
        example_transformer = SemanticPreservingTransformation(
            parser_path=self.parser_path, language=self.language, transform_functions=transformers
        )

    def process_example(self, record):
        global example_transformer
        try:
            func = record.get('func', '')
            target = record.get('target', '')
            original_code = func if isinstance(func, str) else str(func)
            if len(original_code.split()) > self.max_function_length:
                return -1
            variants, stats = example_transformer.transform_code_multi(original_code, num_variants=10)
            if len(variants) == 0:
                return -1
            return {
                'func': original_code,
                'target': target,
                'variants': variants,
                'transform_stats': stats
            }
        except KeyboardInterrupt:
            print("Stopping parsing for ", record)
            return -1
        except:
            return -1


def process_split(pool, example_processor, records, output_path, append=False):
    success = 0
    total_variants = 0
    total_valid_variants = 0
    total_invalid_variants = 0
    out_f = open(output_path, "at" if append else "wt")
    with tqdm(total=len(records), desc=f"proc-{os.path.basename(output_path)}", leave=False) as pbar:
        processed_example_iterator = pool.imap(
            func=example_processor.process_example,
            iterable=records,
            chunksize=1000,
        )
        count = 0
        while True:
            pbar.update()
            count += 1
            try:
                rec = next(processed_example_iterator)
                if isinstance(rec, int) and rec == -1:
                    continue
                variants = rec.get("variants", [])
                stats = rec.get("transform_stats", {})
                total_variants += len(variants) if isinstance(variants, list) else 0
                if isinstance(stats, dict):
                    total_valid_variants += int(stats.get("valid_variants", 0))
                    total_invalid_variants += int(stats.get("invalid_variants", 0))
                # Keep only required fields
                out_f.write(json.dumps({
                    "func": rec.get("func", ""),
                    "target": rec.get("target", ""),
                    "variants": variants,
                }) + "\n")
                out_f.flush()
                success += 1
            except multiprocessing.TimeoutError:
                print(f"{count} encountered timeout")
            except StopIteration:
                print(f"{count} stop iteration")
                break
    out_f.close()
    print(
        f"""
            Total   : {len(records)}, 
            Success : {success},
            Failure : {len(records) - success}
            Total Variants : {total_variants}
            Valid Variants : {total_valid_variants}
            Invalid Variants : {total_invalid_variants}
            """
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Directory containing Devign pickles and split_idx.pkl")
    parser.add_argument('--output_dir', required=True, help="Directory for saving processed code (JSONL)")
    parser.add_argument('--parser_path', help="Tree-Sitter Parser Path", required=True)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--max_function_length", type=int, default=4000)
    args = parser.parse_args()
    set_seeds(args.seed)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load Devign dataset and splits
    devign_df = loads(args.input_dir)
    train_df, val_df, test_df, test_short_df, test_long_df = load_split_datasets(
        args.input_dir, devign_df
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "test_short": test_short_df,
        "test_long": test_long_df,
    }

    for split_name, df in splits.items():
        output_path = os.path.join(out_dir, f"{split_name}.jsonl")
        print(f"Now processing split: {split_name}, records: {len(df)}")

        records = df[["func", "target"]].to_dict("records")
        c_records, cpp_records = [], []
        for r in tqdm(records, desc=f"Detect lang {split_name}", leave=False):
            func = r.get("func", "")
            lang_detected = detect_language(func if isinstance(func, str) else str(func))
            if lang_detected == "cpp":
                cpp_records.append(r)
            else:
                c_records.append(r)

        print("Total Func Lang C: ", len(c_records))
        print("Total Func Lang CPP: ", len(cpp_records))

        wrote_any = False
        for lang_key, recs in [("c", c_records), ("cpp", cpp_records)]:
            if not recs:
                continue
            example_processor = ExampleProcessor(
                language=lang_key,
                parser_path=args.parser_path,
                transformation_config={},
                max_function_length=args.max_function_length
            )
            pool = Pool(
                processes=min(cpu_count(), 20),
                initializer=example_processor.initialize
            )
            process_split(
                pool=pool,
                example_processor=example_processor,
                records=recs,
                output_path=output_path,
                append=wrote_any
            )
            pool.close()
            pool.join()
            del pool
            del example_processor
            wrote_any = True
