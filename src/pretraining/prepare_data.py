import argparse
import json
import multiprocessing
import os
import random
from multiprocessing import Pool, cpu_count

import nltk
import numpy as np
import torch
from tqdm import tqdm

from src.data_preprocessors.transformations import (
    NoTransformation, SemanticPreservingTransformation,
    BlockSwap, ConfusionRemover, DeadCodeInserter,
    ForWhileTransformer, OperandSwap, VarRenamer
)


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
            func_before = record.get('func_before', '')
            func_after = record.get('func_after', '')
            vul = int(record.get('vul', 0))
            original_code = func_before if isinstance(func_before, str) else str(func_before)
            if len(original_code.split()) > self.max_function_length:
                return -1
            variants = example_transformer.transform_code_multi(original_code, num_variants=10)
            if len(variants) == 0:
                return -1
            return {
                'func_before': original_code,
                'func_after': func_after,
                'vul': vul,
                'transformed_code': variants
            }
        except KeyboardInterrupt:
            print("Stopping parsing for ", record)
            return -1
        except:
            return -1


def process_split(pool, example_processor, records, output_path, append=False):
    used_transformers = {}
    success = 0
    total_variants = 0
    out_f = open(output_path, "at" if append else "wt")
    with tqdm(total=len(records)) as pbar:
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
                name = rec.get("transformer", "Mixed")
                if name not in used_transformers.keys():
                    used_transformers[name] = 0
                used_transformers[name] += 1
                if "transformed_code" in rec and isinstance(rec["transformed_code"], list):
                    total_variants += len(rec["transformed_code"])
                out_f.write(json.dumps(rec) + "\n")
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
            Stats   : {json.dumps(used_transformers, indent=4)}
            """
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Directory containing train.json/val.json/test.json")
    parser.add_argument(
        '--output_dir', required=True, help="Directory for saving processed code"
    )
    parser.add_argument(
        '--parser_path', help="Tree-Sitter Parser Path", required=True
    )
    parser.add_argument("--workers", help="Number of worker CPU", type=int, default=20)
    parser.add_argument("--timeout", type=int, help="Maximum number of seconds for a function to process.", default=10)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--max_function_length", type=int, default=400)
    parser.add_argument("--num_variants", type=int, default=10)
    args = parser.parse_args()
    set_seeds(args.seed)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    # Expect train.json, val.json, test.json in input_dir
    split_files = {
        "train": os.path.join(args.input_dir, "train.json"),
        "val": os.path.join(args.input_dir, "val.json"),
        "test": os.path.join(args.input_dir, "test.json"),
    }
    for split, path in split_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")

    for split, path in split_files.items():
        print(f"Now processing split: {split}")
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]

        # Partition raw records by language tag (expected values: C, CPP)
        c_records = [r for r in data if str(r.get("lang", "")).upper() == "C"]
        cpp_records = [r for r in data if str(r.get("lang", "")).upper() == "CPP"]

        print("Total Func Lang C: ", len(c_records))
        print("Total Func Lang CPP: ", len(cpp_records))

        output_path = os.path.join(out_dir, f"{split}.json")
        wrote_any = False

        for lang_key, records in [("c", c_records), ("cpp", cpp_records)]:
            if not records:
                continue
            example_processor = ExampleProcessor(
                language=lang_key,
                parser_path=args.parser_path,
                transformation_config={},
                max_function_length=args.max_function_length
            )
            pool = Pool(
                processes=min(cpu_count(), args.workers),
                initializer=example_processor.initialize
            )
            process_split(
                pool=pool,
                example_processor=example_processor,
                records=records,
                output_path=output_path,
                append=wrote_any
            )
            pool.close()
            pool.join()
            del pool
            del example_processor
            wrote_any = True
