import argparse
import json
import os
from typing import Dict

from tree_sitter import Language, Parser


def build_parser(cache: Dict[str, Parser], parser_path: str, lang: str) -> Parser:
    if lang not in cache:
        lang_obj = Language(parser_path, lang)
        parser = Parser()
        parser.set_language(lang_obj)
        cache[lang] = parser
    return cache[lang]


def has_error(parser: Parser, code: str) -> bool:
    try:
        root = parser.parse(code.encode()).root_node
        return getattr(root, "has_error", False)
    except Exception:
        # If parse throws, treat as error
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect a sample of functions whose original code fails tree-sitter parsing."
    )
    parser.add_argument("--input", required=True, help="Path to train.json (one JSON per line).")
    parser.add_argument("--parser_path", required=True, help="Path to parser/languages.so")
    parser.add_argument("--output", required=True, help="Path to write sampled invalid functions (JSONL).")
    parser.add_argument(
        "--limit", type=int, default=30, help="Number of failing samples to collect (default: 30)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    if not os.path.exists(args.parser_path):
        raise FileNotFoundError(args.parser_path)

    parser_cache: Dict[str, Parser] = {}
    collected = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if collected >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            lang = str(rec.get("lang", "")).lower()
            if lang == "cpp":
                lang_key = "cpp"
            elif lang == "c":
                lang_key = "c"
            else:
                continue

            func_before = rec.get("func_before", "")
            if not isinstance(func_before, str) or not func_before.strip():
                continue

            ts_parser = build_parser(parser_cache, args.parser_path, lang_key)
            if has_error(ts_parser, func_before):
                fout.write(json.dumps(rec) + "\n")
                collected += 1

    print(f"Collected {collected} samples with parse errors into {args.output}")


if __name__ == "__main__":
    main()

