import numpy as np
from typing import Dict, Callable

from src.data_preprocessors.transformations.block_swap_transformations import BlockSwap
from src.data_preprocessors.transformations.confusion_remove import ConfusionRemover
from src.data_preprocessors.transformations.dead_code_inserter import DeadCodeInserter
from src.data_preprocessors.transformations.for_while_transformation import ForWhileTransformer
from src.data_preprocessors.transformations.operand_swap_transformations import OperandSwap
from src.data_preprocessors.transformations.var_renaming_transformation import VarRenamer


class SemanticPreservingTransformation:
    def __init__(
            self,
            parser_path: str,
            language: str,
            transform_functions: Dict[Callable, int] = None,
    ):
        self.language = language
        # Ignore weighting; use one instance per transform (no SyntacticNoising).
        if transform_functions is None:
            self.transform_functions = {
                BlockSwap: 1,
                ConfusionRemover: 1,
                DeadCodeInserter: 1,
                ForWhileTransformer: 1,
                OperandSwap: 1,
                VarRenamer: 1
            }
        else:
            self.transform_functions = transform_functions
        self.transformations = []
        for t in self.transform_functions:
            for _ in range(self.transform_functions[t]):
                self.transformations.append(t(parser_path=parser_path, language=language))

    def _is_valid_code(self, code: str) -> bool:
        """
        Parse code with tree-sitter and reject if parser reports errors.
        Reuses the parser of the first transformation (all share same language/parser).
        """
        if not self.transformations:
            return False
        try:
            root = self.transformations[0].parse_code(code)
        except Exception:
            return False
        if getattr(root, "has_error", False):
            return False
        # Ensure parser consumed the full buffer.
        if hasattr(root, "start_byte") and hasattr(root, "end_byte"):
            if root.start_byte != 0 or root.end_byte != len(code.encode()):
                return False
        return True

    def transform_code(
            self,
            code: str
    ):
        variants, stats = self.transform_code_multi(code, num_variants=10)
        if len(variants) == 0:
            return code, stats
        # Return the last stacked variant for compatibility
        transformed_code = variants[-1]
        return transformed_code, stats

    def transform_code_multi(self, code: str, num_variants: int = 10):
        variants = []
        valid_count = 0
        invalid_count = 0

        for _ in range(num_variants):
            code_current = code
            indices = list(range(len(self.transformations)))
            np.random.shuffle(indices)
            applied = False
            for idx in indices:
                if np.random.uniform() > 0.5:
                    t = self.transformations[idx]
                    out, meta = t.transform_code(code_current)
                    if meta.get("success", False):
                        code_current = out
                        applied = True
            if not applied:
                # Fallback: try first in shuffled list
                t = self.transformations[indices[0]]
                out, meta = t.transform_code(code_current)
                if meta.get("success", False):
                    code_current = out
                    applied = True
            if applied:
                # Always keep applied variants; track validity for stats
                if self._is_valid_code(code_current):
                    valid_count += 1
                else:
                    invalid_count += 1
                variants.append(code_current)

        stats = {
            "valid_variants": valid_count,
            "invalid_variants": invalid_count,
        }
        return variants, stats
