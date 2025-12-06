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

    def transform_code(
            self,
            code: str
    ):
        variants = self.transform_code_multi(code, num_variants=10)
        if len(variants) == 0:
            return code, None
        # Return the last stacked variant for compatibility
        transformed_code = variants[-1]
        return transformed_code, None

    def transform_code_multi(self, code: str, num_variants: int = 10):
        variants = []

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
                variants.append(code_current)
        return variants
