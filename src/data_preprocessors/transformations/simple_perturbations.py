import random
import re
from typing import Union, Tuple
from src.data_preprocessors.transformations.transformation_base import TransformationBase

class CommentInserter(TransformationBase):
    def transform_code(self, code: Union[str, bytes]) -> Tuple[str, dict]:
        if isinstance(code, bytes):
            code = code.decode()
        out = code + f"\n/* noise_{random.randint(0, 9999)} */\n"
        return out, {"success": True}

class SpacingNormalizer(TransformationBase):
    def transform_code(self, code: Union[str, bytes]) -> Tuple[str, dict]:
        if isinstance(code, bytes):
            code = code.decode()
        out = re.sub(r"[ \t]+", " ", code)
        success = out != code
        return out, {"success": success}

class IdentifierRenamer(TransformationBase):
    def transform_code(self, code: Union[str, bytes]) -> Tuple[str, dict]:
        if isinstance(code, bytes):
            code = code.decode()
        out = code
        # Basic identifier pattern
        _identifier_pat = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
        ids = list(dict.fromkeys(_identifier_pat.findall(out)))
        
        blacklist = {
            "int", "char", "float", "double", "void", "return", "if", "else", "for",
            "while", "do", "switch", "case", "break", "continue", "struct", "class",
            "public", "private", "protected", "static", "const", "sizeof", "new",
            "delete", "try", "catch", "throw", "include", "define",
            "NULL", "true", "false", "String", "System", "out", "println", 
            "HttpServletRequest", "HttpServletResponse", "Throwable"
        }
        
        ids = [x for x in ids if x not in blacklist and len(x) >= 2]
        if not ids:
            return out, {"success": False}
        
        random.shuffle(ids)
        # Using a fixed epsilon-like factor for renaming (e.g. 0.2)
        epsilon = 0.2
        n_rename = min(10, max(1, int(len(ids) * min(0.30, max(0.10, epsilon)))))
        
        mapping = {ids[i]: f"v_{i}_{random.randint(0, 999)}" for i in range(min(n_rename, len(ids)))}
        
        for old, new in mapping.items():
            out = re.sub(rf"\b{re.escape(old)}\b", new, out)
            
        return out, {"success": True}

class LineCommenter(TransformationBase):
    def transform_code(self, code: Union[str, bytes]) -> Tuple[str, dict]:
        if isinstance(code, bytes):
            code = code.decode()
        lines = code.splitlines()
        candidate_idx = []
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if s.startswith("//") or s.startswith("/*") or s.endswith("*/"):
                continue
            candidate_idx.append(i)

        if not candidate_idx:
            return code, {"success": False}

        j = random.choice(candidate_idx)
        lines[j] = "// XAI_PERTURB " + lines[j]
        out = "\n".join(lines)
        return out, {"success": True}
