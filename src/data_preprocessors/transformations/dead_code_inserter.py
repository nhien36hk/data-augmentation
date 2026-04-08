import sys
import os

# Add project root to sys.path for direct execution
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import re
from typing import Union, Tuple
import numpy as np

from src.data_preprocessors.language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor,
)
from src.data_preprocessors.language_processors.go_processor import GoProcessor
from src.data_preprocessors.language_processors.ruby_processor import RubyProcessor
from src.data_preprocessors.language_processors.utils import extract_statement_within_size, get_tokens, \
    get_tokens_insert_before, count_nodes
from src.data_preprocessors.transformations import TransformationBase

processor_function = {
    "java": JavaAndCPPProcessor,
    "c": JavaAndCPPProcessor,
    "cpp": JavaAndCPPProcessor,
    "c_sharp": CSharpProcessor,
    "python": PythonProcessor,
    "javascript": JavascriptProcessor,
    "go": GoProcessor,
    "php": PhpProcessor,
    "ruby": RubyProcessor,
}

tokenizer_function = {
    "java": get_tokens,
    "c": get_tokens,
    "cpp": get_tokens,
    "c_sharp": get_tokens,
    "python": PythonProcessor.get_tokens,
    "javascript": JavascriptProcessor.get_tokens,
    "go": get_tokens,
    "php": PhpProcessor.get_tokens,
    "ruby": get_tokens,
}

# No longer strictly needed for byte-based insertion, but kept for compatibility references if any
insertion_function = {
    "java": get_tokens_insert_before,
    "c": get_tokens_insert_before,
    "cpp": get_tokens_insert_before,
    "c_sharp": get_tokens_insert_before,
    "python": PythonProcessor.get_tokens_insert_before,
    "javascript": JavascriptProcessor.get_tokens_insert_before,
    "go": get_tokens_insert_before,
    "php": PhpProcessor.get_tokens_insert_before,
    "ruby": get_tokens_insert_before,
}


class DeadCodeInserter(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(DeadCodeInserter, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        self.insertion_function = insertion_function[self.language]

    def insert_random_dead_code(self, code_string, max_node_in_statement=-1) -> Tuple[str, bool]:
        if isinstance(code_string, str):
            code_bytes = code_string.encode('utf-8')
        else:
            code_bytes = code_string

        root = self.parse_code(code_string)
        original_node_count = count_nodes(root)
        if max_node_in_statement == -1:
            max_node_in_statement = int(original_node_count / 2)
        
        statement_markers = None
        if self.language == "ruby":
            statement_markers = ["assignment", "until", "call", "if", "for", "while"]
            
        statements = extract_statement_within_size(
            root, max_node_in_statement, statement_markers,
            code_string=code_string, tokenizer=self.tokenizer_function,
        )
        
        # Valid parent types where we can safely insert a statement
        # This prevents inserting as the body of a loop/if without braces, which changes program logic
        safe_parent_types = {
            'compound_statement', # C/C++
            'block',              # Java/C#
            'translation_unit',   # C/C++ Global
            'program',            # Java Global
            'class_body',         # Java Class members
            'declaration_list',   # Go?
            'statement_block',    # Go?
            # Add others if needed for other languages, but for C/Java/CPP these covers most
        }

        # Determine number of trials
        trials = 50
        
        for _ in range(trials):
            try:
                if len(statements) < 2:
                    break
                    
                # Pick 2 distinct statements: 1 for body, 1 for insertion point
                random_idxs = np.random.choice(len(statements), 2, replace=False)
                random_stmt = statements[random_idxs[0]]
                insert_before = statements[random_idxs[1]]
                
                # VALIDATION: Check if insert_before has a safe parent
                if str(insert_before.parent.type) not in safe_parent_types:
                    continue
                
                # Extract body text using bytes to preserve formatting
                dead_code_body = code_bytes[random_stmt.start_byte:random_stmt.end_byte].decode('utf-8')
                
                dead_code_function = np.random.choice(
                    [
                        self.processor.create_dead_for_loop,
                        self.processor.create_dead_while_loop,
                        self.processor.create_dead_if
                    ]
                )
                
                # Generate dead code string
                dead_code = dead_code_function(dead_code_body)
                
                # FIX: C language does not have 'false' keyword by default (requires stdbool.h)
                # Use '0' instead for C.
                if self.language == 'c':
                    # Replace " false " with " 0 " padding with spaces to match generator output
                    dead_code = dead_code.replace(" false ", " 0 ")
                
                # INDENTATION handling
                start_byte = insert_before.start_byte
                # Find start of the line
                line_start = start_byte
                while line_start > 0 and code_bytes[line_start - 1] != 10: # 10 is newline
                    line_start -= 1
                
                # Extract indentation of the current line (whitespace at start)
                current_indent = []
                idx = line_start
                while idx < len(code_bytes) and code_bytes[idx] in [32, 9]: # space or tab
                    current_indent.append(code_bytes[idx])
                    idx += 1
                indent_str = bytes(current_indent).decode('utf-8')
                
                # Prepare insertion with correct indentation
                # Prepend newline to start separate line
                # Prepend indentation to dead code
                # Append newline and indentation for the original statement to stay aligned
                insertion = f"\n{indent_str}{dead_code}\n{indent_str}"
                
                # If insertion point is not at start of line (e.g. "stmt; stmt;"), we might break flow
                # But since we use safe_parent_types (blocks), newlines are generally safe.
                
                # Insert dead code before the chosen node
                prefix_code = code_bytes[:start_byte].decode('utf-8')
                suffix_code = code_bytes[start_byte:].decode('utf-8')
                
                # Remove trailing whitespace from prefix if we are adding our own newline/indent?
                # Actually simpler: Just insert. If there was already indentation before `start_byte`,
                # our `indent_str` duplicates it if we are not careful about `line_start`.
                
                # CASE 1: `start_byte` IS `idx` (We are at the first non-whitespace char of line)
                # Prefix ends with `indent_str`.
                # If we add `\n{indent_str}{dead}...`, we get:
                # `...non-white\n{indent_str}\n{indent_str}{dead}...` -> Empty line with indent.
                
                # Better approach for clean look:
                if start_byte == idx:
                    # We are at start of content on the line.
                    # Insert: `dead_code\n{indent_str}`
                    # The `dead_code` needs to be indented.
                    # So: `{dead_code}\n{indent_str}`
                    # Wait, where does `dead_code` start? It needs `indent_str` at its own start.
                    
                    # But `prefix_code` ALREADY ends with `indent_str` (the whitespace of current line).
                    # So: `prefix_code` = `...\n    `
                    # We want: `...\n    dead_code\n    original...`
                    # So we just append `dead_code\n{indent_str}` to prefix?
                    # `...\n    ` + `dead_code\n    ` + `original...`
                    # Result: `...\n    dead_code\n    original...` -> Correct!
                    
                    insertion = f"{dead_code}\n{indent_str}"
                else:
                    # We are in middle of line: `stmt1; <here>stmt2;`
                    # We want:
                    # `stmt1; `
                    # `    dead_code`
                    # `    stmt2;`
                    insertion = f"\n{indent_str}{dead_code}\n{indent_str}"

                new_code = prefix_code + insertion + suffix_code
                
                # Basic check if code actually changed
                if new_code != code_string:
                   return new_code, True

            except Exception as e:
                # In case of any error allow retrying with different statements
                pass
                
        return code_string, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        
        code_str = code
        if isinstance(code, bytes):
            code_str = code.decode('utf-8')
            
        new_code, success = self.insert_random_dead_code(code_str, -1)
        
        return new_code, {
            "success": success
        }


if __name__ == '__main__':
    # Test cases
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    }
    """
    c_code = """
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    """
    
    # We focus on the requested languages in the prompt (and what works generally)
    # Adding Java, C, CPP as primary test targets
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
    }

    # Setup parser path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    parser_path = os.path.join(project_root, "parser/languages.so")
    
    for lang_key in ["c", "cpp", "java"]:
        if lang_key not in input_map: continue
        
        lang, code = input_map[lang_key]
        
        print(f"\n{'='*20} TESTING {lang.upper()} {'='*20}")
        print("--- ORIGINAL ---")
        print(code)

        try:
            inserter = DeadCodeInserter(parser_path, lang)
            code, meta = inserter.transform_code(code)
            
            print("--- TRANSFORMED ---")
            print(code)
            print("-" * 50)
            print(f"Success: {meta['success']}")
            print(f"Metadata: {meta}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

