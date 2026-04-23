import copy
import os
import sys

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
    PhpProcessor
)
from src.data_preprocessors.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "cpp": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c_sharp": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "python": [PythonProcessor.for_to_while_random, PythonProcessor.while_to_for_random],
    "javascript": [JavascriptProcessor.for_to_while_random, JavascriptProcessor.while_to_for_random],
    "go": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "php": [PhpProcessor.for_to_while_random, PhpProcessor.while_to_for_random],
    "ruby": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
}


class ForWhileTransformer(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,
            "c": self.get_tokens_with_node_type,
            "cpp": self.get_tokens_with_node_type,
            "c_sharp": self.get_tokens_with_node_type,
            "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
            "php": PhpProcessor.get_tokens,
            "ruby": self.get_tokens_with_node_type,
            "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            if success:
                code = modified_code
        
        # Calculate metadata types if needed, but return the modified code string directly
        # to preserve whatever formatting the processor function managed to keep/generate.
        try:
            root_node = self.parse_code(code=code)
            return_values = self.final_processor(
                code=code.encode() if isinstance(code, str) else code,
                root=root_node
            )
            if isinstance(return_values, tuple):
                tokens, types = return_values
            else:
                tokens, types = return_values, None
        except:
            types = None

        return code, {
            "types": types,
            "success": success
        }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(i = 0; i < n; i++) {
                int j = 0;
                if (i == 0){
                    foo(7);
                    continue;
                }
                while (j < 10) {
                    j++;
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
                    j++;
                }
            }
            return res;
        }
    """
    
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    parser_path = os.path.join(project_root, "parser/languages.so")
    
    for lang in ["java", "c", "cpp"]:
        if lang not in input_map: continue
        
        lang_key, code = input_map[lang]
        
        print(f"\n{'='*20} TESTING {lang.upper()} {'='*20}")
        print("--- ORIGINAL ---")
        print(code)
        
        for_while_transformer = ForWhileTransformer(parser_path, lang_key)
        
        try:
            code, meta = for_while_transformer.transform_code(code)
            
            print("--- TRANSFORMED ---")
            print(code)
            print("-" * 50)
            print(f"Success: {meta['success']}")
            print(f"Metadata: {meta['types'] is not None}") 
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()