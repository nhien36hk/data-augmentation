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
    PhpProcessor,
    GoProcessor,
    RubyProcessor
)
from src.data_preprocessors.transformations import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.operand_swap],
    "c": [JavaAndCPPProcessor.operand_swap],
    "cpp": [JavaAndCPPProcessor.operand_swap],
    "c_sharp": [CSharpProcessor.operand_swap],
    "python": [PythonProcessor.operand_swap],
    "javascript": [JavascriptProcessor.operand_swap],
    "go": [GoProcessor.operand_swap],
    "php": [PhpProcessor.operand_swap],
    "ruby": [RubyProcessor.operand_swap],
}


class OperandSwap(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(OperandSwap, self).__init__(parser_path=parser_path, language=language)
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
            modified_code, success = function(code, self)
            if success:
                code = modified_code
        
        # Calculate types metadata if possible, but return the modified code directly
        # Determine strict formatting based on language
        if self.language in ["java", "c", "cpp"]:
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
                
                # Apply beautification for supported languages
                code = JavaAndCPPProcessor.beautify_java_code(tokens)
            except:
                types = None
        else:
            # Fallback for other languages (though they fail parser check currently)
            # Just return the modified code or do minimal spacing
            types = None
            if isinstance(code, bytes):
                code = code.decode()

        return code, {
            "types": types,
            "success": success
        }


if __name__ == '__main__':
    java_code = """
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    python_code = """
        from typing import List

        def factorize(n: int) -> List[int]:
            import math
            fact = []
            i = 2
            while i <= int(math.sqrt(n) + 1):
                if n % i == 0:
                    fact.append(i)
                    n //= i
                else:
                    i += 1
            if n > 1:
                fact.append(n)
            return fact
        """
    c_code = """
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    cs_code = """
        void foo(){
            int time = 20;
            if (time < 18) {
              time=10;
            }
             else {
              System.out.println("Good evening.");
            }
        }
        """
    js_code = """function foo(n) {
            if (time < 10) {
              greeting = "Good morning";
            } 
            else {
              greeting = "Good evening";
            }
        }
        """
    ruby_code = """
        x = 1
        if x > 2
           puts "x is greater than 2"   
        else
           puts "I can't guess the number"
        end
        """
    go_code = """
        func main() {
           /* local variable definition */
           var a int = 100;

           /* check the boolean condition */
           if( a < 20 ) {
              /* if condition is true then print the following */
              fmt.Printf("a is less than 20\\n" );
           } else {
              /* if condition is false then print the following */
              fmt.Printf("a is not less than 20\\n" );
           }
           fmt.Printf("value of a is : %d\\n", a);
        }
        """
    php_code = """
        <?php 
        $t = date("H");
        if ($t < "10") {
          echo "Have a good morning!";
        }  else {
          echo "Have a good night!";
        }
        ?> 
        """
    
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    parser_path = os.path.join(project_root, "parser/languages.so")
    
    # Verify parser exists
    if not os.path.exists(parser_path):
        print(f"WARNING: Parser not found at {parser_path}")
    
    print("=" * 80)
    print("STARTING OPERAND SWAP TEST (Supported Languages Only)")
    print("=" * 80)

    # Only test languages that are known to work with the current .so file
    for lang in ["java", "c", "cpp"]:
        if lang not in input_map: continue
        
        lang_key, code = input_map[lang]
        
        print(f"\n>>>> TESTING LANGUAGE: {lang.upper()} <<<<")
        print("--- ORIGINAL CODE ---")
        print(code.strip())
        print("-" * 40)
        
        try:
            operandswap = OperandSwap(parser_path, lang_key)
            code, meta = operandswap.transform_code(code)
            
            print("--- TRANSFORMED CODE ---")
            print(code.strip())
            print("-" * 40)
            print(f"Success: {meta['success']}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
