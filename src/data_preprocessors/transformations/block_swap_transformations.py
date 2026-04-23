import sys
import os

# Add project root to sys.path for direct execution
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import copy
import os
import re
from typing import Union, Tuple

import numpy as np

from src.data_preprocessors.language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor,
    RubyProcessor,
    GoProcessor
)
from src.data_preprocessors.transformations.transformation_base import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.block_swap_java],
    "c": [JavaAndCPPProcessor.block_swap_c],
    "cpp": [JavaAndCPPProcessor.block_swap_c],
    "c_sharp": [CSharpProcessor.block_swap],
    "python": [PythonProcessor.block_swap],
    "javascript": [JavascriptProcessor.block_swap],
    "go": [GoProcessor.block_swap],
    "php": [PhpProcessor.block_swap],
    "ruby": [RubyProcessor.block_swap],
}


class BlockSwap(TransformationBase):
    """
    Swapping if_else block
    """

    def __init__(self, parser_path, language):
        super(BlockSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,  # yes
            "c": self.get_tokens_with_node_type,  # yes
            "cpp": self.get_tokens_with_node_type,  # yes
            "c_sharp": self.get_tokens_with_node_type,  # yes
            "javascript": JavascriptProcessor.get_tokens,  # yes
            "python": PythonProcessor.get_tokens,  # no
            "php": PhpProcessor.get_tokens,  # yes
            "ruby": self.get_tokens_with_node_type,  # yes
            "go": self.get_tokens_with_node_type,  # no
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        
        # Try random transformations until one succeeds or we run out
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            # The processor functions (e.g. block_swap_c) now return the code string directly
            modified_code, success = function(code, self)
            if success:
                code = modified_code

        # Return code directly to preserve formatting (newlines, indentation)
        # We don't need to re-parse and tokenize just to flatten it.
        return code, {
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
          fmt.Printf("a is less than 20\n" );
       } else {
          /* if condition is false then print the following */
          fmt.Printf("a is not less than 20\n" );
       }
       fmt.Printf("value of a is : %d\n", a);
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
    # Assuming script is in src/data_preprocessors/transformations/
    # Project root (data-augmentation) is 3 levels up
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    parser_path = os.path.join(project_root, "parser/languages.so")
    
    # Test only requested languages: Java, C, CPP
    for lang in ["java", "c", "cpp"]:
        if lang not in input_map: continue
        
        lang_key, code = input_map[lang]
        print(f"\n{'='*20} TESTING {lang.upper()} {'='*20}")
        print("--- ORIGINAL ---")
        print(code)
        
        block_swap = BlockSwap(parser_path, lang_key)
        
        try:
            mod_code, meta = block_swap.transform_code(code)
            
            print("--- TRANSFORMED ---")
            print(mod_code)
            print("-" * 50)
            print(f"Success: {meta['success']}")
            print(f"Metadata: {meta}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
