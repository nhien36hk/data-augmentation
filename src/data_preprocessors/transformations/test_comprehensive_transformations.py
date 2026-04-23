
import sys
import os
import traceback

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import transformations
try:
    from src.data_preprocessors.transformations.block_swap_transformations import BlockSwap
    from src.data_preprocessors.transformations.confusion_remove import ConfusionRemover
    from src.data_preprocessors.transformations.dead_code_inserter import DeadCodeInserter
    from src.data_preprocessors.transformations.for_while_transformation import ForWhileTransformer
    from src.data_preprocessors.transformations.operand_swap_transformations import OperandSwap
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def get_samples():
    c_code = """
void matrix_op(int n, int m, int **arr) {
    int sum = 0;
    if (n > m) {
        sum = n + m;
    } else {
        sum = m - n;
    }
    for(int i = 0; i < n; i++) {
        if (arr[i][0] < 0) continue;
        int *ptr = &arr[i][0];
        while (*ptr != 0) {
            *ptr *= 2;
            ptr++;
            if (*ptr > 100) break;
        }
    }
    int result = (sum > 100) ? 100 : sum;
}
"""

    java_code = """
public class DataProcessor {
    public void process(int items) {
        int retry = 0;
        if (items > 0) {
            retry = 1;
        } else {
            retry = 0;
        }
        while (retry < 3) {
            retry++;
            items *= 2;
        }
        for (int i = 0; i < items; i++) {
            if (i == 5) return;
        }
        int status = (retry == 3) ? 1 : 0;
    }
}
"""

    cpp_code = """
void process_vector(int size) {
    int count = 0;
    if (size < 0) {
        size = 0;
    } else {
        size++;
    }
    for (int it = 0; it < size; ++it) {
        if (it < 0) {
            continue;
        }
        int local = 0;
        while (1) {
            if (local > 5) break; 
            local++;
        }
    }
    int val = size * 2;
    int check = (val > 10) ? val : 10;
}
"""
    return [
        ("c", c_code),
        ("java", java_code),
        ("cpp", cpp_code)
    ]

def test_all_transformations():
    parser_path = os.path.join(project_root, "parser/languages.so")
    
    if not os.path.exists(parser_path):
        print(f"CRITICAL WARNING: Parser not found at {parser_path}")
        print("Transformations requiring Tree-sitter will fail.")

    transformers = [
        ("BlockSwap", BlockSwap),
        ("ConfusionRemover", ConfusionRemover),
        ("DeadCodeInserter", DeadCodeInserter),
        ("ForWhileTransformer", ForWhileTransformer),
        ("OperandSwap", OperandSwap)
    ]

    print("="*80)
    print("STARTING COMPREHENSIVE TRANSFORMATION TEST")
    print("="*80)

    samples = get_samples()

    for name, TransformerClass in transformers:
        print(f"\n\n{'#'*30} {name.upper()} {'#'*30}")
        
        for lang, code in samples:
            print(f"\n>>>> TESTING LANGUAGE: {lang.upper()} <<<<")
            print("--- ORIGINAL CODE ---")
            print(code.strip())
            print("-" * 40)
            
            try:
                transformer = TransformerClass(parser_path, lang)
                mod_code, meta = transformer.transform_code(code)
                
                print("--- TRANSFORMED CODE ---")
                print(mod_code.strip())
                print("-" * 40)
                
                status = "SUCCESS" if meta.get('success', False) else "NO CHANGE / FAILED"
                print(f"Status: {status}")
                if 'types' in meta and meta['types']:
                    print(f"Types: {meta['types']}")
                
            except Exception as e:
                print(f"ERROR executing {name}: {e}")
                traceback.print_exc()

    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_all_transformations()
