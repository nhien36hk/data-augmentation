import math
import random
import re
from typing import Union, Tuple
import os

from src.data_preprocessors.language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor,
)
from src.data_preprocessors.language_processors.go_processor import GoProcessor
from src.data_preprocessors.language_processors.ruby_processor import RubyProcessor
from src.data_preprocessors.language_processors.utils import get_tokens
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


class VarRenamer(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(VarRenamer, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        
        # We handle exclusion logic manually in extract_targets now
        # but keep this for reference or fallback
        self.not_var_ptype = [] 

        self.BLACKLIST_NAMES = {
            # --- Java Types & System (DO NOT RENAME) ---
            "String", "Object", "Integer", "Boolean", "Double", "Float", "Long", "Character", "Byte", "Short", "Void", "Class",
            "System", "Math", "Thread", "Runnable", "Exception", "Throwable", "Error", "RuntimeException",
            "List", "Map", "Set", "ArrayList", "HashMap", "HashSet", "LinkedList", "Iterator", "Collections", "Arrays",
            "out", "in", "err", # System.out, System.in
            "Override", "Deprecated", "SuppressWarnings",
            
            # --- C/C++ Input/Output ---
            "main", "printf", "print", "println", "System.out.println",
            "fprintf", "sprintf", "snprintf", "wsprintf",
            "scanf", "fscanf", "sscanf", "swscanf",
            "gets", "fgets", "getchar", "getc",
            "puts", "fputs", "putchar", "putc",
            "cout", "cin", "cerr", "std::cout", "std::cin", "std::cerr",
            
            # --- String Manipulation (Common Sinks) ---
            "strcpy", "strncpy", "strcat", "strncat", 
            "memcpy", "memmove", "memset", "memcmp",
            "strlen", "wcslen",
            "strcmp", "strncmp", "strcasecmp",
            "strchr", "strrchr", "strstr",
            "strdup", "strtok",
            
            # --- Memory Management ---
            "malloc", "calloc", "realloc", "free", "alloca", "new", "delete",
            
            # --- File Operations ---
            "fopen", "fclose", "fread", "fwrite", "open", "read", "write", "close",
            "fseek", "ftell", "rewind", "fflush",
            "access", "stat", "chmod", "chown",
            
            # --- Process & Execution ---
            "system", "popen", "pclose", 
            "execl", "execlp", "execle", "execv", "execvp", "execvpe",
            "fork", "wait", "exit", "abort",
            "dlopen", "dlsym",
            
            # --- Conversions & Utilities ---
            "atoi", "atof", "atol", "atoll", "strtol", "strtoul", "strtod",
            "abs", "rand", "srand", "time",
            "getenv", "putenv", "setenv", "unsetenv",
            
            # --- Java Common Methods ---
            "equals", "length", "size", "toString", "hashCode", "clone",
            "substring", "trim", "charAt", "append",
            "parseInt", "parseFloat", "valueOf", "readLine",
            
            # --- Java SQL ---
            "executeQuery", "executeUpdate", "execute", "addBatch",
            "Connection", "Statement", "PreparedStatement", "ResultSet",
            "prepareCall", "createStatement", "prepareStatement",
            
            # --- Java IO ---
            "readObject", "writeObject", "Serializable",
            "FileInputStream", "FileOutputStream", "ObjectInputStream", "ObjectOutputStream",
            "File", "FileReader", "FileWriter", "BufferedReader", "PrintWriter",
            
            # --- Java System ---
            "Runtime", "exec", "ProcessBuilder", "start",
            
            # --- Java Web ---
            "DocumentBuilder", "DocumentBuilderFactory", "SAXParser", "SAXParserFactory",
            "HttpServletRequest", "HttpServletResponse", "getParameter", "getAttribute",
            "sendRedirect", "getWriter", "cookies", "getSession",

            # --- Juliet Utilities ---
            "printLine", "printIntLine", "printHexCharLine", "printLongLine", 
            "printUnsignedLine", "printDoubleLine", "printStructLine", "printBytesLine",
            "writeLine", "IO.writeLine", "IO.logger.log",
            
            # --- C/C++ Windows API ---
            "LoadLibrary", "LoadLibraryA", "LoadLibraryW", "FreeLibrary",
            "GetProcAddress", "GetModuleHandle",
            "sizeof", "ALLOCA",
            
            # --- C/C++ Network ---
            "socket", "connect", "bind", "listen", "accept", "recv", "send",
            "WSAStartup", "WSACleanup", "htons", "htonl", "ntohs", "ntohl",
            "inet_addr", "inet_ntoa", "gethostbyname", "closesocket", "CLOSE_SOCKET",
            
            # --- C/C++ Wide Char ---
            "wcscpy", "wcslen", "wcschr", "wcsrchr", "wcscat", "wcsncat", "wcsncpy",
            "fgetws", "wprintf", "fwprintf", "swprintf", "vswprintf",
            
            # --- Java Misc ---
            "log", "logger", "InputStreamReader", "OutputStreamWriter",
            "ByteArrayInputStream", "ByteArrayOutputStream",
            "Socket", "ServerSocket", "getInputStream", "getOutputStream",
            "Cookie", "addCookie", "getCookies", "getName", "getValue", "setValue",
            "addHeader", "setHeader", "getHeader", 
            "URLEncoder", "URLDecoder", "encode", "decode",
            "getDBConnection", "closeConnection",
            "add", "put", "get", "remove", "clear", "containsKey", "containsValue",
            "keySet", "entrySet", "iterator", "hasNext", "next",
            "Properties", "getProperty", "setProperty", "load", "store"
        }

    def extract_targets(self, root, code_string):
        """
        Extract variables, functions, strings, and numbers for renaming.
        Returns a list of tuples: (node_text, type) where type is 'VAR', 'FUNC', 'STR', 'NUM'.
        """
        targets = []
        queue = [root]
        
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            
            node_type = current_node.type
            parent_type = str(current_node.parent.type) if current_node.parent else ""

            # --- 1. Identifier (VAR vs FUNC) ---
            if node_type in ["identifier", "variable_name", "type_identifier", "field_identifier"]:
                # Check blacklist regardless of type
                name_tokens = self.tokenizer_function(code_string, current_node)
                if name_tokens:
                    name = name_tokens[0]
                    if name not in self.BLACKLIST_NAMES:
                        # Heuristic to distinguish FUNC vs VAR
                        # This varies by language grammar
                        is_func = False
                        
                        # Common Patterns for Function Calls/Declarations
                        if parent_type in ["function_declarator", "method_declaration", "function_definition"]:
                            # In declaration: "void foo(int a)" -> foo is FUNC
                            # But we need to be careful not to catch return type
                            # Usually the identifier in declarator is the name
                            is_func = True
                        elif parent_type in ["call_expression", "method_invocation", "invocation_expression"]:
                            # In call: "foo(1)" -> foo is FUNC
                            # In tree-sitter, the function name is usually the 'function' field or first child
                            # Simple check: if this node is the one being called
                             if current_node == current_node.parent.child_by_field_name("function"):
                                 is_func = True
                             # For Java method invocation: object.method() -> method is 'name' field
                             if current_node == current_node.parent.child_by_field_name("name"):
                                 is_func = True

                        if is_func:
                            targets.append((name, 'FUNC'))
                        else:
                            # Exclude Types/Classes from renaming if possible (or rename them as VAR/CLASS)
                            # For simplicity, treating remaining identifiers as VAR
                            # Exclude simple property access if needed?
                            targets.append((name, 'VAR'))

            # --- 2. String Literals (STR) ---
            elif node_type in ["string_literal", "string"]:
                str_content = code_string[current_node.start_byte:current_node.end_byte]
                if len(str_content) > 3: 
                     targets.append((str_content, 'STR'))

            # --- 3. Numbers (NUM) ---
            # Expanded list for Java/C/Python support
            elif node_type in ["number_literal", "integer_literal", "float_literal", 
                               "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal", "binary_integer_literal",
                               "decimal_floating_point_literal", "hex_floating_point_literal"]:
                num_content = code_string[current_node.start_byte:current_node.end_byte]
                # Avoid renaming simple 0, 1, -1 which are often logic flags
                if num_content not in ["0", "1", "-1", "0.0"]:
                    targets.append((num_content, 'NUM'))
                    
            for child in current_node.children:
                queue.append(child)
        return targets

    def var_renaming(self, code_string):
        root = self.parse_code(code_string)
        
        # 1. Identify targets
        targets = self.extract_targets(root, code_string)
        
        # Unique identifying to generate maps
        vars_found = list(set([t[0] for t in targets if t[1] == 'VAR']))
        funcs_found = list(set([t[0] for t in targets if t[1] == 'FUNC']))
        strs_found = list(set([t[0] for t in targets if t[1] == 'STR']))
        nums_found = list(set([t[0] for t in targets if t[1] == 'NUM']))
        
        # Create Mappings
        replacement_map = {}
        for i, v in enumerate(vars_found): replacement_map[v] = f"VAR_{i}"
        for i, f in enumerate(funcs_found): replacement_map[f] = f"FUNC_{i}"
        for i, s in enumerate(strs_found): replacement_map[s] = f"\"STR_{i}\"" # Add quotes for strings
        for i, n in enumerate(nums_found): replacement_map[n] = f"NUM_{i}" # Numbers are raw text

        if not replacement_map:
            return root, code_string, False

        # 2. Find Occurrences (Pass 2)
        replacements = [] 
        queue = [root]
        
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            
            start = current_node.start_byte
            end = current_node.end_byte
            content = code_string[start:end]
            
            # Check if this node content is in our map and matches logical type
            # (Simplification: just checking text content match for mapped items)
            # We trust extract_targets logic, so if exact content match, we replace.
            # However, we must ensure we don't replace substrings (e.g. 'var' inside 'variable')
            # But here we are iterating NODES, so 'content' is the full token.
            
            # We strictly check node types again to avoid false positives 
            # (e.g. dont replace string content inside comment node if comment node was whole)
            # But the queue contains leaf nodes too.
            
            ntype = current_node.type
            is_target_type = ntype in ["identifier", "variable_name", "type_identifier", "field_identifier", 
                                       "string_literal", "string", 
                                       "number_literal", "integer_literal", "float_literal",
                                       "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal", "binary_integer_literal",
                                       "decimal_floating_point_literal", "hex_floating_point_literal"]
            
            if is_target_type and content in replacement_map:
                replacements.append((start, end, replacement_map[content]))

            for child in current_node.children:
                queue.append(child)
                
        # 3. Apply replacements REVERSE
        # Filter duplicates (if any parent/child overlap logic existed - unlikely with tree leaves)
        # Sort reverse
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate identical ranges (just in case)
        unique_replacements = []
        last_range = -1
        for r in replacements:
            if r[0] != last_range:
                unique_replacements.append(r)
                last_range = r[0]
        
        if isinstance(code_string, str):
            code_bytes = bytearray(code_string, 'utf-8')
        else:
            code_bytes = bytearray(code_string)
            
        for start, end, new_text in unique_replacements:
             new_text_bytes = new_text.encode('utf-8')
             code_bytes[start:end] = new_text_bytes
            
        modified_code_string = code_bytes.decode('utf-8')
        
        if modified_code_string != code_string:
            modified_root = self.parse_code(modified_code_string)
            return modified_root, modified_code_string, True
        else:
            return root, code_string, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    # Complex Java Example with Function Call differentiation
    java_code = """
    public class Processor {
        public void processdata(int dataId) {
            String secret = "TOP_SECRET";
            int magic = 42;
            int count = 1000;

            magic = 42; 
            magic = 42;
            magic = 42;
            magic = 42;
            
            if (magic == 42) {
                calculate(dataId); // Function Call
                log("Processing"); // Function Call (Blacklisted)
            }
            
            helperFunc(secret, count); // Function Call
        }
        
        private void helperFunc(String s, int n) {
            System.out.println(s);
        }
        
        private void calculate(int val) {
            int result = val * 2;
        }
    }
    """

    c_code = """
    void handle_request(int req_id) {
        char *msg = "Welcome User";
        int timeout = 5000;
        timeout = 5000;
        timeout = 5000;
        timeout = 5000;
        float pi = 3.14;
        
        if (req_id > 0) {
            send_response(msg); // FUNC
            log_access(req_id); // FUNC
        }
        
        int x = compute_hash(msg, timeout); // FUNC
    }
    """
    
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
    }
    
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../..'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    
    for lang_key in ["c", "java"]:
        if lang_key not in input_map: continue
            
        lang, code = input_map[lang_key]
        print(f"\\n{'='*20} TESTING {lang.upper()} {'='*20}")
        
        var_renamer = VarRenamer(parser_path, lang)
        
        print("--- ORIGINAL ---")
        print(code)
        
        new_code, meta = var_renamer.transform_code(code)
        
        print("\\n--- TRANSFORMED (Check VAR vs FUNC vs STR vs NUM) ---")
        print(new_code)
        print("-" * 50)
