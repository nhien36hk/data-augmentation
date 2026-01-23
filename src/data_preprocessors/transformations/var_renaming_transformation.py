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
import os

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
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        # Remove function definitions from blacklist to allow renaming them
        # Kept: "class_declaration", "call" (to avoid renaming external library calls like printf)
        self.not_var_ptype = ["class_declaration", "call"]

    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]
        
        # Simple whitelist for system/library functions to avoid renaming
        # This prevents 'printf', 'print', 'main' from becoming VAR_X
        # while still allowing 'badSink', 'goodG2B' to be renamed.
        # Comprehensive whitelist for system/library functions to avoid renaming.
        # Keeping these names is crucial for ML models to learn vulnerability patterns 
        # (e.g., 'gets' and 'strcpy' are strong indicators of Buffer Overflow).
        BLACKLIST_NAMES = {
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
            
            # --- Process & Execution (Command Injection Sinks) ---
            "system", "popen", "pclose", 
            "execl", "execlp", "execle", "execv", "execvp", "execvpe",
            "fork", "wait", "exit", "abort",
            "dlopen", "dlsym",
            
            # --- Conversions & Utilities ---
            "atoi", "atof", "atol", "atoll", "strtol", "strtoul", "strtod",
            "abs", "rand", "srand", "time",
            "getenv", "putenv", "setenv", "unsetenv",
            
            # --- Java Common Methods & Security Sinks ---
            "equals", "length", "size", "toString", "hashCode", "clone",
            "substring", "trim", "charAt", "append",
            "parseInt", "parseFloat", "valueOf", "readLine",
            
            # --- Java SQL (Injection Sinks) ---
            "executeQuery", "executeUpdate", "execute", "addBatch",
            "Connection", "Statement", "PreparedStatement", "ResultSet",
            "prepareCall", "createStatement", "prepareStatement",
            
            # --- Java IO & Serialization ---
            "readObject", "writeObject", "Serializable",
            "FileInputStream", "FileOutputStream", "ObjectInputStream", "ObjectOutputStream",
            "File", "FileReader", "FileWriter", "BufferedReader", "PrintWriter",
            
            # --- Java Command Injection ---
            "Runtime", "exec", "ProcessBuilder", "start",
            
            # --- Juliet Test Suite Utilities ---
            "printLine", "printIntLine", "printHexCharLine", "printLongLine", 
            "printUnsignedLine", "printDoubleLine", "printStructLine", "printBytesLine",
            "writeLine", "IO.writeLine", "IO.logger.log",
            
            # --- C/C++ Windows API & Dynamic Loading ---
            "LoadLibrary", "LoadLibraryA", "LoadLibraryW", "FreeLibrary",
            "GetProcAddress", "GetModuleHandle",
            "sizeof", "ALLOCA",
            
            # --- C/C++ Network (Winsock/BSD) ---
            "socket", "connect", "bind", "listen", "accept", "recv", "send",
            "WSAStartup", "WSACleanup", "htons", "htonl", "ntohs", "ntohl",
            "inet_addr", "inet_ntoa", "gethostbyname", "closesocket", "CLOSE_SOCKET",
            
            # --- C/C++ Wide Character Strings ---
            "wcscpy", "wcslen", "wcschr", "wcsrchr", "wcscat", "wcsncat", "wcsncpy",
            "fgetws", "wprintf", "fwprintf", "swprintf", "vswprintf",
            
            # --- Java Logging & IO ---
            "log", "logger", "InputStreamReader", "OutputStreamWriter",
            "ByteArrayInputStream", "ByteArrayOutputStream",
            "Socket", "ServerSocket", "getInputStream", "getOutputStream",
            
            # --- Java Web/Cookie (Session & XSS related) ---
            "Cookie", "addCookie", "getCookies", "getName", "getValue", "setValue",
            "addHeader", "setHeader", "getHeader", 
            "URLEncoder", "URLDecoder", "encode", "decode",
            
            # --- Java Database ---
            "getDBConnection", "closeConnection",
            
            # --- Java Collections & Properties ---
            "add", "put", "get", "remove", "clear", "containsKey", "containsValue",
            "keySet", "entrySet", "iterator", "hasNext", "next",
            "Properties", "getProperty", "setProperty", "load", "store"
        }

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            
            is_identifier = (current_node.type == "identifier" or current_node.type == "variable_name")
            if is_identifier and str(current_node.parent.type) not in self.not_var_ptype:
                name = self.tokenizer_function(code_string, current_node)[0]
                # Filter out system functions
                if name not in BLACKLIST_NAMES:
                    var_names.append(name)
                    
            for child in current_node.children:
                queue.append(child)
        return var_names

    def var_renaming(self, code_string):
        root = self.parse_code(code_string)
        
        # 1. Identify WHICH variables to rename
        var_names_all = self.extract_var_names(root, code_string)
        var_names_unique = list(set(var_names_all))
        
        # Rename logic (currently 100%)
        num_to_rename = len(var_names_unique)
        # random.shuffle(var_names_unique) # Optional: shuffle if we weren't renaming all
        target_vars = var_names_unique[:num_to_rename]
        
        var_map = {}
        for idx, v in enumerate(target_vars):
            var_map[v] = f"VAR_{idx}"
            
        if not var_map:
            return root, code_string, False

        # 2. Find ALL occurrences (Nodes) of these variables in the tree
        # We traverse again to get exact byte ranges.
        # Note: We re-use logic from extract_var_names but keep Node info.
        
        replacements = [] # List of (start_byte, end_byte, new_text)
        queue = [root]
        
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            
            is_identifier = (current_node.type == "identifier" or current_node.type == "variable_name")
            
            # Check if this node is a candidate for renaming
            if is_identifier and str(current_node.parent.type) not in self.not_var_ptype:
                # Get the actual text of this node
                # Note: self.tokenizer_function returns a list of tokens, we take [0]
                # But cleaner is to slice the code_string directly if possible, or trust tokenizer.
                # using tokenizer matches extract_var_names logic.
                name_tokens = self.tokenizer_function(code_string, current_node)
                if name_tokens:
                    name = name_tokens[0]
                    # If this name is in our target renaming map, record it
                    if name in var_map:
                        replacements.append((current_node.start_byte, current_node.end_byte, var_map[name]))
            
            for child in current_node.children:
                queue.append(child)
                
        # 3. Apply replacements in REVERSE order (so offsets don't shift)
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Use bytearray for efficient mutable editing
        if isinstance(code_string, str):
            code_bytes = bytearray(code_string, 'utf-8')
        else:
            code_bytes = bytearray(code_string)
            
        for start, end, new_text in replacements:
            # Replace slice
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
        # Removed aggressive whitespace flattening to preserve newlines
        # code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    # Complex Java Example: Generics, Try-Catch, Annotations, Inner Class
    java_code = """
    @Override
    public class ComplexProcessor<T> {
        private static final int MAX_RETRIES = 3;
        
        public void process(List<String> items, Map<String, Object> config) {
            int attempts = 0;
            // Try-catch block with resources
            try (BufferedReader reader = new BufferedReader(new FileReader("config.txt"))) {
                String line = reader.readLine();
                if (line != null) {
                    System.out.println("Processing: " + line); // Whitelisted system call
                }
                
                for (String item : items) {
                    int len = item.length(); // Whitelisted method
                    if (len > 10) {
                        log("Item too long"); // Whitelisted log
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                cleanUp();
            }
        }
        
        private void cleanUp() {
            printLine("Cleanup done"); // Whitelisted Juliet utility
        }
    }
    """

    # Complex C Example: Structs, Pointers, Macros, Comments, Buffer Ops
    c_code = """
    #define BUFFER_SIZE 256
    
    typedef struct {
        int id;
        char name[50];
        double value;
    } Item;

    void process_data(Item *items, int count) {
        char *buffer = (char *)malloc(BUFFER_SIZE * sizeof(char)); // Whitelisted malloc, sizeof
        
        /* Multi-line comment 
           Check for null pointer */
        if (buffer == NULL) {
            printf("Memory error\\n"); // Whitelisted printf
            return;
        }

        memset(buffer, 0, BUFFER_SIZE); // Whitelisted memset
        
        for (int i = 0; i < count; i++) {
            Item *current = &items[i]; // Pointer arithmetic
            
            // Complex expression
            if (current->id > 100 && current->value < 0.5) {
                snprintf(buffer, BUFFER_SIZE, "Item %s", current->name); // Whitelisted snprintf
                printLine(buffer); // Whitelisted Juliet utility
            }
        }
        
        free(buffer); // Whitelisted free
    }
    """
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
    }
    
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../..'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    
    # Only run for C and Java as requested
    for lang_key in ["c", "java"]:
        if lang_key not in input_map: continue
            
        lang, code = input_map[lang_key]
        print(f"\n{'='*20} TESTING {lang.upper()} {'='*20}")
        
        var_renamer = VarRenamer(parser_path, lang)
        
        print("--- ORIGINAL ---")
        print(code)
        
        # Transform
        new_code, meta = var_renamer.transform_code(code)
        
        print("\n--- TRANSFORMED (Check whitespace preservation) ---")
        print(new_code)
        print("-" * 50)

