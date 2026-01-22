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
        original_code = self.tokenizer_function(code_string, root)
        # print(" ".join(original_code))
        var_names = self.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        # TODO: change to 20%
        # num_to_rename = math.ceil(0.2 * len(var_names))
        num_to_rename = len(var_names)
        random.shuffle(var_names)
        var_names = var_names[:num_to_rename]
        var_map = {}
        for idx, v in enumerate(var_names):
            var_map[v] = f"VAR_{idx}"
        modified_code = []
        for t in original_code:
            if t in var_names:
                modified_code.append(var_map[t])
            else:
                modified_code.append(t)

        modified_code_string = " ".join(modified_code)
        if modified_code != original_code:
            modified_root = self.parse_code(modified_code_string)
            return modified_root, modified_code_string, True
        else:
            return root, code_string, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                executeQuery("hello");
                print("hello");
                System.out.println("hello");
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    }
    """
    python_code = """def foo(n):
    res = 0
    for i in range(0, 19, 2):
        res += i
    i = 0
    while i in range(n):
        res += i
        i += 1
    return res
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
    cs_code = """
    int foo(int n){
            int res = 0, i = 0;
            while(i < n) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    """
    js_code = """function foo(n) {
        let res = '';
        for(let i = 0; i < 10; i++){
            res += i.toString();
            res += '<br>';
        } 
        while ( i < 10 ; ) { 
            res += 'bk'; 
        }
        return res;
    }
    """
    ruby_code = """
        for i in 0..5 do
           puts "Value of local variable is #{i}"
           if false then
                puts "False printed"
                while i == 10 do
                    print i;
                end
                i = u + 8
            end
        end
        """
    go_code = """
        func main() {
            sum := 0;
            i := 0;
            for ; i < 10;  {
                sum += i;
            }
            i++;
            fmt.Println(sum);
        }
        """
    php_code = """
    <?php 
    for ($x = 0; $x <= 10; $x++) {
        echo "The number is: $x <br>";
    }
    $x = 0 ; 
    while ( $x <= 10 ) { 
        echo "The number is:  $x  <br> "; 
        $x++; 
    } 
    ?> 
    """
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
        "cs": ("c_sharp", cs_code),
        "js": ("javascript", js_code),
        "python": ("python", python_code),
        "php": ("php", php_code),
        "ruby": ("ruby", ruby_code),
        "go": ("go", go_code),
    }
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../..'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    # Only run for C, CPP, and Java as requested
    for lang in ["c", "cpp", "java"]:
        lang, code = input_map[lang]
        var_renamer = VarRenamer(
            parser_path, lang
        )
        print(lang)
        code, meta = var_renamer.transform_code(code)
        print(re.sub("[ \t\n]+", " ", code))
        print(meta)
        print("=" * 150)
