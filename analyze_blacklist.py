import json
import re
from collections import Counter
from pathlib import Path

# Current Blacklist provided by user
BLACKLIST = {
    "main", "printf", "print", "println", "System.out.println",
    "fprintf", "sprintf", "snprintf", "wsprintf",
    "scanf", "fscanf", "sscanf", "swscanf",
    "gets", "fgets", "getchar", "getc",
    "puts", "fputs", "putchar", "putc",
    "cout", "cin", "cerr", "std::cout", "std::cin", "std::cerr",
    "strcpy", "strncpy", "strcat", "strncat", 
    "memcpy", "memmove", "memset", "memcmp",
    "strlen", "wcslen",
    "strcmp", "strncmp", "strcasecmp",
    "strchr", "strrchr", "strstr",
    "strdup", "strtok",
    "malloc", "calloc", "realloc", "free", "alloca", "new", "delete",
    "fopen", "fclose", "fread", "fwrite", "open", "read", "write", "close",
    "fseek", "ftell", "rewind", "fflush",
    "access", "stat", "chmod", "chown",
    "system", "popen", "pclose", 
    "execl", "execlp", "execle", "execv", "execvp", "execvpe",
    "fork", "wait", "exit", "abort",
    "dlopen", "dlsym",
    "atoi", "atof", "atol", "atoll", "strtol", "strtoul", "strtod",
    "abs", "rand", "srand", "time",
    "getenv", "putenv", "setenv", "unsetenv",
    "equals", "length", "size", "toString", "hashCode", "clone",
    "substring", "trim", "charAt", "append",
    "parseInt", "parseFloat", "valueOf", "readLine",
    "executeQuery", "executeUpdate", "execute", "addBatch",
    "Connection", "Statement", "PreparedStatement", "ResultSet",
    "prepareCall", "createStatement", "prepareStatement",
    "readObject", "writeObject", "Serializable",
    "FileInputStream", "FileOutputStream", "ObjectInputStream", "ObjectOutputStream",
    "File", "FileReader", "FileWriter", "BufferedReader", "PrintWriter",
    "Runtime", "exec", "ProcessBuilder", "start",
    "DocumentBuilder", "DocumentBuilderFactory", "SAXParser", "SAXParserFactory",
    "HttpServletRequest", "HttpServletResponse", "getParameter", "getAttribute",
    "sendRedirect", "getWriter", "cookies", "getSession"
}

def analyze_file(filepath, lines_to_read=5000):
    print(f"\nAnalyzing {filepath}...")
    func_calls = Counter()
    
    # Regex basic để bắt tên hàm: word theo sau bởi (
    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= lines_to_read: break
            try:
                data = json.loads(line)
                code = data.get('func', '') or data.get('code', '')
                
                # Tìm tất cả function call trong code
                matches = pattern.findall(code)
                for m in matches:
                    # Ignore Control structures
                    if m in ['if', 'while', 'for', 'switch', 'catch', 'synchronized']: continue
                    # Ignore User Defined functions (typically start with CWE or bad/good)
                    if m.startswith('CWE') or m.startswith('bad') or m.startswith('good') or 'G2B' in m or 'B2G' in m: continue
                    # Ignore types (cast)
                    if m in ['static', 'void', 'int', 'char', 'float', 'double', 'long']: continue
                    
                    if m not in BLACKLIST:
                        func_calls[m] += 1
            except: pass

    # In ra Top 50 hàm chưa có trong blacklist
    print("Top 50 functions detected NOT in blacklist:")
    for func, count in func_calls.most_common(50):
        print(f"{func}: {count}")

analyze_file('data/raw/juliet_dataset_c.jsonl')
analyze_file('data/raw/juliet_dataset_java.jsonl')
