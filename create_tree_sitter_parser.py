from tree_sitter import Language
import os
import sys


lib_dir = sys.argv[1]
libs = [os.path.join(lib_dir, d) for d in os.listdir(lib_dir)]

# Ensure C/C++ scanners are compiled with C11 so `static_assert` is available.
for key in ["CFLAGS", "CPPFLAGS", "CXXFLAGS"]:
    existing = os.environ.get(key, "")
    os.environ[key] = (existing + " -std=c11").strip()

# Patch tree-sitter-cpp scanner if static_assert is missing (older gcc default modes)
for path in libs:
    scanner = os.path.join(path, "src", "scanner.c")
    if os.path.exists(scanner):
        with open(scanner, "r", encoding="utf-8") as f:
            content = f.read()
        patch = "#ifndef static_assert\n#define static_assert _Static_assert\n#endif\n"
        if "static_assert _Static_assert" not in content:
            # Insert after includes (if any), else at top
            new_content = patch + content
            with open(scanner, "w", encoding="utf-8") as f:
                f.write(new_content)

Language.build_library(
    # Store the library in the `build` directory
    'parser/languages.so',
    libs,                                                                                                                           
)
                                                                                                                                                                                         