
import os
import sys
import subprocess
import glob

def compile_parser():
    # 1. Định nghĩa đường dẫn
    cwd = os.getcwd()
    lib_dir = "sitter-libs"
    output_dir = "parser"
    output_file = os.path.join(output_dir, "languages.so")
    
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Danh sách các file cần compile
    # Cấu trúc: (file_path, compiler_command, include_dir)
    tasks = []

    # Tree-sitter C
    c_src = os.path.join(lib_dir, "c", "src")
    tasks.append({
        "src": os.path.join(c_src, "parser.c"),
        "compiler": "gcc",
        "flags": ["-std=c99", "-fPIC", "-I", c_src, "-c"],
        "out": os.path.join(output_dir, "c_parser.o")
    })

    # Tree-sitter CPP
    cpp_src = os.path.join(lib_dir, "cpp", "src")
    tasks.append({
        "src": os.path.join(cpp_src, "parser.c"),
        # Lưu ý: parser.c của cpp vẫn là C code, dùng gcc
        "compiler": "gcc",
        "flags": ["-std=c99", "-fPIC", "-I", cpp_src, "-c"],
        "out": os.path.join(output_dir, "cpp_parser.o")
    })
    
    scanner_cc = os.path.join(cpp_src, "scanner.cc")
    if os.path.exists(scanner_cc):
         tasks.append({
            "src": scanner_cc,
            "compiler": "g++",
            "flags": ["-fPIC", "-I", cpp_src, "-c"],
            "out": os.path.join(output_dir, "cpp_scanner.o")
        })

    # Tree-sitter Java
    java_src = os.path.join(lib_dir, "java", "src")
    tasks.append({
        "src": os.path.join(java_src, "parser.c"),
        "compiler": "gcc",
        "flags": ["-std=c99", "-fPIC", "-I", java_src, "-c"],
        "out": os.path.join(output_dir, "java_parser.o")
    })


    # 3. Thực hiện compile từng file
    obj_files = []
    print("Building object files...")
    for task in tasks:
        cmd = [task["compiler"]] + task["flags"] + [task["src"], "-o", task["out"]]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            obj_files.append(task["out"])
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {task['src']}: {e}")
            sys.exit(1)

    # 4. Link tất cả thành languages.so
    print("Linking shared library...")
    link_cmd = ["g++", "-shared"] + obj_files + ["-o", output_file]
    print(f"Running: {' '.join(link_cmd)}")
    try:
        subprocess.check_call(link_cmd)
        print(f"Successfully created: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error linking: {e}")
        sys.exit(1)
        
    # 5. Dọn dẹp file .o
    print("Cleaning up...")
    for f in obj_files:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    compile_parser()