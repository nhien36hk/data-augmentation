#!/bin/bash

# --- 1. Cài đặt thư viện Python ---
function install_deps() {
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    # Tree-sitter 0.20.4 ổn định với các grammar cũ
    pip install tree-sitter==0.20.4
    pip install numpy
    pip install tqdm
    # NLTK cần thiết cho các tác vụ tokenizer
    pip install nltk==3.8.1
}

# --- 2. Clone Grammar Sources ---
function setup_repo() {
    echo "Setting up Tree-sitter grammars..."
    mkdir -p sitter-libs
    
    # Clone C Grammar (Version tương thích)
    if [ ! -d "sitter-libs/c" ]; then
        echo "Cloning tree-sitter-c..."
        git clone https://github.com/tree-sitter/tree-sitter-c sitter-libs/c
        # Checkout về phiên bản ổn định
        cd sitter-libs/c && git checkout v0.20.6 && cd ../..
    else
        echo "tree-sitter-c already exists."
    fi

    # Clone CPP Grammar (Version tương thích)
    if [ ! -d "sitter-libs/cpp" ]; then
        echo "Cloning tree-sitter-cpp..."
        git clone https://github.com/tree-sitter/tree-sitter-cpp sitter-libs/cpp
        # Checkout về phiên bản ổn định
        cd sitter-libs/cpp && git checkout v0.20.0 && cd ../..
    else
        echo "tree-sitter-cpp already exists."
    fi

    # Clone Java Grammar
    if [ ! -d "sitter-libs/java" ]; then
        echo "Cloning tree-sitter-java..."
        git clone https://github.com/tree-sitter/tree-sitter-java sitter-libs/java
        # Checkout về phiên bản ổn định (v0.20.1 tương thích tốt với tree-sitter 0.20.x)
        cd sitter-libs/java && git checkout v0.20.1 && cd ../..
    else
        echo "tree-sitter-java already exists."
    fi

    # --- 3. Build Parser ---
    echo "Building parser/languages.so..."
    # Xóa parser cũ nếu có để đảm bảo build sạch
    rm -rf parser && mkdir -p parser
    
    # Chạy script build (đã được viết lại để dùng gcc/g++ trực tiếp)
    python3 create_tree_sitter_parser.py
    
    # Kiểm tra kết quả
    if [ -f "parser/languages.so" ]; then
        echo "Build SUCCESS: parser/languages.so created."
    else
        echo "Build FAILED."
        exit 1
    fi
}

# --- Main Execution Flow ---
install_deps
setup_repo
