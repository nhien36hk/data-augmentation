#!/bin/bash

function setup_repo() {
    mkdir -p sitter-libs;
    git clone https://github.com/tree-sitter/tree-sitter-c sitter-libs/c;
    git clone https://github.com/tree-sitter/tree-sitter-cpp sitter-libs/cpp;
    mkdir -p "parser";
    python3 create_tree_sitter_parser.py sitter-libs;
    cp parser/languages.so src/evaluator/CodeBLEU/parser/languages.so
}

function create_and_activate() {
    # Tạo env conda ngay trong thư mục dự án với Python 3.11.14
    conda create -p ./env python=3.11.14 -y
    conda activate ./env
}

function install_deps() {
    # Dùng pip để chủ động phiên bản (hợp với Python 3.11)
    pip install --upgrade pip
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
    pip install datasets==2.18.0
    pip install transformers==4.38.1
    pip install tensorboard==2.14.0
    pip install tree-sitter==0.20.4
    pip install nltk==3.8.1
    pip install scipy==1.11.4
    # Please add the command if you add any package.
}

# create_and_activate
install_deps
setup_repo;


