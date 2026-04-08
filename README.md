# NatGen: Generative Pre-training by "Naturalizing" Source Code.
Saikat Chakraborty, Toufique Ahmed, Yangruibo Ding, Premkumar T Devanbu, Baishakhi Ray. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE ’22), November 14-18, 2022, Singapore, Singapore. ACM, New York, NY, USA, 13 pages. [https://doi.org/10.1145/3540250.3549162](https://doi.org/10.1145/3540250.3549162).

<br/>

<p align="center">
  <a href="https://github.com/saikat107/NatGen/issues-raw">
    <img src="https://img.shields.io/github/issues-raw/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/issues-closed-raw">
    <img src="https://img.shields.io/github/issues-closed-raw/saikat107/NatGen" /> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/issues-pr-raw">
    <img src="https://img.shields.io/github/issues-pr-raw/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/issues-pr-closed-raw">
    <img src="https://img.shields.io/github/issues-pr-closed-raw/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/network/members">
    <img src="https://img.shields.io/github/forks/saikat107/NatGen"/> 
  </a>  
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/stargazers">
    <img src="https://img.shields.io/github/stars/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/LICENSE">
    <img src="https://img.shields.io/github/license/saikat107/NatGen"/> 
  </a> 
  &nbsp;
  <img src="https://img.shields.io/github/languages/count/saikat107/NatGen"/>
  &nbsp;
  <img src="https://img.shields.io/github/languages/top/saikat107/NatGen"/>
  &nbsp;
  <img src="https://img.shields.io/github/last-commit/saikat107/NatGen"/>
</p>

### <p align="center">[The paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549162) &emsp; [Slide Deck](https://docs.google.com/presentation/d/1T6kjiohAAR1YvcNvTASR94HptA3xHGCl/edit?usp=sharing&ouid=111755026725574085503&rtpof=true&sd=true)</p>

## Getting Started (This Fork)

This fork focuses on **code data augmentation** for C/C++ and Java datasets using Tree‑sitter.

### Environment Requirements

- Python 3.8+
- Git
- A C/C++ toolchain (for building Tree‑sitter parsers), e.g. `gcc`, `g++`, `make`

All required Python packages and Tree‑sitter grammars are installed for you by `setup.sh`.

### 1. Setup environment and Tree‑sitter parsers

Từ thư mục gốc của project:

```bash
# (tuỳ chọn) tạo virtualenv
python3 -m venv .venv
source .venv/bin/activate

# cài thư viện Python + build parser/languages.so
bash setup.sh
```

`setup.sh` sẽ:

- Cài các thư viện Python cơ bản: `tree-sitter`, `numpy`, `tqdm`, `nltk`, …
- Clone các grammar Tree‑sitter cho C, C++ và Java vào thư mục `sitter-libs/`
- Build file parser `parser/languages.so` dùng chung cho các script augment

Nếu `parser/languages.so` chưa tồn tại, cả `run_rename.py` và `run_augmentation.py` sẽ báo lỗi và yêu cầu chạy `bash setup.sh` trước.

### 2. Chuẩn bị dữ liệu đầu vào

Các script augmentation làm việc với **các file `.jsonl`** trong một thư mục đầu vào.

- Mỗi dòng là một JSON object, ví dụ:

```json
{"file": "example.c", "func": "int add(int a, int b) { return a + b; }"}
```

- Trường tối thiểu cần có:
  - `"file"`: tên file gốc (dùng để đoán ngôn ngữ C/C++/Java)
  - `"func"`: nội dung code nguồn cần augment

Giả sử bạn đặt các file vào thư mục `data/raw/`, ví dụ:

- `data/raw/train_c.jsonl`
- `data/raw/train_java.jsonl`

### 3. Chạy `run_rename.py` – Variable Renaming

`run_rename.py` áp dụng **biến đổi đổi tên biến (variable renaming)** trên từng mẫu code.

Usage cơ bản:

```bash
python run_rename.py \
  --input_dir data/raw \
  --output_dir data/renamed \
  --mode c

python run_rename.py \
  --input_dir data/raw \
  --output_dir data/renamed \
  --mode java
```

Trong đó:

- `--input_dir`: thư mục chứa các file `.jsonl` gốc
- `--output_dir`: thư mục để lưu kết quả (sẽ được tạo nếu chưa tồn tại)
- `--mode`: `c` (auto detect `c` / `cpp` theo phần mở rộng) hoặc `java`
- `--workers` (tuỳ chọn): số process chạy song song (mặc định = số CPU logic)

Logic hoạt động chính (`run_rename.py`):

- Đọc toàn bộ các dòng JSON từ từng file trong `input_dir`.
- Với mỗi dòng:
  - Parse JSON, lấy trường `"func"` và `"file"`.
  - Xác định ngôn ngữ Tree‑sitter (`c` / `cpp` / `java`) dựa trên `--mode` và đuôi file.
  - Gọi `VarRenamer` để thực hiện biến đổi tên biến.
  - Nếu thành công, cập nhật:
    - `"func"`: code đã được rename
    - Thêm metadata `"renamed": true`, `"transformation": "var_renaming"`.
- Ghi kết quả ra file mới trong `output_dir` với hậu tố `_normalize`, ví dụ:
  - `train_c_normalize.jsonl`

Kết quả: dataset có cùng số dòng, nhưng nhiều đoạn code đã được đổi tên biến (semantically equivalent).

### 4. Chạy `run_augmentation.py` – Nhiều kiểu biến đổi ngẫu nhiên

`run_augmentation.py` là script **augmentation tổng hợp**, thử áp dụng **ngẫu nhiên 1 trong 5 transformation** cho mỗi sample:

- `BlockSwap`
- `ConfusionRemover`
- `DeadCodeInserter`
- `ForWhileTransformer`
- `OperandSwap`

Usage:

```bash
python run_augmentation.py \
  --input_dir data/raw \
  --output_dir data/augmented \
  --mode c

python run_augmentation.py \
  --input_dir data/raw \
  --output_dir data/augmented \
  --mode java
```

Các tham số:

- `--input_dir`: thư mục chứa các file `.jsonl` gốc
- `--output_dir`: thư mục để lưu dataset đã augment
- `--mode`: `c` hoặc `java`
- `--workers`: số process chạy song song (mặc định = số CPU logic)

Flow chính (`run_augmentation.py`):

1. Đọc tất cả các file `.jsonl` trong `input_dir`.
2. Lọc file theo `--mode`:
   - `mode=java`: chỉ giữ file có chữ `"java"` trong tên.
   - `mode=c`: bỏ qua file có `"java"` trong tên (coi là dataset Java).
3. Với mỗi dòng JSON:
   - Parse JSON, lấy `"func"` và `"file"`, xác định ngôn ngữ Tree‑sitter (`c`/`cpp`/`java`).
   - Shuffle danh sách 5 transformer ở trên.
   - Thử lần lượt từng transformer đến khi có 1 transformer:
     - `meta["success"] == True`
     - và code mới khác code cũ.
   - Nếu thành công:
     - Ghi lại:
       - `"func"`: code đã được augment
       - `"augmented": true`
       - `"transformation_used": <Tên transformer>`
   - Nếu tất cả transformer thất bại: giữ nguyên dòng gốc (không set cờ `augmented`).
4. Ghi kết quả ra file mới trong `output_dir` với hậu tố `_augmented`, ví dụ:
   - `train_c_augmented.jsonl`

Script cũng in **tỉ lệ augment thành công** trên toàn bộ mẫu đã xử lý:

- `Augmentation Success Rate: <augmented_count>/<total> (<percent>%)`

### 5. Gợi ý workflow đơn giản

1. Chuẩn bị dữ liệu JSONL gốc vào `data/raw/`.
2. Chạy `bash setup.sh` đúng 1 lần để build `parser/languages.so`.
3. (Tuỳ chọn) Chạy `run_rename.py` để normalize / rename biến:

   ```bash
   python run_rename.py --input_dir data/raw --output_dir data/renamed --mode c
   ```

4. Chạy `run_augmentation.py` trên dataset gốc hoặc đã rename để sinh thêm biến thể:

   ```bash
   python run_augmentation.py --input_dir data/renamed --output_dir data/augmented --mode c
   ```

5. Sử dụng các file trong `data/augmented/` cho huấn luyện / phân tích tiếp theo.

# Citation
If you use  this repository, please cite,
```
@inproceedings{chakraborty2022natgen,
    author = {Chakraborty, Saikat and Ahmed, Toufique and Ding, Yangruibo and Devanbu, Premkumar T. and Ray, Baishakhi},
    title = {NatGen: Generative Pre-Training by “Naturalizing” Source Code},
    year = {2022},
    isbn = {9781450394130},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3540250.3549162},
    doi = {10.1145/3540250.3549162},
    booktitle = {Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
    pages = {18–30},
    numpages = {13},
    keywords = {Neural Network, Semantic Preserving Transformation, Source Code Transformer, Source Code Pre-training},
    location = {Singapore, Singapore},
    series = {ESEC/FSE 2022}
}
```
