## Getting Started (This Fork)

Mục tiêu chính của fork này là **augment dữ liệu code C/C++ và Java**.

### Cài môi trường và parser

Từ thư mục gốc:

```bash
python3 -m venv .venv        # tuỳ chọn
source .venv/bin/activate    # tuỳ chọn

bash setup.sh                # bắt buộc, cài deps + build parser/languages.so
```

`setup.sh` sẽ tự:
- Cài các thư viện Python cần thiết.
- Clone các grammar Tree‑sitter (C, C++, Java).
- Build `parser/languages.so` dùng cho mọi script.

Nếu thiếu `parser/languages.so` thì `run_rename.py` / `run_augmentation.py` sẽ báo lỗi và yêu cầu chạy lại `bash setup.sh`.

### Chạy augment

Bạn có thể tham khảo `slurm.sh` để xem nhanh ví dụ lệnh chạy. Một số lệnh cơ bản:

```bash
# Rename C/C++
python3 run_rename.py --input_dir data/raw --output_dir data/renamed_c --mode c

# Rename Java
python3 run_rename.py --input_dir data/raw --output_dir data/renamed_java --mode java

# Augment nhiều kiểu (ví dụ cho C/C++)
python3 run_augmentation.py --input_dir data/raw --output_dir data/augmented_c --mode c
```

Các script làm việc với các file `.jsonl` trong `--input_dir`, mỗi dòng tối thiểu nên có:
- `file`: tên file code gốc (dùng đoán C/C++/Java).
- `func`: nội dung hàm/code cần augment.
