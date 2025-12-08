nohup env PYTHONPATH=/datastore/inseclab/HuuNhien/data-augmentation \
  /datastore/inseclab/HuuNhien/shared_env/bin/python \
  /datastore/inseclab/HuuNhien/data-augmentation/src/pretraining/prepare_data.py \
  --input_dir data/augmentation \
  --output_dir data/augmentation/output \
  --parser_path parser/languages.so \
  --workers 20 \
  --max_function_length 2000 \
  --num_variants 5 \
  > augmentation.log 2>&1 &

# Collect a small sample of functions whose original code fails tree-sitter parsing (for analysis)
# Adjust --limit and --input if needed.
nohup env PYTHONPATH=/datastore/inseclab/HuuNhien/data-augmentation \
  /datastore/inseclab/HuuNhien/shared_env/bin/python \
  /datastore/inseclab/HuuNhien/data-augmentation/scripts/extract_parse_error_samples.py \
  --input data/augmentation/train.json \
  --parser_path parser/languages.so \
  --output data/augmentation/output/parse_error_samples.jsonl \
  --limit 30 \
  > parse_errors.log 2>&1 &