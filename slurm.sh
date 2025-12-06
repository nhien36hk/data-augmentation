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