nohup env PYTHONPATH=/datastore/inseclab/HuuNhien/data-augmentation \
  /datastore/inseclab/HuuNhien/shared_env/bin/python \
  /datastore/inseclab/HuuNhien/data-augmentation/src/pretraining/prepare_data.py \
  --input_dir data/augmentation \
  --output_dir data/augmentation/output-clean \
  --parser_path parser/languages.so \
  --workers 20 \
  --max_function_length 4000 \
  > augmentation-clean.log 2>&1 &

# Devign
nohup env PYTHONPATH=/datastore/inseclab/HuuNhien/data-augmentation \
  /datastore/inseclab/HuuNhien/shared_env/bin/python \
  /datastore/inseclab/HuuNhien/data-augmentation/src/pretraining/prepare_devign.py \
  --input_dir data/augmentation/input-101-easy \
  --output_dir data/augmentation/output-devign \
  --parser_path parser/languages.so \
  --max_function_length 4000 \
  > augmentation-devign.log 2>&1 &


ps -ef | grep prepare_data.py