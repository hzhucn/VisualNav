MODEL_DIR=run3

CUDA_VISIBLE_DEVICES=0 py main.py --model plain_cnn --output_dir data/$MODEL_DIR/plain_cnn

CUDA_VISIBLE_DEVICES=0 py main.py --model plain_cnn_mean --output_dir data/$MODEL_DIR/plain_cnn_mean

CUDA_VISIBLE_DEVICES=0 py main.py --model gda_no_gef --output_dir data/$MODEL_DIR/gda_no_gef

CUDA_VISIBLE_DEVICES=0 py main.py --model gda --output_dir data/$MODEL_DIR/gda

