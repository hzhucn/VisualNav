MODEL_DIR=run2

CUDA_VISIBLE_DEVICES=0 py main.py --model plain_cnn --output_dir data/$MODEL_DIR/plain_cnn

CUDA_VISIBLE_DEVICES=0 py main.py --model plain_cnn_mean --output_dir data/$MODEL_DIR/plain_cnn_mean

CUDA_VISIBLE_DEVICES=0 py main.py --model gda_no_gef --output_dir data/$MODEL_DIR/gda_no_gef

CUDA_VISIBLE_DEVICES=0 py main.py --model gda --output_dir data/$MODEL_DIR/gda

CUDA_VISIBLE_DEVICES=0 py main.py --model gdda_no_sie --output_dir data/$MODEL_DIR/gdda_no_sie

CUDA_VISIBLE_DEVICES=0 py main.py --model gdda_no_gef --output_dir data/$MODEL_DIR/gdda_no_gef

CUDA_VISIBLE_DEVICES=0 py main.py --model gdda --output_dir data/$MODEL_DIR/gdda

CUDA_VISIBLE_DEVICES=0 py main.py --model gdda_residual --output_dir data/$MODEL_DIR/gdda_residual
