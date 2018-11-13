MODEL_DIR=run3

CUDA_VISIBLE_DEVICES=1 py main.py --model gdda_no_sie --output_dir data/$MODEL_DIR/gdda_no_sie

CUDA_VISIBLE_DEVICES=1 py main.py --model gdda_no_gef --output_dir data/$MODEL_DIR/gdda_no_gef

CUDA_VISIBLE_DEVICES=1 py main.py --model gdda --output_dir data/$MODEL_DIR/gdda

CUDA_VISIBLE_DEVICES=1 py main.py --model gdda_residual --output_dir data/$MODEL_DIR/gdda_residual
