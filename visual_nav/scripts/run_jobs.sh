MODEL_DIR=run9_human8_fov120
DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn --output_dir data/$MODEL_DIR/plain_cnn

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn_mean --output_dir data/$MODEL_DIR/plain_cnn_mean

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gda_regressor --output_dir data/$MODEL_DIR/gda_regressor

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda_residual_regressor --output_dir data/$MODEL_DIR/gdda_residual_regressor