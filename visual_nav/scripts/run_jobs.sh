MODEL_DIR=run11_human8_regression
DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn --output_dir data/$MODEL_DIR/plain_cnn --il_training regression --num_epochs 50

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn_mean --output_dir data/$MODEL_DIR/plain_cnn_mean --il_training regression --num_epochs 50

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gda_regressor --output_dir data/$MODEL_DIR/gda_regressor --il_training regression --num_epochs 50

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda_residual_regressor --output_dir data/$MODEL_DIR/gdda_residual_regressor --il_training regression --num_epochs 50