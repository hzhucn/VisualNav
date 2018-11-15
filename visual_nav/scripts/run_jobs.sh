MODEL_DIR=run10_human4_use_best_wts
DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn --output_dir data/$MODEL_DIR/plain_cnn --num_epochs 200

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn_mean --output_dir data/$MODEL_DIR/plain_cnn_mean --num_epochs 200

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gda_regressor --output_dir data/$MODEL_DIR/gda_regressor --num_epochs 200

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda_residual_regressor --output_dir data/$MODEL_DIR/gdda_residual_regressor --num_epochs 200