MODEL_DIR=run7_overfit_single_frame_150_epochs
DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn --output_dir data/$MODEL_DIR/plain_cnn --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model plain_cnn_mean --output_dir data/$MODEL_DIR/plain_cnn_mean --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gda_no_gef --output_dir data/$MODEL_DIR/gda_no_gef --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gda --output_dir data/$MODEL_DIR/gda --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda_no_sie --output_dir data/$MODEL_DIR/gdda_no_sie --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda_no_gef --output_dir data/$MODEL_DIR/gdda_no_gef --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda --output_dir data/$MODEL_DIR/gdda --test_il

CUDA_VISIBLE_DEVICES=$DEVICE py main.py --model gdda_residual --output_dir data/$MODEL_DIR/gdda_residual --test_il
