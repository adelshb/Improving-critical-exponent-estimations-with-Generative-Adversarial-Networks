python src/cnn/train.py  \
    --save_dir './saved_models/cnn_regression' \
    --lattice_size 128 \
    --dataset_size 5000 \
    --epochs 1000 \
    --batch_size 64 \
    --dropout_rate 0 \
    --learning_rate 10e-4 