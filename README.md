# Improving critical exponent estimations with Generative Adversarial Networks

## Requirements

* Python 3.8+
* Tensorflow
* sklearn
* scipy
* tqdm
* jupyter
* pandas

```shell
python setup.py install
```

## Generate and save configurations.

It is possible to generate a chosen number of configurations for a specific lattice size and for chosen control parameters.

```shell
python src/statphy/data_factory.py      \
    --model square_lattice_percolation  \
    --L 32 128                          \
    --control_parameter 0.52 0.6        \
    --samples 100                       \
    --path "."
```

## Control parameter estimation via CNN regression

### Train the CNN 

The trained model for control parameter estimation with CNN (lattice size: 128x128) is available [here](https://drive.google.com/file/d/1T1a5Z00auQUzX4hayw4rnxqXKjjdwoPT/view?usp=sharing).

It possible to train a CNN with the following command:

 ```shell
python src/cnn/train.py  \
    --save_dir './saved_models/cnn_regression' \
    --lattice_size 128 \
    --dataset_size 5000 \
    --epochs 1000 \
    --batch_size 64 \
    --dropout_rate 0 \
    --learning_rate 10e-4 
 ```

## Data augmentation via GAN with HYDRA

### Train the GAN

Make sure that configurations have been created first (at "/data/L_128/p_0.5928" in the following example) and the CNN is located at "./saved_models/cnn/saved-model.h5". 

```shell
python src/hydra/main.py  \
    --noise_dim 100 \
    --noise_mean 0.0 \
    --noise_std 1.0 \
    --batch_size 256 \
    --epochs 100 \
    --reg_coeff 1.0 \
    --lr 1e-3 \
    --CNN_model_path "./saved_models/cnn/saved-model.h5" \
    --crit_parameter 0.5928 \
    --lattice_size 128 \
    --save_dir "./saved_models/hydra" \
    --ckpt_freq 10 
``` 

## License
[Apache License 2.0](https://github.com/adelshb/Improving-critical-exponent-estimations-with-Generative-Adversarial-Networks/blob/main/LICENSE)
