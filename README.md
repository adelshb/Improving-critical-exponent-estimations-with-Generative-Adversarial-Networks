# Improving critical exponent estimations with Generative Adversarial Networks

## Requirements

* Python 3.8+
* Tensorflow
* sklearn
* spicy
* tqdm
* jupyter

```shell
pip install .

```
## Generate and save configurations.

```shell
    python src/statphy/data_factory.py \
        --model square lattice percolation \
        --L 128 \
        --crit_parameter 0.5928 \
        --sample_per_configuration 100
```

## Control parameter estimation via CNN

[CNN](https://drive.google.com/file/d/1672V_ZPCHSVUohgRHw1nHLROkyo8_rJI/view?usp=sharing)

 ```shell
    python3 CNN_percolation/main.py             \
       --odir "saved_files"      \
       --L 128                    \
       --p_down 0.5               \
       --p_up 0.7                 \
       --p_increment 0.02         \
       --round_digit 2            \
       --epochs 100              \
       --n_configs_per_p 1000     \
       --n_gpus 1                 \
       --patience 10              \
       --test_size 0.2            \
       --batch_size 32          \
       --random_state 42          \
       --dropout_rate 0          
 ```

## Data augmentation via GAN

### Train GAN

Make sure that configurations have been created first (at "/data/0.5928" in the following example).

```shell
    python src/GAN/train.py \
        --data_dir /data/0.5928 \
        --batch_size 32 \
        --epochs 200 \
        --noise_dim 100 \
        --save_dir /saved-files
``` 

### Generate configurations with GAN

TO DO

## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
