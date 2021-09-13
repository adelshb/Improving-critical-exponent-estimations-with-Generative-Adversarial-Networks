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
pip install .
```

or with

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

## Control parameter estimation via CNN

The trained model for control parameter estimation with CNN is available [here](https://drive.google.com/file/d/1672V_ZPCHSVUohgRHw1nHLROkyo8_rJI/view?usp=sharing).

It possible to train a CNN with the following command:

 ```shell
    python src/CNN_percolation/main.py  \
       --odir "saved_models"        \
       --L 128                      \
       --p_down 0.5                 \
       --p_up 0.7                   \
       --p_increment 0.02           \
       --round_digit 2              \
       --epochs 100                 \
       --n_configs_per_p 1000       \
       --patience 10                \
       --test_size 0.2              \
       --batch_size 32              \
       --random_state 42            \
       --dropout_rate 0          
 ```

## Data augmentation via GAN

### Train GAN

Make sure that configurations have been created first (at "/data/L_128/p_0.5928" in the following example).

```shell
    python src/GAN/train.py \
        --data_dir /data/L_128/p_0.5928 \
        --batch_size 32 \
        --epochs 200 \
        --noise_dim 100 \
        --save_dir /data/models/gan
``` 

### Generate configurations with GAN

Here is the link to a trained generator:
[GAN](https://drive.google.com/file/d/1kfpgoXJTj8s2v96tVL6XtkmiBmWDpG61/view?usp=sharing),
[metadata](https://drive.google.com/file/d/1qIOxPaLd-ORYZdoDV5iOw6wvp3AIoeg-/view?usp=sharing)

```shell
    python src/GAN/generate.py \
        --num 10 \
        --data_dir ./data/generated/
        --model_dir ./data/models/gan/ \
        --noise_dim 100
```

### Verify the control parameter of the GAN-generated configurations

Make sure that synthetic data has been generated by the GAN in 'data/generated/' as .npy files, one file per configuration.

```shell
    python src/benchmark/benchmark.py \
        --synthetic_data_dir ./data/generated/ \
        --CNN_model_dir ./saved_models/CNN_L128_N10000/saved-model.h5 \
        --noise_dim 100
```

## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
