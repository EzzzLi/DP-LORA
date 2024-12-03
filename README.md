# DP-LORA
The official code of  paper "DP-LORA for high quality image synthesis"


## Step 1: Pretraining your own model
After determining the private data, the first step is to select appropriate public data for training the diffusion model.The selection of public data basically follows the condition that there is a small domain gap between public data and private data.Especially for the Latent Diffusion Model, choosing appropriate public data means that the private images can be correctly projected into the Latent Space and restored.Here are some examples of private-public data pairs.

|Private data|Public data|
|:----:|:----:|
| cifar-10  | ImageNet |
| CelebA | ImageNet |

- Auto Encoder:

```
python main.py --base ./configs/autoencoder/yourconfig.yaml -t --gpus 0,
```
- Latent Diffusion Model:
```
python main.py --base ./configs/latent-diffusion/yourconfig.yaml -t --gpus 0,
```

## Step 2: Private Fine-tuning

```
python main.py --base ./configs/finetuning/yourconfig.yaml -t --gpus 0, --accelerator gpu -l <your_log_path>
```

## Step 3: Sampling

- For conditional models:
```
python ./sampling/conditional_sampling.py --yaml <your_log_yaml> --ckpt <your_log_checkpoint> --output output.pt --num_samples 50000 --batch_size 200 --classes 0 1 2 3 4 5 6 7 8 9
```

- For unconditional models:
```
python ./sampling/unconditional_sampling.py --yaml <your_log_yaml> --ckpt <your_log_checkpoint> -o output.pt --num_samples 60000 --batch_size 300
```

## step 4: Evaluation

- FID：
```
python ./fid/compute_dataset_stats.py --dataset ldm.data.celeba.CelebATrain --args size:32 --output celeba_train_stats.npz
python ./fid/compute_samples_stats.py --samples celeba32_samples.pt --output celeba_samples_stats.npz
python ./fid/compute_fid.py  --path1 celeba32_train_stats.npz  --path2 celeba32_samples_stats.npz
```
- Downstream:
```
python ./scripts/cifar_downstream.py
```
