# Virtual Tryon - Dresscode + Viton HD
<small>Based on a fork from the [Viton-HR](https://github.com/sangyun884/HR-VITON) model. Changes were made to accomodate Dresscode dataset and the inference and training scripts.</small>

<details>
<summary><h2>Getting Started</h2></summary>

### Installation

1. Clone the repository

```sh
git clone https://github.com/green-mint/virtual-tryon
```

2. Install Python dependencies

```sh
conda env create -n tryon -f environment.yml
conda activate tryon
```

### Data Preparation

#### DressCode

1. Download the [DressCode](https://github.com/aimagelab/dress-code) dataset

Once the dataset is downloaded, the folder structure should look like this:

```
├── DressCode
|   ├── test_pairs_paired.txt
|   ├── test_pairs_unpaired.txt
|   ├── train_pairs.txt
│   ├── [dresses | lower_body | upper_body]
|   |   ├── test_pairs_paired.txt
|   |   ├── test_pairs_unpaired.txt
|   |   ├── train_pairs.txt
│   │   ├── images
│   │   │   ├── [013563_0.jpg | 013563_1.jpg | 013564_0.jpg | 013564_1.jpg | ...]
│   │   ├── masks
│   │   │   ├── [013563_1.png| 013564_1.png | ...]
│   │   ├── keypoints
│   │   │   ├── [013563_2.json | 013564_2.json | ...]
│   │   ├── label_maps
│   │   │   ├── [013563_4.png | 013564_4.png | ...]
│   │   ├── skeletons
│   │   │   ├── [013563_5.jpg | 013564_5.jpg | ...]
│   │   ├── dense
│   │   │   ├── [013563_5.png | 013563_5_uv.npz | 013564_5.png | 013564_5_uv.npz | ...]
```

#### VITON-HD

1. Download the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset

Once the dataset is downloaded, the folder structure should look like this:

```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train | test]
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── image-parse-v3
│   │   │   ├── [000006_00.png | 000008_00.png | ...]
│   │   ├── openpose_img
│   │   │   ├── [000006_00_rendered.png | 000008_00_rendered.png | ...]
│   │   ├── openpose_json
│   │   │   ├── [000006_00_keypoints.json | 000008_00_keypoints.json | ...]
```

</details>

<details>
<summary><h2>Inference with Pre-trained Models</h2></summary>

To run the inference on the Dress Code or VITON-HD dataset, run the following command:

```sh
python src/inference.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --test_order [paired | unpaired] --category [all | lower_body | upper_body | dresses ] --mixed_precision [no | fp16 | bf16] --enable_xformers_memory_efficient_attention --use_png --compute_metrics
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --test_order <str>             test setting, options: ['paired', 'unpaired']
    --category <str>               category to test, options: ['all', 'lower_body', 'upper_body', 'dresses'] (default=all)
    --output_dir <str>             output directory
    --batch_size <int>             batch size (default=8)
    --mixed_precision <str>        mixed precision (no, fp16, bf16) (default=no)
    --enable_xformers_memory_efficient_attention <store_true>
                                   enable memory efficient attention in xformers (default=False)
    --allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
    --num_workers <int>            number of workers (default=8)
    --use_png <store_true>         use png instead of jpg (default=False)
    --compute_metrics              compute metrics at the end of inference (default=False)
```

The models wil be automatically downloaded from the online sources.
### Metrics computation

Once you have run the inference script and extracted the images, you can compute the metrics by running the following
command:

```sh
python src/utils/val_metrics.py --gen_folder <path> --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --test_order [paired | unpaired] --category [all | lower_body | upper_body | dresses ]
```

```
    --gen_folder <str>             Path to the generated images folder.
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --test_order <str>             test setting, options: ['paired', 'unpaired']
    --category <str>               category to test, options: ['all', 'lower_body', 'upper_body', 'dresses'] (default=all)
    --batch_size                   batch size (default=32)
    --workers                      number of workers (default=8)
```

</details>

<details>
<summary><h2>Training</h2></summary>

In this section, you'll find instructions on how to train all the components of our model from scratch.

### 1. Train Warping Module

First of all, we need to train the warping module. To do so, run the following command:

```sh
python src/train_tps.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --checkpoints_dir <path> --exp_name <str>
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     dataroot of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       dataroot of vitonhd dataset (required when dataset=vitonhd)
    --checkpoints_dir <str>        checkpoints directory
    --exp_name <str>               experiment name
    --batch_size <int>             batch size (default=16)
    --workers <int>                number of workers (default=10)
    --height <int>                 height of the input images (default=512)
    --width <int>                  width of the input images (default=384)
    --lr <float>                   learning rate (default=1e-4)
    --const_weight <float>         weight for the TPS constraint loss (default=0.01)
    --wandb_log <store_true>       log training on wandb (default=False)
    --wandb_project <str>          wandb project name (default=LaDI_VTON_tps)
    --dense <store_true>           use dense uv map instead of keypoints (default=False)
    --only_extraction <store_true> only extract the images using the trained networks without training (default=False)
    --vgg_weight <int>             weight for the VGG loss (refinement network) (default=0.25)
    --l1_weight <int>              weight for the L1 loss (refinement network) (default=1.0)
    --epochs_tps <int>             number of epochs for the TPS training (default=50)
    --epochs_refinement <int>      number of epochs for the refinement network training (default=50)
```

At the end of the training, the warped cloth images will be saved in the `data/warped_cloths`
and `data/warped_cloths_unpaired` folders.
To save computation time, in the following steps, we will use the pre-extracted warped cloth images.

### 2. Train EMASC

To train the EMASC module, run the following command:

```sh
python src/train_emasc.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path>
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --output_dir <str>             output directory where the model predictions and checkpoints will be written
    --pretrained_model_name_or_path <str>
                                   model identifier from huggingface.co/models (default=stabilityai/stable-diffusion-2-inpainting)
    --seed <int>                   seed for reproducible training (default=1234)
    --train_batch_size <int>       batch size for training (default=16)
    --test_batch_size <int>        batch size for testing (default=16)
    --num_train_epochs <int>       number of training epochs (default=100)
    --max_train_steps <int>        maximum number of training steps. If provided, overrides num_train_epochs (default=40k)
    --gradient_accumulation_steps <int>
                                   number of update steps to accumulate before performing a backward/update pass (default=1)
    --learning_rate <float>        learning rate (default=1e-5)
    --lr_scheduler <str>           learning rate scheduler, options: ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'] (default=constant_with_warmup)
    --lr_warmup_steps <int>        number of warmup steps for learning rate scheduler (default=500)
    --allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
    --adam_beta1 <float>           value of beta_1 for Adam optimizer (default=0.9)
    --adam_beta2 <float>           value of beta_2 for Adam optimizer (default=0.999)
    --adam_weight_decay <float>    value of weight decay for Adam optimizer (default=1e-2)
    --adam_epsilon <float>         value of epsilon for Adam optimizer (default=1e-8)
    --max_grad_norm <float>        maximum value of gradient norm for gradient clipping (default=1.0)
    --mixed_precision <str>        mixed precision training, options: ['no', 'fp16', 'bf16'] (default=fp16)
    --report_to <str>              where to report metrics, options: ['wandb', 'tensorboard', 'comet_ml'] (default=wandb)
    --checkpointing_steps <int>    number of steps between each checkpoint (default=10000)
    --resume_from_checkpoint <str> whether training should be resumed from a previous checkpoint. Use a "latest" to automatically select the last available checkpoint. (default=None)
    --num_workers <int>            number of workers (default=8)
    --num_workers_test <int>       number of workers for test dataloader (default=8)
    --test_order <str>             test setting, options: ['paired', 'unpaired'] (default=paired)
    --emasc_type <str>             type of EMASC, options: ['linear', 'nonlinear'] (default=nonlinear)
    --vgg_weight <float>           weight for the VGG loss (default=0.5)
    --emasc_kernel <int>           kernel size for the EMASC module (default=3)
    --emasc_padding <int>          padding for the EMASC module (default=1)
```

At the end of the training, the EMASC checkpoints will be saved in the `output_dir` folder.

To accelerate the training process for subsequent steps, consider pre-computing the CLIP cloth embeddings for each image
in the dataset.

To do so, run the following command:

```bash
python src/utils/compute_cloth_clip_features.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path>
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --pretrained_model_name_or_path <str>
                                   model identifier from huggingface.co/models (default=stabilityai/stable-diffusion-2-inpainting)
    --batch_size <int>             batch size (default=16)
    --num_workers <int>            number of workers (default=8)
```

The computed features will be saved in the `data/clip_cloth_embeddings` folder.

In the following steps, to use the pre-computed features, make sure to use the `--use_clip_cloth_features` flag.

### 3. Pre-train the inversion adapter

To pre-train the inversion adapter, run the following command:

```sh
python src/train_inversion_adapter.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --gradient_checkpointing --enable_xformers_memory_efficient_attention --use_clip_cloth_features
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --output_dir <str>             output directory where the model predictions and checkpoints will be written
    --pretrained_model_name_or_path <str>
                                   model identifier from huggingface.co/models (default=stabilityai/stable-diffusion-2-inpainting)
    --seed <int>                   seed for reproducible training (default=1234)
    --train_batch_size <int>       batch size for training (default=16)
    --test_batch_size <int>        batch size for testing (default=16)
    --num_train_epochs <int>       number of training epochs (default=100)
    --max_train_steps <int>        maximum number of training steps. If provided, overrides num_train_epochs (default=200k)
    --gradient_accumulation_steps <int>
                                   number of update steps to accumulate before performing a backward/update pass (default=1)
    --gradient_checkpointing <store_true>
                                   use gradient checkpointing to save memory at the expense of slower backward pass (default=False)
    --learning_rate <float>        learning rate (default=1e-5)
    --lr_scheduler <str>           learning rate scheduler, options: ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'] (default=constant_with_warmup)
    --lr_warmup_steps <int>        number of warmup steps for learning rate scheduler (default=500)
    --allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
    --adam_beta1 <float>           value of beta_1 for Adam optimizer (default=0.9)
    --adam_beta2 <float>           value of beta_2 for Adam optimizer (default=0.999)
    --adam_weight_decay <float>    value of weight decay for Adam optimizer (default=1e-2)
    --adam_epsilon <float>         value of epsilon for Adam optimizer (default=1e-8)
    --max_grad_norm <float>        maximum value of gradient norm for gradient clipping (default=1.0)
    --mixed_precision <str>        mixed precision training, options: ['no', 'fp16', 'bf16'] (default=fp16)
    --report_to <str>              where to report metrics, options: ['wandb', 'tensorboard', 'comet_ml'] (default=wandb)
    --checkpointing_steps <int>    number of steps between each checkpoint (default=50000)
    --resume_from_checkpoint <str> whether training should be resumed from a previous checkpoint. Use a "latest" to automatically select the last available checkpoint. (default=None)
    --enable_xformers_memory_efficient_attention <store_true>
                                   enable memory efficient attention in xformers (default=False)
    --num_workers <int>            number of workers (default=8)
    --num_workers_test <int>       number of workers for test dataloader (default=8)
    --test_order <str>             test setting, options: ['paired', 'unpaired'] (default=paired)
    --num_vstar <int>              number of predicted v* per image to use (default=16)
    --num_encoder_layers <int>     number of ViT layers to use in inversion adapter (default=1)
    --use_clip_cloth_features <store_true>
                                   use precomputed clip cloth features instead of computing them each iteration (default=False).
```

At the end of the training, the inversion adapter checkpoints will be saved in the `output_dir` folder.

**NOTE**: You can use the `--use_clip_cloth_features` flag only if you have previously computed the clip cloth features
using the `src/utils/compute_cloth_clip_features.py` script (step 2.5).

### 4. Train VTO

To successfully train the VTO model, ensure that you specify the correct path to the pre-trained inversion adapter
checkpoint. If omitted, the inversion adapter will be trained from scratch. Additionally, don't forget to include the
`--train_inversion_adapter` flag to enable the inversion adapter training during the VTO training process.

To train the VTO model, run the following command:

```sh
python src/train_vto.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --inversion_adapter_dir <path> --gradient_checkpointing --enable_xformers_memory_efficient_attention --use_clip_cloth_features --train_inversion_adapter
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --output_dir <str>             output directory where the model predictions and checkpoints will be written
    --inversion_adapter_dir <str>  path to the inversion adapter checkpoint directory. Should be the same as `output_dir` of the inversion adapter training script. If not specified, the inversion adapter will be trained from scratch. (default=None)
    --inversion_adapter_name <str> name of the inversion adapter checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
     --pretrained_model_name_or_path <str>
                                   model identifier from huggingface.co/models (default=stabilityai/stable-diffusion-2-inpainting)
    --seed <int>                   seed for reproducible training (default=1234)
    --train_batch_size <int>       batch size for training (default=16)
    --test_batch_size <int>        batch size for testing (default=16)
    --num_train_epochs <int>       number of training epochs (default=100)
    --max_train_steps <int>        maximum number of training steps. If provided, overrides num_train_epochs (default=200k)
    --gradient_accumulation_steps <int>
                                   number of update steps to accumulate before performing a backward/update pass (default=1)
    --gradient_checkpointing <store_true>
                                   use gradient checkpointing to save memory at the expense of slower backward pass (default=False)
    --learning_rate <float>        learning rate (default=1e-5)
    --lr_scheduler <str>           learning rate scheduler, options: ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'] (default=constant_with_warmup)
    --lr_warmup_steps <int>        number of warmup steps for learning rate scheduler (default=500)
    --allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
    --adam_beta1 <float>           value of beta_1 for Adam optimizer (default=0.9)
    --adam_beta2 <float>           value of beta_2 for Adam optimizer (default=0.999)
    --adam_weight_decay <float>    value of weight decay for Adam optimizer (default=1e-2)
    --adam_epsilon <float>         value of epsilon for Adam optimizer (default=1e-8)
    --max_grad_norm <float>        maximum value of gradient norm for gradient clipping (default=1.0)
    --mixed_precision <str>        mixed precision training, options: ['no', 'fp16', 'bf16'] (default=fp16)
    --report_to <str>              where to report metrics, options: ['wandb', 'tensorboard', 'comet_ml'] (default=wandb)
    --checkpointing_steps <int>    number of steps between each checkpoint (default=50000)
    --resume_from_checkpoint <str> whether training should be resumed from a previous checkpoint. Use a "latest" to automatically select the last available checkpoint. (default=None)
    --enable_xformers_memory_efficient_attention <store_true>
                                   enable memory efficient attention in xformers (default=False)
    --num_workers <int>            number of workers (default=8)
    --num_workers_test <int>       number of workers for test dataloader (default=8)
    --test_order <str>             test setting, options: ['paired', 'unpaired'] (default=paired)
    --uncond_fraction <float>      fraction of unconditioned training samples (default=0.2)
    --text_usage <str>             text features to use, options: ['none', 'noun_chunks', 'inversion_adapter'] (default=inversion_adapter)
    --cloth_input_type <str>       cloth input type, options: ['none', 'warped'], (default=warped)
    --num_vstar <int>              number of predicted v* per image to use (default=16)
    --num_encoder_layers <int>     number of ViT layers to use in inversion adapter (default=1)
    --train_inversion_adapter <store_true>
                                   train the inversion adapter during the VTO training (default=False)
    --use_clip_cloth_features <store_true>
                                   use precomputed clip cloth features instead of computing them each iteration (default=False).
            
```

At the end of the training, the checkpoints will be saved in the `output_dir` folder.

**NOTE**: You can use the `--use_clip_cloth_features` flag only if you have previously computed the clip cloth features
using the `src/utils/compute_cloth_clip_features.py` script (step 2.5).

### 5. Inference with the trained models

Before running the inference, make sure to specify the correct path to all the trained checkpoints.
Make sure to also use coherent hyperparameters with the ones used during training.

To run the inference on the Dress Code or VITON-HD dataset, run the following command:

```sh
python src/eval.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --save_name <str> --test_order [paired | unpaired]  --unet_dir <path> --inversion_adapter_dir <path> --emasc_dir <path>  --category [all | lower_body | upper_body | dresses ] --enable_xformers_memory_efficient_attention --use_png --compute_metrics
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --output_dir <str>             output directory where the generated images will be written
    --save_name <str>              name of the generated images folder inside `output_dir`
    --test_order <str>             test setting, options: ['paired', 'unpaired']
    --unet_dir <str>               path to the UNet checkpoint directory. Should be the same as `output_dir` of the VTO training script
    --unet_name <str>              name of the UNet checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
    --inversion_adapter_dir <str>  path to the inversion adapter checkpoint directory. Should be the same as `output_dir` of the VTO training script. Needed only if `--text_usage` is set to `inversion_adapter`. (default=None)
    --inversion_adapter_name <str> name of the inversion adapter checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
    --emasc_dir <str>              path to the EMASC checkpoint directory. Should be the same as `output_dir` of the EMASC training script. Needed when --emasc_type!=none. (default=None)
    --emasc_name <str>             name of the EMASC checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
    --pretrained_model_name_or_path <str>
                                   model identifier from huggingface.co/models (default=stabilityai/stable-diffusion-2-inpainting)
    --seed <int>                   seed for reproducible training (default=1234)
    --batch_size <int>             batch size(default=8)
    --allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
    --enable_xformers_memory_efficient_attention <store_true>
                                   enable memory efficient attention in xformers (default=False)
    --num_workers <int>            number of workers (default=8)
    --category <str>               category to test, options: ['all', 'lower_body', 'upper_body', 'dresses'] (default=all)
    --emasc_type <str>             type of EMASC, options: ['linear', 'nonlinear'] (default=nonlinear)
    --emasc_kernel <int>           kernel size for the EMASC module (default=3)
    --emasc_padding <int>          padding for the EMASC module (default=1)
    --text_usage <str>             text features to use, options: ['none', 'noun_chunks', 'inversion_adapter'] (default=inversion_adapter)
    --cloth_input_type <str>       cloth input type, options: ['none', 'warped'], (default=warped)
    --num_vstar <int>              number of predicted v* per image to use (default=16)
    --num_encoder_layers <int>     number of ViT layers to use in inversion adapter (default=1)
    --use_png <store_true>         use png instead of jpg (default=False)
    --num_inference_steps <int>    number of diffusion steps at inference time (default=50)
    --guidance_scale <float>       guidance scale of the diffusion (default=7.5)
    --use_clip_cloth_features <store_true>
                                   use precomputed clip cloth features instead of computing them each iteration (default=False).
    --compute_metrics              compute metrics at the end of inference (default=False)
```

The generated images will be saved in the `output_dir/save_name_{test_order}` folder.

**NOTE**: You can use the `--use_clip_cloth_features` flag only if you have previously computed the clip cloth features
using the `src/utils/compute_cloth_clip_features.py` script (step 2.5).

</details>


## Inference on Custom Images:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wb_SHQNIBohKwSOn1npRfxOcseJd11nx?usp=sharing)