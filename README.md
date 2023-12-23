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
```

At the end of the training, the EMASC checkpoints will be saved in the `output_dir` folder.

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
```

At the end of the training, the checkpoints will be saved in the `output_dir` folder.

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
```

The generated images will be saved in the `output_dir/save_name_{test_order}` folder.

</details>


## Inference on Custom Images:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wb_SHQNIBohKwSOn1npRfxOcseJd11nx?usp=sharing)