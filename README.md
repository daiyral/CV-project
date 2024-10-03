# CV-project
Authors:
- Daniel Aibinder
- Noy Cohen
  
## Project Overview

In this project, our goal is to introduce a form of supervised learning by adding supervision on the predicted splatter image. To accomplish this, we generate a ground truth splatter image using a pretrained model from the original code with a resolution of 128x128. 

The objective is to train a 'weaker' model from scratch and have it converge faster while achieving comparable results to the 'stronger' standard model by leveraging the ground truth generated from the stronger model.

## Installation Guide

### Step 1: Clone the Repository
```
git clone https://github.com/daiyral/CV-project.git --recursive
```

### Step 2: Set Up C++ Compiler for PyTorch Extensions
- Ensure you have a C++ compiler installed for PyTorch extensions. We used Visual Studio 2022 for Windows.
- Add the following path to your environment variables:
  ```
    C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64
  ```

### Step 3: Ensure CUDA SDK is Installed
- We used CUDA 12.1. To verify that you have the CUDA compiler driver installed, run:
  ```
  nvcc --version
  ```
  If CUDA is not installed, download it from [here](https://developer.nvidia.com/cuda-12-1-0-download-archive).

### Step 4: Modify CUDA Toolkit Configuration
- Open Notepad as an administrator.
- Go to the file:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include\crt\host_config.h
  ```
- Replace the following code:

  ```c++
  #if _MSC_VER < 1910 || _MSC_VER >= 1910
  #error -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
  #elif _MSC_VER >= 1910 && _MSC_VER < 1910
  ```

  with:

  ```c++
  #if _MSC_VER < 1910 || _MSC_VER >= 2940
  #error -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect runtime execution. Use at your own risk.
  #elif _MSC_VER >= 1910 && _MSC_VER < 1910
  ```
- Save the file.

### Step 5: Install Conda
If Conda is not installed, download it from [here](https://docs.anaconda.com/miniconda/)

### Step 5: Install Dependencies
1. Navigate to the `gaussian-splatting` directory:
   ```
   cd gaussian-splatting
   ```
2. For Windows only, set the `DISTUTILS_USE_SDK` environment variable:
   ```
   SET DISTUTILS_USE_SDK=1
   ```
3. Activate your C++ environment: Nagivate to the directory where vcvars64.bat is located The common path is:
   ```
   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build
   ```
   Run the following command:
   ```
   ./vcvars64.bat
   ```
4. Create and activate the conda environment:
   ```
   conda env create --file environment.yml
   conda activate gaussian_splatting
   ```
   This will install a conda environment with Python 3.8 and CUDA 12.1.

### Step 6: Install Additional Requirements
1. Navigate to the `splatter image` directory:
   ```
   cd splatter image
   ```
2. Install other required packages:
   ```
   pip install -r requirements.txt
   ```
### Note: if you are still running into issues you might need to install colmap and ffmpeg and add them to your environment variable path.
```
colmap : https://colmap.github.io/install.html
FFmpeg: https://www.ffmpeg.org/download.html
```

## Downloading and extracting the data:
For training / evaluating on cars dataset please download the srn_cars.zip from PixelNeRF data folder. Unzip the data file and change SHAPENET_DATASET_ROOT in datasets/srn.py to the parent folder of the unzipped folder. For example, if your folder structure is: /home/user/SRN/srn_cars/cars_train, in datasets/srn.py set SHAPENET_DATASET_ROOT="/home/user/SRN". No additional preprocessing is needed.

## Training Configuration:
To activate our additions to the project go to default_config.yaml
```
Make sure useSplatterGT Variable is set to True
And 0<lambda_spaltter<=1
```
This activates the supervised learning using ground truth splatter image.
The second paramater is lambda_splatter.
This controls how much weight is given to the ground truth splatter in the loss function. The default we used is 0.5 (1 is the max where you only use the new loss function we added)

You can also change the number of iterations the model trains using the iterations variable and change the image resolution for lower vram using the training_resolution variable.
## Training:
Run the following command:
```
python train_network.py +dataset=cars
```
Once it is completed, place the output directory path in configs/experiment/lpips_$experiment_name.yaml in the option opt.pretrained_ckpt (by default set to null).
and run:
```
python train_network.py +dataset=cars +experiment=lpips_100k.yaml
```

## Proposed Approach:
We introduced a custom loss function for comparing splatter images generated by the network with their ground truth. This loss function combines both Smooth L1 loss and KL divergence, taking into account the opacity of the splatter images and scaling based on the magnitude of the components.

## `eval.py` Changes
We added functionality to save all splatter images from the original model's predictions into a pickle file called `splatter_gt.pickle`. This allows us to later load and compare the ground truth splatter images against the model's predictions.

## `loss_utils.py` Changes

1. **`normalize_to_distribution function`**  
   This function applies the softmax activation to the splatter image tensor, converting it into a probability distribution. This is required for the KL divergence calculation.

2. **`kl_divergence_loss function`**  
   We added a function that calculates the KL divergence using the log distribution of softmax-activated tensors to compare the model’s predictions with the ground truth.

3. **`down_scale_splatter function`**  
   This function downscales the ground truth splatter images from a resolution of 128x128 down to 64x64, which is the resolution used for training.

4. **`filter_by_opacity function`**  
   This function filters out pixels in the splatter image that have low opacity. We do not want our loss function to account for these pixels, as they do not contribute meaningfully to the loss.

#### `splatter_image_loss function`
The main function `splatter_image_loss` computes the total loss by taking into account both the Smooth L1 loss and the KL divergence between the ground truth and the model's prediction:

- It filters out low-opacity pixels.
- It downsamples the ground truth image to the resolution used in training (64x64).
- For each component of the splatter image (color, sigma, xyz, opacity):
  1. Calculates the Smooth L1 loss between the ground truth and prediction.
  2. Calculates the KL divergence loss using the softmax-activated distributions.
  3. Computes the magnitude of the components to scale the losses.
  4. Combines the losses and scales them by the magnitude.
  5. The mean loss across all components is calculated.

This can be formulated as follows:
- **N** be the number of images the splatter-image is composed of.
- **network_output_k** be the output tensor for component **k**.
- **gt_k** be the corresponding ground truth tensor for component **k**.
- **L_KL(x, y)** be the KL divergence loss function.
- **L_smoothL1(x, y)** be the Smooth L1 loss function.
- **v_m = (1/n) * sum(|v_i|)** be the mean magnitude of the values in **v**.

The total loss for each component **k** is a combination of the Smooth L1 loss and the KL divergence loss, scaled by the magnitude of **v**:

\[
L_k = v_m \left( 0.1 \cdot L_{\text{smoothL1}}(v_k, gt_k') + 0.9 \cdot L_{KL}(v_k, gt_k') \right)
\]

Where:
- **v_k** represents the model's prediction for component **k**.
- **gt_k'** represents the ground truth for component **k**, downsampled to the training resolution (if applicable).
- The terms are weighted with **0.1** for Smooth L1 loss and **0.9** for KL divergence.

The final total loss is computed as the mean loss across all **N** components:

\[
L_{\text{total}} = \frac{1}{N} \sum_{k=1}^{N} L_k
\]
### Reducing Points Based on Opacity Levels:
Since the object (a car) contrasts with the white background, we can filter out points with low opacity that likely belong to the background.
By eliminating these outliers, we focus on higher-quality data that better represents the object.
This preprocessing step helps the model learn from more relevant and accurate data, improving its overall performance.

#### With opacity=0:
![image](https://github.com/user-attachments/assets/669d3d1d-aa0c-4d55-aff5-b60aabecaf6c)

#### With opacity=0.2:
![image](https://github.com/user-attachments/assets/aedbae83-29a8-435a-92db-67d313c02ac0)


