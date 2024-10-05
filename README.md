![image](https://github.com/user-attachments/assets/943931c3-711b-4fe2-8b17-3348860ab3c9)# CV-project
Authors:
- Daniel Aibinder
- Noy Cohen
  
## Project Overview

In this project, our goal is to introduce a form of supervised learning by adding supervision on the predicted splatter image. To accomplish this, we generate a ground truth splatter image using a pretrained model from the original code with a resolution of 128x128. 

The objective is to train a 'weaker' model from scratch and have it converge faster while achieving comparable results to the 'stronger' standard model by leveraging the ground truth generated from the stronger model.

![diagram](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/Idea%20Diagram.png)

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
And 0 < lambda_spaltter <= 1
```
This activates the supervised learning using ground truth splatter image.
The second paramater is lambda_splatter.
This controls how much weight is given to the ground truth splatter in the loss function. The default we used is 0.5 (1 is the max where you only use the new loss function we added)

You can also change the number of iterations the model trains using the iterations variable and change the image resolution for lower vram using the training_resolution variable.
## Training:
First you must produce the splatter gt locally:
Run the following command :
```
python eval.py cars
```
once the eval is finished you should see a file named splatter_gt.pickle generated in splatter-image directory.
Now to train a new network with the ground truth supervision:
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
- **v** be the output tensor for component **k** (the value for the component k).
- **gt_k** be the corresponding ground truth tensor for component **k**.
- **L_KL(x, y)** be the KL divergence loss function.
- **L_smoothL1(x, y)** be the Smooth L1 loss function.
- ![eq1](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/equation1.png) - the mean magnitude of the values in **v** for component **k**.

The total loss for each component **k** is a combination of the Smooth L1 loss and the KL divergence loss, scaled by the magnitude of **v**:

![eq2](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/equation2.png)

The terms are weighted with **alpha** for Smooth L1 loss and **beta** for KL divergence. **We used 0.1 and 0.9 respectively for the best results but can be any number**

The final splatter loss is computed as the mean loss across all **N** components:

![eq3](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/equation3.png)

This splatter loss with be added to the networks total loss calculated in train_network.py

## `train_network.py` Changes
- Added the ability to log the splatter images to wandb
- In the training configuration if useSplatterGT is enabled then we load the splatter_gt.pickle file produced from eval.py 
- We calculate the splatter loss as described above
- We multiply it by lambda_splatter which is a regularization factor that you can change in the defualt_config.yaml file
If we define **L_prev** to be the loss value from the original paper our new loss function is:
![eq4](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/equation4.png)

## Reducing Points Based on Opacity Levels:
Since the object (a car) contrasts with the white background, we can filter out points with low opacity that likely belong to the background.
By eliminating these outliers, we focus on higher-quality data that better represents the object.
This preprocessing step helps the model learn from more relevant and accurate data, improving its overall performance.
As you can see from the images above, increasing the opacity threshold from 0 to 0.2 significantly reduces the number of background points, resulting in a cleaner representation of the car object. 

#### Visual Comparison:

##### With opacity=0:
![image](https://github.com/user-attachments/assets/669d3d1d-aa0c-4d55-aff5-b60aabecaf6c)

##### With opacity=0.2:
![image](https://github.com/user-attachments/assets/aedbae83-29a8-435a-92db-67d313c02ac0)

#### Part 1: Visualization
##### How to run?
1. To run the code, you need to have the following Python libraries installed:
```bash
pip install torch numpy plotly dash
```

2. Name your pickle file 'splatter_gt.pickle' and save it in the same location as the Python script.

3. Run the Python script:
```bash
python opacity_control.py
```
  This will start a Dash web app on http://127.0.0.1:8050/ by default.

 4. Using the web interface:
    * Open a web browser and go to http://127.0.0.1:8050/.
    * You will see a 3D scatter plot of the sampled points from the pickle file.
    * Use the opacity slider to adjust the opacity threshold, dynamically filtering the points displayed.
  
  ##### Customization
  * Change the opacity threshold: Modify the default opacity threshold in the slider by adjusting the value parameter in the dcc.Slider component.
  * Sample size: The number of points displayed can be adjusted by changing the sample_size parameter in the sample_points() function.
  
#### Part 2: Easy Access with get_filtered_points_with_opacity
The get_filtered_points_with_opacity() function provides easy access to filtered point cloud data based on opacity thresholds. This function is crucial for processing the point cloud data and can be used independently of the visualization interface.

##### How to run?
1. To run the code, you need to have the following Python libraries installed:
```bash
pip install torch numpy
```

2. Import the function:
   ```python
   from your_module import get_filtered_points_with_opacity
   ```
   
3. Call the function:
   ```python
   x_sample, y_sample, z_sample = get_filtered_points_with_opacity(image, opacity_threshold=0.2, sample_size=10000)
   ```

## Experiments
Using our loss function, our model demonstrated an improvment in performance in terms of SSIM, PSNR, and LPIPS on novel view synthesis tasks. These improvements indicate that our model not only learns faster but also converges to the optimal solution more efficiently.
We conducted experiments by testing various hyperparameters, regularization techniques, and loss functions (e.g., MSE, L2, L1). Additionally, we experimented with different opacity thresholds to effectively filter out certain parts of the splatted image. And many more...

[Here](https://wandb.ai/radiostars/gs_pred/workspace) you can see some of the runs we attempted:

![runs](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/runs.png)

Wandb logs allowed us to test different loss functions and different paramaters to improve our model.

Here for example, you can see how we compared mse(orange) with smooth_l1(green) with a combination of ssim and regular l1(brown) with kl loss with different values for the lambdas. This gave us an indication that kl loss gave the best results as it gave the best psnr, lpips and ssim results.

![comapre](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/compare.png)

## Results
Below are some performance graphs from our most successful run. The we trained the original model with no additions and it is represented by the gray line, while our model using the proposed approach is shown in orange:
![lpips](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/lpips.png)
![psnr](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/psnr.png)
![ssim](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/ssim.png)

Here we can see some of the model renders:

**Note the gt we show here is from the pretrained provided model that was trained for much longer and thats why it looks much better**
![pred](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/rot_gif.gif)
![pred](https://raw.githubusercontent.com/daiyral/CV-project/refs/heads/main/imgs/img_pred_vs_img_gt.png)




