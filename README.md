# RBT Tracking


RGSpipe is a fully automated pipeline that consist of two interconnected procedures: MediaPipe to detect 2D joint coordinates and a neural network with roughly 5 million trainable parameters with batch normalization, dropout regularization and rectified Linear units that allows mapping the 2D coordinates into the Kinect's 3D coordinate system. 


### Dataset information
mocap_dataset:
    - v1: 3D and 2D keypoint coordinates transformed, rotated and projected from MoCap studio to Kinect camera frame
    - v1_augmennted_strategy_1: 3D and 2D keypoint coordinates transformed, rotated and projected from MoCap studio to Kinect camera frame and augmented within the X axis to be more homogeneous in all table surface
    - v1_augmented_strategy_2_gravity: Same dataset as v1 but augmented increasing the Y axis for elbow and wrist coordinates.
    - v1_augmented_strategy_3_gravity: Same dataset as v12 but augmented increasing the Y axis for elbow and wrist coordinates. Limiting the Y to other values than "v1_augmented_strategy_2_gravity".
    - v2_standing: 3D and 2D keypoint coordinates of Standing videos in MoCap transformed, rotated and displaced from MoCaP to Kinect camera frame. Moreover, coordinates have been displaced to avoid "Standing" and having "Sit" view.
    - v3_standing: Same data as v2_standing but not displacing coordinates to "Sit" view. Just MoCap to Kinect frame

    
synthetic_dataset:
    - v1_dataset_50k: Dataset with just 8 objective points as targets and regression between intermediate points
    - v2_dataset_150k: Dataset with 150k entries generated in an homogeneous way via RayCast along all table surface
    - v3_dataset_500k: Dataset with 500k entries generated in an homogeneous way via RayCast along all table surface
    - v4_dataset_300k: Dataset with 300k entries per Gravity training

### Environment reproduction
The proposed solution and its installation is only described for a Windows 10 OS. 
To run RGSpipe on your own computer it is required to reproduce a specific environment. The solution runs entirely in CPU in 16,27 ms in average per frame. 

Steps to reproduce the environment: 

1. Clone the github repository of the URL in the desired path
2. Install Anaconda for Windows10 from: https://www.anaconda.com/products/individual
3. Open Anaconda Prompt (Anaconda3), go to the path where the github repo has been cloned and run the following commands:
    <pre><code>
    conda create --name [name_environment] python==3.7.0
    conda activate [name_environment]
    pip install -r requirements.txt
    </code></pre>
    
  

RGSpipe CPU&GPU:

1. Clone the github repository of the URL in the desired path: 
2. Download the CUDA Toolkit 10.1 for Windows 10 from: https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
3. To process images through GPU it is still missing a DLL. From the following URL: https://developer.nvidia.com/rdp/cudnn-archive download the cuDNN Library foir Windows 10 named “Download cuDNN v7.6.5 (November 5th, 2019), for cuda 10.1. Uncompress the downloaded file and copy the cudnn64.7.dll in the bin folder inside the CUDA10.1 folder downloaded in the previous step.
4. Check and update, if required, the Driver’s Version regarding the downloaded CUDA toolkit: https://docs.nvidia.com/dpeloy/cuda-compatibility/index.html. The driver version can be updated directly from NVIDIA GeForce Experience software.
5. Install Anaconda for Windows 10 from: https://www.anaconda.com/products/individual
6. Open Anaconda Prompt (Anaconda3), go to the path where the github repo has been cloned and run the following commands:
<pre><code>
conda create --name [name_environment] python==3.7.0
conda activate [name_environment]
pip install -r requirements.txt
</code></pre>

