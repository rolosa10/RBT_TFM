# RBT Tracking

The present repository contains the dataset, the Jupyter notebooks and the python executable scripts developed for the master thesis entitled "Rehabilitation Body Tracking (RBT): A deep end-to-end 3D human pose estimation system for a science-based neuro-rehabilitation solution". 



### Environment reproduction (CPU & GPU)
The proposed solution and its installation is only described for a Windows 10 OS. 
To reproduce the results and run it on your own computer it is required to create an environment with specific dependencies. 

Steps to reproduce the environment: 

1. Clone the github repository. 
2. Install Anaconda for Windows10 from: https://www.anaconda.com/products/individual
3. Open Anaconda Prompt (Anaconda3), go to the path where the github repo has been cloned and run the following commands:
    <pre><code>
    conda create --name [name_environment] python==3.7.0
    conda activate [name_environment]
    pip install -r requirements.txt
    </code></pre>
    

#### GPU steps only
Perform the steps from above and the following if you want to run the solutions in GPU: 

1. Download the CUDA Toolkit 10.1 for Windows 10 from: https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
2. To process images through GPU it is still missing a DLL. From the following URL: https://developer.nvidia.com/rdp/cudnn-archive download the cuDNN Library foir Windows 10 named “Download cuDNN v7.6.5 (November 5th, 2019), for cuda 10.1. Uncompress the downloaded file and copy the cudnn64.7.dll in the bin folder inside the CUDA10.1 folder downloaded in the previous step.
3. Check and update, if required, the Driver’s Version regarding the downloaded CUDA toolkit: https://docs.nvidia.com/dpeloy/cuda-compatibility/index.html. The driver version can be updated directly from NVIDIA 


### Folders information 

#### Blender Scripts
It contains the python script that has been used to load the FBX files generated from the MoCap studio and obtain the 3D coordinates in the camera frame as well as the 2D project keypoint coordinates. 

#### Datasets

It contains under RGSClinic/No-Gravity folders three csv files: 
1. Dataset_MoCap_RGSclinic_KinectCamera: 2D and 3D ground-truth coordinates for MoCap data generated through Blender
2. Augmented_Dataset_Synthetically: Dataset generated synthetically through Unity in for assessing the improvement pipeline.
3. Exported_Synth_Dataset_Unity.csv: Dataset created synthetically to assess MediaPipe's performance. (PCK@0.5)
 

#### General notebooks
1. MP_2D_Analysis_Performance: Dataset that loads the "Exported_Synth_Dataset_Unity" to compute the PCK@0.5 metric and assess the performance for MediaPipe
2. Quantitative_and_qualitative_results: As the name itself suggests, the qualitative and quantitative results for the feed-forward with residual connections architecture trained with noise lvl 1. 

#### Metrics
Contains the two testing methods that have been used: The test set and the marker testing. 

1. Test set: There are two folders, LSTM and Martinez, both containing the 2D ground-truth data for the test set. There's a notebook that allows computing the MPJPE metrics obtained. 
2. Marker testing: The scripts and notebooks are provided for the camera calibration and the Marker testing itself for AzureSDK, MediaPipe3D, LSTM and Martinez. 

#### Solutions
The trained models files, the notebooks for defining the architecture and train them and the Python scripts that run in the background when implemented into RGS are provided for the sequence-to-sequence and feed-forward with residual connection architectures. 

#### Video of how synthetic data has been generated for data augmentation purposes


![bloggif_62af4d8757775](https://user-images.githubusercontent.com/41288642/174490640-6e5ab782-dfa5-4a87-a339-6748f8d90c94.gif)




