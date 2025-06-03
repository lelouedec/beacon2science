# Beacon2Science: Enhancing STEREO/HI beacon data with machine learning for efficient CME tracking


Code for the different components of the "Beacon2Science" pipeline. A copy of the python packages used to produce this paper results can be found in requirements.txt

### Data

In the Data folder, all the functions required to create the training and testing datasets can be found, along with the processing function to produce L1 and L2 data for STEREO HI. Some examples of how to call the reduction function are provided. Since creating the dataset is a lengthy process for science and beacon, each function should be called sequentially. 
In each folder, the functions.py is an adapted version of the code found in https://github.com/maikebauer/STEREO-HI-Data-Processing , with the core functions remaining the same.

###  Training and testing
The different train and test .py files enable reproducing the paper's results from scratch once the data is processed and saved.
For training for example: python trainNN1.py configs/config_NN1.json

We provide the dataset files list in dataset_train_final.json for NN1 and sequences_dataset.json for NN2. 


### Realtime

To execute the code in real-time, we provide scripts and functions to process data, launch models, and generate the required plots in the RealTime folder. The models can be downloaded from the figshare link found in the paper. Simply launch RT_b2s.py (or adapt the bash script to have a cron job do the process at regular intervals). Videos and Jplots are available at https://helioforecast.space/cme, for faster access. 

### Tracking

We provide the tracking tool used in the paper for all the events in the tracking folder, but also a lighter version in the RealTime folder.


