# Beacon2Sciencee: Enhancing STEREO/HI beacon data1 with machine learning for efficient CME tracking


Code for the different components of the "Beacon2Science" pipeline. 

### Data

In the Data folder, all the functions required to create the training and testing datasets can be found, along with the processing function to produce L1 and L2 data for STEREO HI. Some examples of how to call the reduction function are provided. Since creating the dataset is a lengthy process for science and beacon, each function should be called sequentially. 

###  Training and testing
The different train and test .py files enable reproducing the paper's results from scratch once the data is processed and saved.

### Realtime

To execute the code in real-time, we provide scripts and functions to process data, launch models, and generate the required plots in the RealTime folder. The models can be downloaded from the figshare link found in the paper. 

### Tracking

We provide the tracking tool used in the paper for all the events in the tracking folder, but also a lighter version in the RealTime folder.


