# MuscleFibre

#Installation
Install Anaconda by following instrctions at this link: https://docs.anaconda.com/anaconda/install/.

Create a new conda environment and install the required packages:

```python
conda create --name muscles
conda activate muscles
conda install -c anaconda scikit-image
conda install -c pytorch pytorch
conda install -c conda-forge tifffile
conda install -c conda-forge opencv
```


#Usage
Place all images you want to segment in the input folder. Double click 1.bat to get initial segmentation. Navigate to contours folder to view initial segmentation and correct using imageJ where necessary (draw white where you want detected contours to be erased and black where you want new contours to be drawn). Then run 2.bat to get final segmentation and navigate to the contours folder to view this. 

In the feret folder you will find an image of the contours with the feret diamteter included. In the output folder you will find a 2 column matrix, column 1 contains the area of the detected fibre in pixels and column 2 contains the feret diameter. Each row corresponds to a different detected fibre.
