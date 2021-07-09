# MuscleFibre Detector

# Installation
If you don't already have anaconda, install it by following instrctions at this link: https://docs.anaconda.com/anaconda/install/.

You will also need to have ImageJ installed: https://imagej.nih.gov/ij/download.html.

Open Anaconda Prompt and enter the following commands to create a new conda environment and install the required packages:

```python
conda create --name muscles
conda activate muscles
conda install -c anaconda scikit-image
conda install -c pytorch pytorch
conda install -c conda-forge tifffile
conda install -c conda-forge opencv
```
Now, within this GitHub repository, go to releases on the right to download the weights.pth file. Place this file in the master directory.

# Usage
Place all images you want to segment in the input folder. Run the fully automated pass, by navigating to the master directory, opening Anaconda Prompt and running the following commands:
```python
conda activate muscles
python 1.py
```
Navigate to contours folder to view initial segmentation and correct using imageJ where necessary (draw white where you want detected contours to be erased and black where you want new contours to be drawn). Then run the manual correction script, by running the following command:
```python
python 2.py
```
In the feret folder you will find an image of the contours with the feret diamteter included. In the output folder you will find a 2 column matrix, column 1 contains the area of the detected fibre in pixels and column 2 contains the feret diameter. Each row corresponds to a different detected fibre.
