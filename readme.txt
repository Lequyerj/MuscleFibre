After installing anaconda, go to terminal and enter the following commands in sequence:

conda create --name muscles
conda activate muscles
conda install -c anaconda scikit-image
conda install -c pytorch pytorch-cpu
conda install -c conda-forge tifffile
conda install -c conda-forge opencv

#Usage
Place all images you want to segment in the input folder. Open a terminal in the same folder as 1.py and enter the commands:

conda activate muscles
python 1.py

This gets you an initial segmentation. Navigate to contours folder to view initial segmentation and correct using imageJ where necessary (draw white where you want detected contours to be erased and black where you want new contours to be drawn). Then enter the command:

python 2.py

Which gets you a final segmentation. Navigate to the contours folder to view this. 

In the feret folder you will find an image of the contours with the feret diamteter included. In the output folder you will find a 2 column matrix, column 1 contains the area of the detected fibre in pixels and column 2 contains the feret diameter. Each row corresponds to a different detected fibre.
