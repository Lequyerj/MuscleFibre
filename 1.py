from model import *
from data import *
from tifffile import imread, imwrite
directoryin = "input/"
directoryout = "mask/"

#delete any existing contents of output directory
folder = directoryout
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
   
#count number of elements in input directory
folder = directoryin
filecounter = 0
for file in os.listdir(directoryin):
    filecounter+=1
    
img = imread(directoryin+'0.tif')
shapey = img.shape
numimg = filecounter
testGene = testGenerator(directoryin,num_image = numimg,target_size = (512,512))
model = neuralnet()
model.load_weights("weights.hdf5")
results = model.predict_generator(testGene, numimg*6, verbose=1)
saveResult(directoryout,results,dims = shapey)