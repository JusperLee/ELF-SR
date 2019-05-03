If you want to train ESRGAN-Pro, use the following command:

cd train/code
python train.py -use option/train/train_ESRGAN.json

If you want to get the results of this competition, 
please use the following command:

cd test
python test.py models/model_best.pth



The SR image obtained by performing the above steps is post-processed again by the Back projection method, 
and the post-processing method is post processing in the folder. 
Put the LR image into /LR, 
put the result of the model processing into /results, 
and the final generated result is stored in /Post_processing. 
Use the command as follows:

matlab -nodisplay -r back-projection
