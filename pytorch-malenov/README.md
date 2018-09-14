# Pytorch-MalenoV

This is an educational implementation of MalenoV (Machine Learning of Voxels) by Ildstadt and Bormann ([GitHub](https://github.com/bolgebrygg/MalenoV)) in Pytorch.  

The data and labels required to run Pytorch-MalenoV have been ported from segy formats to numpy and are available [here](https://drive.google.com/drive/folders/1qeUTCsTBtjj7GbBXClwdguVwoLwcdO7h)  

Thanks also to @JesperDramsch for providing a modularised verison of the original MalenoV code ([Github](https://github.com/JesperDramsch/MalenoV/tree/master/malenov))  
 

To run Pytorch-MalenoV you will need a standard scientific Python 3.6 stack such as anaconda, Pytorch version > 0.4 and tensorboardX for training visualization.  

To run Pytorch-MalenoV with default parameters run the following:  

```
python train.py --use_stratified_kfold --lr 0.001 --batch_size  32 --num_examples 10000 --beta1 0.9 --beta2 0.999
```

This will run 10 epochs of training with 10000 examples shown per epoch.  
The stratified shuffle split will split into 80% train and 20% validation while reproducing the original class distribution in train and validation sets.  

If you wish, you may also run with a spatial split to have less correlation between training and validation set.  
Please see the accompanying jupyter notebooks for more detail.  

To run with a spatial split of the original labels:  
```
python train.py --lr 0.001 --batch_size  32 --num_examples 10000 --beta1 0.9 --beta2 0.999
```


You can also test models by running for example:  
```
python test.py --inline 500 --checkpoint_path ./results/malenov/180913_2034/model_epoch_2.pth.tar --batch_size 64 
```

This will make predictions on inline 500 of the F3 dataset and output to the local folder.  
