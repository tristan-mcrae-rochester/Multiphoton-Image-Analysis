



#Preprocessing module-------------------------------------------------
#Subtract background noise
#Split into different channels

#PSF module-----------------------------------------------------------
#calculate PSF from z-stacks of sub-resolution beads

#Training module------------------------------------------------------
#import DL and other libraries

#import z-stack from preprocessing module

#import PSF from PSF module


#create inputs and labels
#find rotated PSF (or other blur function for prototyping)
#split lateral (xy) slices into patches if more training data is desired to create gxy. Be aware of whether rotation is compatible with psf
#apply blurring function to gxy
#downsample blurred gxy to get pxy (inputs)
#subtract gxy from pxy to get residuals (labels)


#Define network layers
#Conv 16,7,7
#Max 2,2
#Conv 32, 7, 7
#Max 2,2
#Conv 64, 7, 7
#Upsample 2, 2
#Conv 32, 7, 7
#Upsample 2, 2
#Conv 16, 7, 7
#Conv 1, 1, 1
#Add symmetric skip connections if desired
#Add output of final layer to input to get final prediction


#Define other network info
#100 epochs
#adam optimizer
#lr of 5E-3
#dropout of .2
#define loss as in equation 3 of Weigert et al.
#minimize loss
#plot training error and PSNR



#Testing module-------------------------------------------------------
#load model from training module

#use axial images as inputs

#display original axial images
#display corresponding super-resolution corrected axial images

#for prototyping/validation:
#apply blur to unseen lateral images
#use blurred images as input
#score output on PSNR

