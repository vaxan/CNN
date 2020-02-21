# CNN
Sample IRIS classification of flower image data set using CNN classifier

The flower dataset which consists of images of 5 different flowers is split into training, validation and testing (0.6, 0.2, 0.2) ratios respectively making use of python library split_folder. Network is build using sequential model of Cov2d of 3 dense layer 32, 68, 128. To prevent overfitting a dropout of 0.5 has been added to the network. Relu activation used for the layers and for the final output to classify into 5 classes softmax activation function is used. In model compilation adam(Adaptive moment estimation) optimizer to update weights and bias iterative based. Data augmentation is done using imagedatagenerator.

