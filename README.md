# Custom-SVM-kernel
The repo builds upon the Neural Architecture Search(NAS) technique to find a kernel function for the SVM over MNIST dataset.  
1000 samples are used to train the SVM while the reward for the controller is the accuracy over a validation set of 500 samples.  

The discovered kernel function is :-
```
k(x,y) = ||min{sin(x*y), sin(<x,y>/gamma)}|| 
```
Here (x*y) denotes the elementise product, <x,y> denotes the dot product of vectors and the norm is L-1 norm. Also, gamma is a constant, that is equal to the number of features in the data(784 for MNIST).

The final results below are the accuracy for the entire 10000 test samples of MNIST.  

|  Kernel function |   Validation Acc.   |  Test Acc.    |
|------------|----------|----------|
|     Linear    |  89.0    |   87.96   |
|     RBF   |  86.4  |  82.76   |
|     Sigmoid   |  81.0 |   74.42  |
|  Discovered kernel | ** 90.2 ** | ** 91.01 ** |
