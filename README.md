# Improving Classification Performance of Support Vector Machines via Guided Custom Kernel Search
This work builds upon the Neural Architecture Search (NAS) technique to find a custom kernel function for the SVM over MNIST dataset.  
An RNN controller emits a kernel function for SVM. The SVM with the emitted kernel function is trained over 1000 MNIST training samples and the accuracy over a seperate 500 validation samples is used as a reward signal for the RNN controller. The RNN controller is trained via the vanilla policy gradient algorithm.

The final discovered kernel function by the controller is :-
```
k(x,y) = ||min{sin(x*y), sin(<x,y>/gamma)}|| 
```
Here (x*y) denotes the elementise product, <x,y> denotes the dot product of vectors and the norm is L-1 norm. Also, gamma is a constant, that is equal to the number of features in the data(784 for MNIST).

The results below are the accuracy for the entire 10000 test samples of MNIST when the SVM is trained with diffferent kernel functions over 1000 training samples.

|  Kernel function |   Validation Acc.   |  Test Acc.    |
|------------|----------|----------|
|     Linear    |  89.0    |   87.96   |
|     RBF   |  86.4  |  82.76   |
|     Sigmoid   |  81.0 |   74.42  |
|  Discovered kernel | **90.2** | **91.01** |


To train the RNN controller that learns the custom kernel function run - 
```
python3 mnist_kernel_search.py
```

To evaluate the trained model uncomment the two lines [here](https://github.com/neuralCollab/Custom-SVM-kernel/blob/master/mnist_kernel_test_final.py#L295) to restore your trained model and then run - 

```
python3 mnist_kernel_test_final.py
```
Running mnist_kernel_test_final.py currently will show the results corresponding to the above discovered custom kernel function.

Even though the RNN controller has been trained to learn a kernel function that best fits the 1000 training samples, the discovered kernel function works better than the other kernel functions even when it is used to fit 2000 training samples.
The below are the test accuracy results when the different kernel functions are used to train SVM over 2000 samples.

|  Kernel function | Test Acc.    |
|------------|----------|
|     Linear |    89.87   |
|     RBF   |  87.85   |
|     Sigmoid  |  84.94  |
|  Discovered kernel |  **93.12** |
