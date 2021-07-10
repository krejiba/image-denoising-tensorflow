# Image denoising | Tensorflow
This is a personal project using Python. We implement a denoising autoencoder using Tensorflow.

### How to use:
You need to download all the files in your working directory. Make sure to install the packages listed below in order to make use of this code.  
We used a subset of the NIH Chest X-ray database available on [Kaggle](https://www.kaggle.com/nih-chest-xrays/sample). The metadata can be found in `sample_labels.csv`. Make sure to download the images and the CSV file and store them in the same directory (for example `data/nih-cxr-sample/`).  
You need to have Jupyter Notebook installed on your system. You can use the notebook by typing `jupyter notebook demo.ipynb` in the command line. A new window will open in your browser.  
You can run all code by clicking on "Cell" -> "Run All".  

----
Python 3.7.4  
NumPy 1.19.5  
Pandas 1.2.0  
Pillow 6.1.0  
Tensorflow 2.5.0  

### References:
L. Gondara. (2016). “Medical image denoising using convolutional denoising autoencoders.” [Online]. Available: https://arxiv.org/abs/1608.04667  
I. Goodfellow, Y. Bengio, and A. Courville. (2016). "Deep Learning.” [Online]. Available: https://www.deeplearningbook.org/
