# ImageClassification by using transfer learning

## Dataset_preparation.py

In order to use the k-folded methodology for image classification, is very important to generate a context filename "dataset.csv". 

## Usage

1. Take several pictures for every single class.
2. Store them per folder class name.
3. Activate your python environment, and then, run next command: python dataset_preparation.py
4. Enters the dataset training parent directory, where all the class folders are stored, e.g. "c:\users\mlops\data\Tranining"
5. Once script running is completed, a dataset,csv file has been generated. Which will be utilized to feed the Jupyter Notebook for Image Classifier k-folded

## Training_notebook_mlflow.ipynb



# Notes:
## 1 The transfer learning model is InceptionV3, with maximum input shape of (299, 299) see more details in keras.applications for more models.
## 2 You must install Anaconda or miniconda or python pyenv, I prefer to use anaconda/miniconda, since these packages has more available libraries with conda cmd
## 3 Once, you download the repository, and invironment has been created, you can run pip -r requiriements.txt in order to install all the dependencies needed to run the ImageClassification Training phase
