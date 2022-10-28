# ImageClassification by using transfer learning

## Dataset preparation

In order to use the k-folded methodology for image classification, is very important to generate a context filename "dataset.csv". 

## Usage

1. Take several pictures for every single class.
2. Store them per folder class name.
3. Activate your python environment, and then, run next command: python dataset_preparation.py
4. Enters the dataset training parent directory, where all the class folders are stored, e.g. "c:\users\mlops\data\Tranining"
5. Once script running is completed, a dataset,csv file has been generated. Which will be utilized to feed the Jupyter Notebook for Image Classifier k-folded

# Notes:
The main model utilized by using transfer learning is InceptionV3, with maximum input shape of (299, 299) see more details in keras.applications for more models.
