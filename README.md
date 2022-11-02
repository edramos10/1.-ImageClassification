# ImageClassification by using transfer learning

## Dataset_preparation.py

In order to use the k-folded methodology for image classification, is very important to generate a context filename "dataset.csv". 

## Usage

1. Take several pictures for every single class.
2. Store them per folder class name:<br/>
  ../Training/<br/>
      /Class1<br/>
      /Class2<br/>
      /Class3<br/>
      /Class4<br/>
     .<br/>
     .<br/>
     ./ClassN<br/>
    
   ../Testing<br/>
      /Class1<br/>
      /Class2<br/>
      /Class3<br/>
      /Class4<br/>
      .<br/>
      .<br/>
      ./ClassN/<br/>
3. Activate your python environment, and then, run next command: python dataset_preparation.py
4. Enters the dataset training parent directory, where all the class folders are stored, e.g. "c:\users\mlops\data\Training"
5. Once script running is completed, a dataset,csv file has been generated. Which will be utilized to feed the Jupyter Notebook for Image Classifier k-folded

## Training_notebook_mlflow.ipynb
### Prework:
1. Once mlflow-extend and mlflow is installed within your environment, activate the environment, e.g., conda activate [environment].
2. Go to your directory of repository.
3. Run the mlflow server thru: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 1234
4. Your MLflow server is activated and will monitor the entire traning, evaluation and registry model. Open a browser and type http://127.0.0.1:1234.


# Notes:
## 1 The transfer learning model is InceptionV3, with maximum input shape of (299, 299) see more details in keras.applications for more models.
## 2 You must install Anaconda or miniconda or python pyenv, I prefer to use anaconda/miniconda, since these packages has more available libraries with conda cmd
## 3 Once, you download the repository, and invironment has been created, you can run pip -r requiriements.txt in order to install all the dependencies needed to run the ImageClassification Training phase
