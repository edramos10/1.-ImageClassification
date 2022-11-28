import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from mlflow_extend import mlflow
import gc

from tensorflow import keras 
from keras.applications import InceptionV3, Xception, ResNet152V2,InceptionResNetV2, MobileNetV2, EfficientNetB0, ResNet50,ResNet101 
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold ,KFold ,RepeatedKFold, train_test_split
from sklearn.metrics import classification_report, roc_curve,precision_recall_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

class dataset_preparation:
    def generate_csv(DATASET_DIR, TRAIN_PATH):
        '''Genera archivo CSV con la ubicacion de las imagenes y sus respectivas clases.'''
        os.chdir(TRAIN_PATH)

        folders =os.listdir(TRAIN_PATH)
        print(folders)
        files = []
        classes = []

        for folder in folders:
            classes.append(folder)
            for file in os.listdir(folder):
                files.append([TRAIN_PATH+"/"+folder+"/"+file, folder])
            pd.DataFrame(files, columns=['Image', 'Class']).to_csv(DATASET_DIR+'/dataset.csv',index=False)

        return classes

    def Image_augmentation(rotation, shear, zoom):
        '''Define parametros para la aumentacion de imagenes.'''
        datagen_kwargs = dict(rescale=1./255,
        rotation_range=rotation,
        shear_range=shear,
        zoom_range=zoom,
        brightness_range=[0,1],
        horizontal_flip=True)

        return datagen_kwargs

    def Create_Images_for_training(DATASET_DIR,TRAIN_PATH,IMG_SIZE,**datagen_kwargs):
        os.chdir(DATASET_DIR)
        IMAGE_SHAPE=(IMG_SIZE, IMG_SIZE)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
        train_generator = train_datagen.flow_from_directory(TRAIN_PATH, batch_size=32, subset="training", seed=42, shuffle=True, target_size=IMAGE_SHAPE)
        for image_batch, label_batch in train_generator:
            break
        print(image_batch.shape, label_batch.shape)
        image_batch_train, label_batch_train = next(iter(train_generator))
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
        valid_generator = valid_datagen.flow_from_directory(TRAIN_PATH, subset="validation", shuffle=True, target_size=IMAGE_SHAPE)
        val_image_batch, val_label_batch = next(iter(valid_generator))
        print("Image batch shape: ", image_batch_train.shape)
        print("Label batch shape: ", label_batch_train.shape)
        dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
        dataset_labels = np.array([key.title() for key, value in dataset_labels])
        print(dataset_labels)
        
        labels = '\n'.join(sorted(train_generator.class_indices.keys()))
        with open('labels.txt', 'w') as f:
            f.write(labels)

class training:
    def get_model(IMG_SIZE, Number_classes):
        '''Define arquitectura del modelo.'''
        METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.SpecificityAtSensitivity(0.5),
            tf.keras.metrics.SensitivityAtSpecificity(0.5),
        ]

        base_model =applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        add_model.add(Dropout(0.3))
        add_model.add(Dense(64, activation='relu'))
        add_model.add(Dropout(0.4))
        add_model.add(Dense(Number_classes, activation='softmax'))
        model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9), metrics=METRICS)
        return model

    def mlflow_train(PROJECT_DIR, DATASET_DIR, TESTING_DATA_DIR, SERVER, HOST, EXPERIMENT_NAME, classes, IMG_SIZE, EPOCHS, BATCH_SIZE, datagen_kwargs):
        '''Entrena el modelo y lo registra en mlflow.'''
        os.chdir(PROJECT_DIR)

        Number_classes = len(classes)
        N_SPLIT = Number_classes+1

        train = pd.read_csv(DATASET_DIR + '/dataset.csv')

        # As we are going to divide dataset
        df = train.copy()

        # Increasing the size of dataset
        clase = []
        for label in classes:
            i = 0
            clase.append(train[train["Class"]==label])
            df = pd.concat([df,clase[i]])
            i+=1

        # Creating X, Y for training 
        train_y = df.Class
        train_x = df.drop(['Class'],axis=1)
        
        mlflow.set_tracking_uri(SERVER + ":" + HOST)

        mlflow.set_experiment(EXPERIMENT_NAME)
        experiment=mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        print("experiment_id:", experiment.experiment_id)
        mlflow.tensorflow.autolog()

        with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=("Run: " + EXPERIMENT_NAME)):
    
            train_datagen = ImageDataGenerator(**datagen_kwargs)
            validation_datagen = ImageDataGenerator(**datagen_kwargs)

            # k-fold
            kfold = StratifiedKFold(n_splits=N_SPLIT,shuffle=True,random_state=42)

            # Variable for keeping count of split we are executing
            j = 0

            # K-fold Train and test for each split
            for train_idx, val_idx in list(kfold.split(train_x,train_y)):
                x_train_df = df.iloc[train_idx]
                x_valid_df = df.iloc[val_idx]
                j+=1

                training_set = train_datagen.flow_from_dataframe(dataframe=x_train_df, train_dir=None, x_col="Image", y_col="Class",shuffle=True, class_mode="categorical",validate_filenames=True, 
                                                                target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE)
                
                validation_set = validation_datagen.flow_from_dataframe(dataframe=x_valid_df, train_dir=None, x_col="Image", y_col="Class",shuffle=False ,class_mode="categorical",validate_filenames=True,
                                                                        target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE)

                model_test = training.get_model(IMG_SIZE,Number_classes)
                
                history = model_test.fit( training_set, validation_data=validation_set, epochs = EPOCHS, steps_per_epoch=(x_train_df.shape[0])/BATCH_SIZE)  
                gc.collect()
                
            testing.test(model_test, TESTING_DATA_DIR)
            mlflow.end_run()

class testing:
    def testGen(TESTING_DATA_DIR, IMG_SIZE):
        '''Genera directorio para pruebas.'''
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
        test_data_dir=TESTING_DATA_DIR
        Test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(IMG_SIZE,IMG_SIZE),
                batch_size=1,
                class_mode='categorical',
                shuffle=False)
                
        return Test_generator

    def test(loaded_model, TESTING_DATA_DIR):
        '''Calcula y registra las metricas en mlflow'''
        Images4ValidationGenerator=testing.testGen(TESTING_DATA_DIR,IMG_SIZE)
        y_test=Images4ValidationGenerator.classes
        class_labels = Images4ValidationGenerator.class_indices
        class_labels = {v: k for k, v in class_labels.items()}
        classes = list(class_labels.values())
        y_true = Images4ValidationGenerator.classes
        Y_pred = loaded_model.predict(Images4ValidationGenerator)
        y_pred = np.argmax(Y_pred, axis=1)

        conf_mat = confusion_matrix(y_true,y_pred)
        conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        fig1=plt.figure(figsize = (10,10))
        sns.heatmap(conf_mat, annot=True,fmt='d',cmap="Blues")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='macro')  
        test_precision = precision_score(y_true, y_pred, average='macro')
        test_recall = recall_score(y_true, y_pred, average='macro')
        print("Test Accuracy: {0:.2f}".format(test_acc))
        print("Test f1: {0:.2f}".format(test_f1))
        print("Test Precision: {0:.2f}".format(test_precision))
        print("Test Recall: {0:.2f}".format(test_recall))
        mlflow.log_metric("Test Accuracy", test_acc)
        mlflow.log_metric("Test F1", test_f1)
        mlflow.log_metric("Test Precision", test_precision)
        mlflow.log_metric("Test Recall", test_recall)
    

        # get confusion matrix values
        conf_matrix = confusion_matrix(y_true,y_pred)
        true_positive = conf_matrix[0][0]
        true_negative = conf_matrix[1][1]
        false_positive = conf_matrix[0][1]
        false_negative = conf_matrix[1][0]

        mlflow.log_metric("true_positive", true_positive)
        mlflow.log_metric("true_negative", true_negative)
        mlflow.log_metric("false_positive", false_positive)
        mlflow.log_metric("false_negative", false_negative)
        mlflow.log_confusion_matrix(conf_matrix)

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

if __name__=="__main__":
    physical_devices=tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    PROJECT_DIR='G:\Public\DeepLearning\TOUCHPADS'#'C:/Users/anavarro4/Documents/mlprojects' #Ubicacion para guardar la carpeta artifacts
    DATASET_DIR='G:\Public\DeepLearning\TOUCHPADS\Dataset'#'C:/Users/anavarro4/Documents/mlprojects/Dataset' #Ubicacion para guardar dataset.csv y labels.txt
    TRAIN_PATH='G:\Public\DeepLearning\TOUCHPADS\Dataset\Train'#'C:/Users/anavarro4/Documents/mlprojects/Dataset/PCB_Train'
    TESTING_DATA_DIR='G:\Public\DeepLearning\TOUCHPADS\Dataset\Test'#'C:/Users/anavarro4/Documents/mlprojects/Dataset/PCB_Test'
    IMG_SIZE = 299
    EPOCHS = 5
    BATCH_SIZE = 8
    EXPERIMENT_NAME='KEYPADSv2'
    SERVER = 'http://127.0.0.1'
    HOST = '1234'

    classes = dataset_preparation.generate_csv(DATASET_DIR, TRAIN_PATH)

    rotation_range=10
    shear_range=0.1
    zoom_range=0.1
    datagen_kwargs = dataset_preparation.Image_augmentation(rotation_range, shear_range, zoom_range)

    training.mlflow_train(PROJECT_DIR, DATASET_DIR, TESTING_DATA_DIR, SERVER, HOST, EXPERIMENT_NAME, classes, IMG_SIZE, EPOCHS, BATCH_SIZE, datagen_kwargs)


    