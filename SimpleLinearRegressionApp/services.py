import io
import json
import ntpath
from http.client import HTTPResponse
import os
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import generics, status
from rest_framework.decorators import action, api_view
from rest_framework.views import APIView

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.models import save_model

from .serializers import SLRSerializer
from .models import *
from .permissions import IsOwner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.core.files.storage import default_storage
import pickle
import os
######################################################################

class LogisticRegressionModel(APIView):
    permission_classes = [IsOwner,]

    def post(self, request):

        #try:
        # Logistic Regression
        # Importing the dataset
        # Read the csv files:
        parser_classes = [JSONParser, MultiPartParser, FormParser]
        my_file = request.FILES['file']
        dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Classification\\ana - Binary (True or False)\\Churn_Modelling.csv'

        global dataset
        # Excel Extensions
        xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

        # Reading CSV Files
        if (dataset_path[-3:] == 'csv'):
            try:
                dataset = pd.read_csv(my_file)
            except (OSError, FileNotFoundError):
                print("Error Reading File!")

        # Reading excel files
        for file_extension in xl_extensions:
            if (dataset_path[-3:] == file_extension):
                try:
                    dataset = pd.read_excel(my_file)
                except OSError:
                    print("Error Reading File!")
        # firstrow = list(dataset.columns.values)

        X = dataset.iloc[:, :-1].values
        # y is the binary column (yes/no)
        y = dataset.iloc[:, -1].values

        # taking care of missing data
        if (dataset.isnull().values.any()):
            dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

        # Encoding Categorical Data    : converting categories (strings) to numbers

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        x_indpndnt_v_encoder = pd.get_dummies(X)
        X = np.array(x_indpndnt_v_encoder)

        y_dpndnt_v_encoder = pd.get_dummies(y)
        y = np.array(y_dpndnt_v_encoder)



        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)


        # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X, np.ravel(y))


        # Predicting the Test set results
        # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
        y_pred = classifier.predict(X)
        framed_ypred = (pd.DataFrame(y_pred))

        # predicting a single observation/row
        row_1 = np.expand_dims(y_pred[0], axis=0)
        print("Row Number 10 prediction is:", row_1)

        # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
        #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

        filepath, filename = ntpath.split(dataset_path)

        # Saving the model
        saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\LogisticRegressionModel.pkl"
        #Creating an object

        with open(SLRModel.user_folder.path, "wb") as weightsfolder:
            pickle.dump(classifier, weightsfolder)
        #print(new_weights)

        """
        # Iterating & Getting the name of the POST'ed file
        for myfilename, file in request.FILES.iteritems():
            name = request.FILES[myfilename].name
            the_file = request.FILES[myfilename]
            print("File name: ", name, "file: ", the_file)
        """
        response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
        return Response(response_dict, status=status.HTTP_201_CREATED)


        #except Exception as err:
        #return Response("Error ! please make sure everything is configured.")



###############################################################################
class LinearRegression(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Linear Regression
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Classification\\ana - Binary (True or False)\\Churn_Modelling.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)



            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.linear_model import LinearRegression

            classifier = LinearRegression()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))
            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response = "y_pred: {}".format(framed_ypred)
            return Response(response, status = status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


#################################################################################################

class RandomForestClassifier(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Random Forest Classifier
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.ensemble import RandomForestClassifier

            classifier = RandomForestClassifier()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status= status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


##############################################################################
class RandomForestRegressor(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Random Forest Regressor
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.ensemble import RandomForestRegressor

            classifier = RandomForestRegressor()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")



##########################################################################
class DecisionTreeClassifier(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Decision Tree Classifier
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.tree import DecisionTreeClassifier

            classifier = DecisionTreeClassifier()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[y_pred], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


##########################################################################
class DecisionTreeRegressor(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Decision Tree Regressor
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)

            # Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            # print(X_train)
            # print(y_train)
            # print(X_test)
            # print(y_test)

            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X_train)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.tree import DecisionTreeRegressor

            classifier = DecisionTreeRegressor()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")




############################################################################
class SupportVectorMachineClassifier(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Support Vector Machine Classifier
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)



            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)

            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.svm import SVC

            classifier = SVC()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


############################################################################
class SupportVectorMachineRegressor(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Support Vector Machine Regressor
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.svm import SVR

            classifier = SVR()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


##############################################################################
class KNearestNeighborClassifier(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # KNearest Neighbor Classifier
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)



            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.neighbors import KNeighborsClassifier

            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)


        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


#############################################################################
class KNearestNeighborRegressor(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # KNearest Neighbor Regressor
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the regressor will be the only thing that will be changed--
            from sklearn.neighbors import KNeighborsRegressor

            regressor = KNeighborsRegressor(n_neighbors=5)
            regressor.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = regressor.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(regressor, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")

#############################################################################
class NaiveBayesGaussian(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):
        try:

            # Naive Bayes Gaussian
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)



            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.naive_bayes import GaussianNB

            classifier = GaussianNB()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


#############################################################################

class NaiveBayesBernoulli(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Naive Bayes BernoulliNB
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\Logistic Regression Parent\\Simple Linear Regression\\Social_Network_Ads.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")
            # firstrow = list(dataset.columns.values)

            X = dataset.iloc[:, :-1].values
            # y is the binary column (yes/no)
            y = dataset.iloc[:, -1].values

            framed_x = pd.DataFrame(X)
            framed_y = pd.DataFrame(y)

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # second & third columns are numbers, so we exclude them

            # counting columns
            column_count = 0
            for column in dataset:
                column_count = column_count + 1

            # Encoding Categorical Data    : converting categories (strings) to numbers

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(X)
            X = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)


            # Training the Logistic Regression model on the Training set        --The name of the classifier will be the only thing that will be changed--
            from sklearn.naive_bayes import BernoulliNB
            classifier = BernoulliNB()
            classifier.fit(X, np.ravel(y))


            # Predicting the Test set results
            # Note: y_pred are the predictions for the last column (our dependent variable, that we are evaluating)
            y_pred = classifier.predict(X)
            framed_ypred = (pd.DataFrame(y_pred))

            # predicting a single observation/row
            row_1 = np.expand_dims(y_pred[0], axis=0)
            print("Row Number 10 prediction is:", row_1)

            # The following line is optional, thinking of removing it, it shows us the prediction column next to the actual column
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            #tf.keras.models.save_model(classifier, saving_path)

            response_dict = {'y_pred: {} row_number 1: {}'.format(framed_ypred, row_1)}
            return Response(response_dict, status=status.HTTP_201_CREATED)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")



#############################################################################
##ANN (Regression)

class ANNRegression(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):

        try:
            # Electricity Costs Prediction of a factory
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Regression\\ana - SingleOutput Regression\\Folds5x2_pp.csv'

            # filepath, filename = ntpath.split(dataset_path)

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")



            # Splitting the dataset into a Dependent variable & Independent Variable

            x = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values

            # print(x)
            # print(y)

            # taking care of missing data
            if (dataset.isnull().values.any()):
                dataset.fillna(dataset.mean())  # fills the empty cells with the mean of the whole column

            '''3) Encoding Categorical Data --> we will not doit here,
                    because there are no text columns, for column headers to be text is ok'''
            column_count = 0
            for column in dataset:
                column_count = column_count + 1

            x = pd.DataFrame(x)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(x)
            x = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # 5)Feature scaling  --- Manadatory in Deep Learning ---
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler()
            x = sc.fit_transform(x)


            # Part2: Building the Artificial Neural Network
            # 6) Initialize the ANN as a sequence of layers
            ann = Sequential()

            # 7) Adding the Input Layer (First Hidden Layer)  - NOTE: each Neuron corresponds to a column in the dataset
            num_neurons = 0
            if (column_count <= 3):
                num_neurons = column_count
            else:
                num_neurons = int(column_count // 2)

            ann.add(Dense(units= num_neurons, activation='relu'))

            # 8) Adding the Second Hidden Layer
            ann.add(Dense(units= num_neurons, activation='relu'))

            # 9) Adding the Output Layer
            '''
            - NOTE: units= 1 because the dependent variable ('Exited' column is either a 0 or 1) so 1 neuron is enough
            '''

            ann.add(Dense(units=1))

            # part 3: Training the ANN
            # 10) Compiling the ANN

            # metrics are what metrics we want to evaluate our ANN against, here we are saying accuracy
            ann.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

            # 11) Training the ANN on the Training set
            '''
                fit: means Train
                batch_size: this is for batch learning, (comparing predictions in batch to actual
                            results in batch, instead of comparing one by one.
                epochs: A Neural Network has to be trained over a number of epochs (rounds)
                        to improve accuracy. (should be any big number)
            '''
            learn_control = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.2, mode='min', min_lr=0.001)
            ann.fit(x, np.ravel(y), batch_size=32, epochs=100, callbacks=[learn_control])

            # part 4: Making the predictions and evaluating the model
            # 12) no need for this step, we are not predicting a single observation
            # 13) predicting the test result (comparing Predicted Results with Test Results)
            y_pred = ann.predict(x)
            framed_ypred = (pd.DataFrame(y_pred))

            np.set_printoptions(precision= 2)  # if y_pred > 0.5 (result will be 0:False) - if y_pred < 0.5 (result will be 1:True)
            #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
            # notice the last 1 in the above line, its 1, because we want to do vertical concatenation, we type 0 if we want Horizontal concatenation

            # Lets get a summary of our model
            ann.summary()

            filepath, filename = ntpath.split(dataset_path)

            # Saving the model
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            tf.keras.models.save_model(ann, saving_path)

            response_dict = "y_pred: {}".format(framed_ypred)
            return Response(response_dict)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")




class ANNBinaryClassification(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):


        # Importing the dataset
        # Read the csv files:

        dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Classification\\ana - Binary (True or False)\\Churn_Modelling.csv'



        global dataset
        # Excel Extensions
        xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

        # Reading CSV Files
        if (dataset_path[-3:] == 'csv'):
            try:
                dataset = pd.read_csv(dataset_path)
            except (OSError, FileNotFoundError):
                print("Error Reading File!")

        # Reading excel files
        for file_extension in xl_extensions:
            if (dataset_path[-3:] == file_extension):
                try:
                    dataset = pd.read_excel(dataset_path)
                except OSError:
                    print("Error Reading File!")


        x = dataset.iloc[:, :-1].values
        # Dependent Variable:
        y = dataset.iloc[:, -1].values

        # print(x)
        # print(y)

        # taking care of missing data
        if (dataset.isnull().values.any() == True):
            dataset.fillna(dataset.mean())  # whenever it finds empty columns are numbers, it fills them with its mean



        column_count = 0
        for column in dataset:
            column_count = column_count + 1

        # converting to onehotencoding all the datatypes of type 'object'
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

        x_indpndnt_v_encoder = pd.get_dummies(x)
        x = np.array(x_indpndnt_v_encoder)

        y_dpndnt_v_encoder = pd.get_dummies(y)
        y = np.array(y_dpndnt_v_encoder)


        # 5)Feature scaling  --- Manadatory in Deep Learning ---
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x = sc.fit_transform(x)


        # Part2: Building the Artificial Neural Network
        # 6) Initialize the ANN as a sequence of layers
        ann = tf.keras.models.Sequential()

        # 7) Adding the Input Layer (First Hidden Layer)  - NOTE: each Neuron corresponds to a column in the dataset
        num_neurons = 0
        if (column_count <= 3):
            num_neurons = column_count
        else:
            num_neurons = int(column_count // 2)

        ann.add(tf.keras.layers.Dense(units= num_neurons, activation='relu'))

        # 8) Adding the Second Hidden Layer
        ann.add(tf.keras.layers.Dense(units= num_neurons, activation='relu'))

        # 9) Adding the Output Layer
        '''- NOTE: units= 1 because the dependent variable ('Exited' column is either a 0 or 1) so 1 neuron is enough
        - NOTE: activation function = 'sigmoid', it tells us not only the predictions, but also the the
                probability of whether a customer will leave the bank or not'''
        # the units will change depending on how many dependent variable column(s) the user selects, here its only 1 column
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # part 3: Training the ANN
        # 10)Compiling the ANN
        '''NOTE: -for Binary Classification (2 or less categories): 
                    loss should be equal to: 'binary_crossentropy'
                    activation should be equal to: 'sigmoid'
                 -for Non-Binary Classification (2 or more categories):
                    loss should be equal to: 'categorical_crossentropy'
                    activation should be equal to: 'softmax'          '''
        # metrics are what metrics we want to evaluate our ANN against, here we are saying accuracy

        ann.compile(optimizer= 'sgd', loss='binary_crossentropy', metrics=['accuracy'])

        # 11 Training the ANN on the Training set

        learn_control = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.2,mode= 'min', min_lr=0.001)

        ann.fit(x, np.ravel(y), batch_size=32, epochs=100, callbacks=[learn_control])

        # part 4: Making the predictions and evaluating the model
        # 12) Predicting the result of a single observation
        '''the 0.5 is OPTIONAL, this says does the probability of the customer 
            leaving the bank higher than 0.5, if we removed '>0.5' it will give us 
            the actual probability instead of True or False 

            Customer sample: 
              country: france, credit score: 600, Gender: Male, Age: 40, Tenure: 3, Balance: $6000
              No.of products: 2, customer has credit card: YES, Active customer: YES, Salary:$50000     
            '''

        # 13) predicting the test result (comparing Predicted Results with Test Results)
        #Making it Binary

        y_pred = ann.predict(x)
        framed_ypred = (pd.DataFrame(y_pred))

        # Making a single observation
        row_1 = np.expand_dims(y_pred[0], axis=0)
        print("row number 10 prediction is:", row_1)  # predicting row number 10
        y_pred_grea_prob = y_pred[y_pred > 0.5] = 1
        y_pred_less_prob = y_pred[y_pred <= 0.5] = 0
        # if y_pred > 0.5 (result will be 0:False) - if y_pred < 0.5 (result will be 1:True)
        #print(y_pred)

        # This following line is just for having both predictions next to each other as columns
        #print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
        # notice the last 1 in the above line, its 1, because we want to do vertical concatenation, we type 0 if we want Horizontal concatenation


        filepath, filename = ntpath.split(dataset_path)

        #Saving the model
        saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
        #tf.keras.models.save_model(ann, saving_path)



        response_dict = "y_pred: {}, y_pred (probability) > 0.5: {}".format(framed_ypred, y_pred_grea_prob)
        return Response(response_dict)

        ######################################################################



class ANNCategoricalClassification(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):
        #try:


        # Importing the dataset
        # Read the csv files:

        dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Classification\\ana - Binary (True or False)\\Churn_Modelling.csv'



        global dataset
        # Excel Extensions
        xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

        # Reading CSV Files
        if (dataset_path[-3:] == 'csv'):
            try:
                dataset = pd.read_csv(dataset_path)
            except (OSError, FileNotFoundError):
                print("Error Reading File!")

        # Reading excel files
        for file_extension in xl_extensions:
            if (dataset_path[-3:] == file_extension):
                try:
                    dataset = pd.read_excel(dataset_path)
                except OSError:
                    print("Error Reading File!")


        x = dataset.iloc[:, :-1].values
        # Dependent Variable:
        y = dataset.iloc[:, -1].values

        # print(x)
        # print(y)

        # taking care of missing data
        if (dataset.isnull().values.any() == True):
            filled = dataset.fillna(
                dataset.mean())  # whenever it finds empty columns are numbers, it fills them with its mean

        column_count = 0
        for column in dataset:
            column_count = column_count + 1

        # Data Encoding
        # converting to onehotencoding all the datatypes of type 'object'
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

        x_indpndnt_v_encoder = pd.get_dummies(x)
        x = np.array(x_indpndnt_v_encoder)

        y_dpndnt_v_encoder = pd.get_dummies(y)
        y = np.array(y_dpndnt_v_encoder)


        # 5)Feature scaling  --- Manadatory in Deep Learning ---
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x = sc.fit_transform(x)


        # Part2: Building the Artificial Neural Network
        # 6) Initialize the ANN as a sequence of layers
        ann = tf.keras.models.Sequential()

        # 7) Adding the Input Layer (First Hidden Layer)  - NOTE: each Neuron corresponds to a column in the dataset
        num_neurons = 0
        if(column_count <=3):
            num_neurons = column_count
        else:
            num_neurons = int(column_count // 2)

        ann.add(tf.keras.layers.Dense(units=num_neurons, activation='relu'))

        # 8) Adding the Second Hidden Layer
        ann.add(tf.keras.layers.Dense(units=num_neurons, activation='relu'))

        # 9) Adding the Output Layer
        '''- NOTE: units= 1 because the dependent variable ('Exited' column is either a 0 or 1) so 1 neuron is enough
        - NOTE: activation function = 'sigmoid', it tells us not only the predictions, but also the the
                probability of whether a customer will leave the bank or not'''

        # the units will change depending on how many dependent variable column(s) the user selects, here its only 1 column
        # determining the number of neurons automatically based on the
        # values in the column, excluding repeated values.

        dpndnt_vrbl_output_neurons = pd.DataFrame(y).nunique()
        print("Output Neurons Count", dpndnt_vrbl_output_neurons)

        # Output Layer
        ann.add(tf.keras.layers.Dense(units=dpndnt_vrbl_output_neurons, activation='softmax'))
        print(dpndnt_vrbl_output_neurons)
        # part 3: Training the ANN
        # 10)Compiling the ANN

        # metrics are what metrics we want to evaluate our ANN against, here we are saying accuracy
        ann.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 11 Training the ANN on the Training set
        learn_control = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.2, mode='min', min_lr=0.001)

        ann.fit(x, np.ravel(y), batch_size=32, epochs=100, callbacks=[learn_control])

        # part 4: Making the predictions and evaluating the model
        # 12) Predicting the result of a single observation
        '''the 0.5 is OPTIONAL, this says does the probability of the customer 
            leaving the bank higher than 0.5, if we removed '>0.5' it will give us 
            the actual probability instead of True or False 

            Customer sample: 
              country: france, credit score: 600, Gender: Male, Age: 40, Tenure: 3, Balance: $6000
              No.of products: 2, customer has credit card: YES, Active customer: YES, Salary:$50000     
            '''
        # ann.predict(sc.transform(([[1,0,0, 600,1,40,3, 6000, 2, 1, 1, 50000]])))
        # ann.predict(x_test)

        # 13) predicting the test result (comparing Predicted Results with Test Results)
        y_pred = ann.predict(x)


        #Framing data
        framed_ypred = (pd.DataFrame(y_pred))

        # Making a single observation
        row_1 = np.expand_dims(y_pred[0], axis=0)
        print("row number 10 prediction is:", row_1)  # predicting row number 10
        # y_pred = (y_pred[10] > 0.5)   # if y_pred > 0.5 (result will be 0:False) - if y_pred < 0.5 (result will be 1:True)
        # print(y_pred)

        # This following line is just for having both predictions next to each other as columns
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

        #Saving the model
        filepath, filename = ntpath.split(dataset_path)

        # Saving the model
        saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
        tf.keras.models.save_model(ann, saving_path)

        response_dict = "y_pred: {}".format(framed_ypred)
        return Response(response_dict, status= status.HTTP_201_CREATED)

        #except Exception as err:
        #return Response("Error ! please make sure everything is configured.")



class TimeSeries(APIView):
    permission_classes = [IsOwner,]
    def post(self, request):
        #try:
        # Importing the training set

        # Importing the dataset
        # Read the csv files:

        dataset_path = 'D:\\My AI Projects\\BusinessMan\\Recurrent Neural Network\\Part 3 - Recurrent Neural Networks\\Google_Stock_Price_Train.csv'

        global dataset
        # Excel Extensions
        xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

        # Reading CSV Files
        if (dataset_path[-3:] == 'csv'):
            try:
                dataset = pd.read_csv(dataset_path)
            except (OSError, FileNotFoundError):
                print("Error Reading File!")

        # Reading excel files
        for file_extension in xl_extensions:
            if (dataset_path[-3:] == file_extension):
                try:
                    dataset = pd.read_excel(dataset_path)
                except OSError:
                    print("Error Reading File!")
        dataset_train = pd.read_csv(dataset_path)
        #range of columns selected in the dataset

        # might change the ranges later to make it take the entire dataset
        training_set = dataset_train.iloc[:, :].values
        training_set = pd.DataFrame(training_set)
        #y = dataset_train.iloc[:, -1].values
        print(training_set.shape)

        #Time-steps to be predicted
        past_days = 60
        num_predictors = 1  #1 column
        future_days = 20



        # taking care of missing data
        if (dataset_train.isnull().values.any()):
            dataset_train.fillna(dataset_train.mean())  # fills empty cells with the mean of all values in the column


        column_count = 0
        for column in dataset_train:
            column_count = column_count + 1

        # Data Encoding
        # converting to onehotencoding all the datatypes of type 'object'

        x_indpndnt_v_encoder = pd.get_dummies(training_set)
        x = np.array(x_indpndnt_v_encoder)

        #y_dpndnt_v_encoder = pd.get_dummies(y)
        #y = np.array(y_dpndnt_v_encoder)

        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(x)

        ### Creating a data structure with 60 timesteps and 1 output

        X_train = []
        y_train = []
        for i in range(60, 1258):
            X_train.append(training_set_scaled[i - past_days:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_predictors))
        # except Exception as err:
        # return Response("Error Reshaping Array!", status= status.HTTP_400_BAD_REQUEST)
        ## Part 2 - Building and Training the RNN

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import Dropout

        ### Initialising the RNN

        regressor = Sequential()

        ### Adding the first LSTM layer and some Dropout regularisation

        regressor.add(LSTM(units=50, input_shape=(X_train.shape[1:]), return_sequences=True))
        regressor.add(Dropout(0.2))

        ### Adding a second LSTM layer and some Dropout regularisation

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        ### Adding a third LSTM layer and some Dropout regularisation

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        ### Adding a fourth LSTM layer and some Dropout regularisation

        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))

        ### Adding the output layer

        regressor.add(Dense(units=1))

        ### Compiling the RNN

        regressor.compile(optimizer='adam', loss='mean_squared_error')

        ### Fitting the RNN to the Training set

        regressor.fit(X_train, np.ravel(y_train), epochs=100, batch_size=32)

        ##############################################################################
        # Part 3 - Making the predictions and visualising the results
        y_pred = regressor.predict(X_train)

        #Saving the model
        filepath, filename = ntpath.split(dataset_path)

        # Saving the model
        saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
        tf.keras.models.save_model(regressor, saving_path)


        framed_ypred = pd.DataFrame(y_pred)

        response_dict = "y_pred: {}".format(framed_ypred)
        return Response(response_dict, status=status.HTTP_201_CREATED)

        #except Exception as err:
        #return Response("Error ! please make sure everything is configured.")
"""
class TimeSeriesWWeights(APIView):
    def post(self, request):

        # Importing the training set

        # Importing the dataset
        # Read the csv files:

        dataset_path = 'D:\\My AI Projects\\BusinessMan\\Recurrent Neural Network\\Part 3 - Recurrent Neural Networks\\Google_Stock_Price_Train.csv'

        global dataset
        # Excel Extensions
        xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

        # Reading CSV Files
        if (dataset_path[-3:] == 'csv'):
            try:
                dataset = pd.read_csv(dataset_path)
            except (OSError, FileNotFoundError):
                print("Error Reading File!")

        # Reading excel files
        for file_extension in xl_extensions:
            if (dataset_path[-3:] == file_extension):
                try:
                    dataset = pd.read_excel(dataset_path)
                except OSError:
                    print("Error Reading File!")
        dataset_train = pd.read_csv(dataset_path)
        # range of columns selected in the dataset
        range_a = 1
        range_b = 2
        # might change the ranges later to make it take the entire dataset
        training_set = pd.DataFrame(dataset_train).iloc[:, range_a:range_b].values
        # y = dataset_train.iloc[:, -1].values

        # Time-steps to be predicted
        time_start = 60
        time_end = 1258

        # taking care of missing data
        if (dataset_train.isnull().values.any()):
            dataset_train.fillna(dataset_train.mean())  # fills empty cells with the mean of all values in the column

        column_count = 0
        for column in dataset_train:
            column_count = column_count + 1

        # Data Encoding
        # converting to onehotencoding all the datatypes of type 'object'
        x = pd.DataFrame(training_set)
        # y = pd.DataFrame(y)

        x_indpndnt_v_encoder = pd.get_dummies(x)
        x = np.array(x_indpndnt_v_encoder)

        # y_dpndnt_v_encoder = pd.get_dummies(y)
        # y = np.array(y_dpndnt_v_encoder)

        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.transform(x)

        ### Creating a data structure with 60 timesteps and 1 output

        X_train = []
        y_train = []
        for i in range(60, 1258):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)


        ## Part 2 - Building and Training the RNN

        ### Importing the Keras libraries and packages

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import Dropout


        ### Initialising the RNN
        regressor = Sequential()

        ### Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        ### Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        ### Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        ### Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))

        ### Adding the output layer
        regressor.add(TimeDistributed(Dense(units=1)))

        ### Compiling the RNN
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        ### Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs=100, batch_size=32)

        ##############################################################################
        # Part 3 - Making the predictions and visualising the results
        inputs = dataset_train[len(dataset_train) - time_start:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(time_start, 80):
            X_test.append(inputs[i - time_start:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Using the experience and predicting
        # type in the name of the file at the end of the next line (path) to load weights
        saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
        model_weights = tf.keras.models.load_model(saving_path)

        predicted_stock_price = model_weights.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)


        framed_ypred = pd.DataFrame(predicted_stock_price)

        response_dict = "y_pred: {}".format(framed_ypred)
        return Response(response_dict, status=status.HTTP_201_CREATED)

"""
class ModelWeightsWithoutUpdatingWeights(APIView):
    permission_classes = [IsOwner]
    def post(self, request):
        try:
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Classification\\ana - Binary (True or False)\\Churn_Modelling.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")

            x = dataset.iloc[:, :-1].values
            # Dependent Variable:
            y = dataset.iloc[:, -1].values

            # print(x)
            # print(y)

            # taking care of missing data
            if (dataset.isnull().values.any() == True):
                dataset.fillna(dataset.mean())  # whenever it finds empty columns are numbers, it fills them with its mean



            column_count = 0
            for column in dataset:
                column_count = column_count + 1

            # converting to onehotencoding all the datatypes of type 'object'
            x = pd.DataFrame(x)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(x)
            x = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)


            # 5)Feature scaling  --- Manadatory in Deep Learning ---
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x = sc.transform(x)


            # Using the experience and predicting
            #type in the name of the file at the end of the next line (path) to load weights
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            model_weights = tf.keras.models.load_model(saving_path)

            y_pred_x = model_weights.predict(x)


            # Making it Binary
            # y_pred_new[y_pred_new > 0.5] = 1
            # y_pred_new[y_pred_new <= 0.5] = 0
            # Framing data
            framed_ypred_x = (pd.DataFrame(y_pred_x))
            #framed_ypred_y = (pd.DataFrame(y_pred_y))

            response_dict = "y_pred_x: {}".format(framed_ypred_x)
            return Response(response_dict, status=status.HTTP_201_CREATED)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")


class ModelWeightsWWeightsUpdating(APIView):
    permission_classes = [IsOwner]
    def post(self, request):

        try:
            # Importing the dataset
            # Read the csv files:

            dataset_path = 'D:\\My AI Projects\\BusinessMan\\ANN - Classification\\ana - Binary (True or False)\\Churn_Modelling.csv'

            global dataset
            # Excel Extensions
            xl_extensions = ["xls", "xls", "xlsb", "xlsm", "xlsx", "xltx", "xlw"]

            # Reading CSV Files
            if (dataset_path[-3:] == 'csv'):
                try:
                    dataset = pd.read_csv(dataset_path)
                except (OSError, FileNotFoundError):
                    print("Error Reading File!")

            # Reading excel files
            for file_extension in xl_extensions:
                if (dataset_path[-3:] == file_extension):
                    try:
                        dataset = pd.read_excel(dataset_path)
                    except OSError:
                        print("Error Reading File!")

            x = dataset.iloc[:, :-1].values
            # Dependent Variable:
            y = dataset.iloc[:, -1].values

            # print(x)
            # print(y)

            # taking care of missing data
            if (dataset.isnull().values.any() == True):
                dataset.fillna(
                    dataset.mean())  # whenever it finds empty columns are numbers, it fills them with its mean

            column_count = 0
            for column in dataset:
                column_count = column_count + 1

            # converting to onehotencoding all the datatypes of type 'object'
            x = pd.DataFrame(x)
            y = pd.DataFrame(y)

            x_indpndnt_v_encoder = pd.get_dummies(x)
            x = np.array(x_indpndnt_v_encoder)

            y_dpndnt_v_encoder = pd.get_dummies(y)
            y = np.array(y_dpndnt_v_encoder)

            # 5)Feature scaling  --- Manadatory in Deep Learning ---
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x = sc.transform(x)

            # Using the experience and predicting
            # type in the name of the file at the end of the next line (path) to load weights
            saving_path = "C:\\Users\\alhus\\Desktop\\MyWebsite\\PickledModels\\"
            model_weights = tf.keras.models.load_model(saving_path)

            model_weights.fit(x,y, epochs= 100, batch_size= 32)
            y_pred_x = model_weights.predict(x)

            # Making it Binary
            # y_pred_new[y_pred_new > 0.5] = 1
            # y_pred_new[y_pred_new <= 0.5] = 0
            # Framing data
            framed_ypred_x = (pd.DataFrame(y_pred_x))
            # framed_ypred_y = (pd.DataFrame(y_pred_y))

            response_dict = "y_pred_x: {}".format(framed_ypred_x)
            return Response(response_dict, status=status.HTTP_201_CREATED)

        except Exception as err:
            return Response("Error ! please make sure everything is configured.")

