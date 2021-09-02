
from django.urls import path, re_path
from rest_framework.routers import DefaultRouter
from .services import *
from .views import *

urlpatterns = [
    path('list/', ListSLRPredictions.as_view()),
    path('detail/', DetailSLRPredictions.as_view()),
    #path('auth/user/login/', LoginView.as_view()),
    path('logisticregres/', LogisticRegressionModel.as_view()),
    path('linearregres/', LinearRegression.as_view()),
    path('randforestregres/', RandomForestRegressor.as_view()),
    path('randforestclasifi/', RandomForestClassifier.as_view()),
    path('decisiontreeclasifi/', DecisionTreeClassifier.as_view()),
    path('decisiontreeregres/', DecisionTreeRegressor.as_view()),
    path('supportvectmachclassifi/', SupportVectorMachineClassifier.as_view()),
    path('supportvectmachregres/', SupportVectorMachineRegressor.as_view()),
    path('nivebyesgauss/', NaiveBayesGaussian.as_view()),
    path('nivebyesbernlli/', NaiveBayesBernoulli.as_view()),
    path('knrstneibrclassifi/', KNearestNeighborClassifier.as_view()),
    path('knrstneibrregres/', KNearestNeighborRegressor.as_view()),
    path('annregres/', ANNRegression.as_view()),
    path('annbinaryclassification/', ANNBinaryClassification.as_view()),
    path('annctgrcalclassifi/', ANNCategoricalClassification.as_view()),
    path('timeseries/', TimeSeries.as_view()),
    path('modelwweights/', ModelWeightsWithoutUpdatingWeights.as_view()),
    path('modelwweightsupdating/', ModelWeightsWWeightsUpdating.as_view()),
    path('uploadfile/', FileUploadView.as_view()),

]