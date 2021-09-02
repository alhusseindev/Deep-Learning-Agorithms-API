import json
import ntpath
from http.client import HTTPResponse
import os

from django.contrib import auth
from django.contrib.auth import get_user_model, login, logout
from rest_framework.authentication import TokenAuthentication
from rest_framework.generics import CreateAPIView, GenericAPIView
from rest_framework.parsers import JSONParser,FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.permissions import *
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView


from .permissions import IsOwner
from .serializers import *
from .models import SLRModel


# Create your views here.
class ListSLRPredictions(generics.ListAPIView):
    queryset = SLRModel.objects.all()
    serializer_class = SLRSerializer


class DetailSLRPredictions(generics.RetrieveAPIView):
    queryset = SLRModel.objects.all()
    serializer_class = SLRSerializer



#Handling user signup
class CreateUserView(CreateAPIView):
    queryset = get_user_model()
    permission_classes = [AllowAny, ]
    serializer_class = UserSignUpSerializer
    def post(self, request):
        serializer = UserSignUpSerializer(data= request.data)
        data = {}
        if(serializer.is_valid()):
            account = serializer.save()
            data['response'] = "Successfully Registered A New User!"
            data['response'] = serializer.data
            token = Token.objects.get(user= account).key
            data['token'] = token

        else:
            data = serializer.errors
        return Response(data)



#Login
class UserLoginView(GenericAPIView):
    serializer_class = UserLoginSerializer
    def post(self, request):
        serializer = UserLoginSerializer(data= request.data)
        serializer.is_valid(raise_exception= True)
        user = serializer.validated_data["user"]
        login(request, user)
        token, created = Token.objects.get_or_create(user= user)
        return Response({"token": token.key}, status= status.HTTP_200_OK)




#Logout
class UserLogoutView(APIView):
    authentication_classes = (TokenAuthentication,)
    def post(self, request):
        # Taking care of logging out from the device were you want to
        # logout from, for example if you are logged in 2 devices, logging out from one device
        # logs you out from that device only
        # so we logout from the session only
        logout(request)
        return Response(status= status.HTTP_200_OK)




class FileUploadView(APIView):
    parser_classes= (JSONParser,MultiPartParser, FormParser)
    permission_classes= [IsOwner,]
    def post(self, request, *args, **kwargs):
        my_file = request.FILES['file']
        my_user = request.user

        try:
            saving_file = open("users/{}".format(my_user), "w+")
            for chunks in saving_file:
                saving_file.write(chunks)
            saving_file.close()
        except Exception as err:
            raise Exception("Cannot Locate User Folder!")

        return Response(request.data, status= status.HTTP_201_CREATED)

