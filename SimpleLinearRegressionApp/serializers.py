from SimpleLinearRegressionApp.models import *
from rest_framework import serializers, status
from django.contrib.auth import get_user_model, login
from django.contrib.auth import authenticate
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class SLRSerializer(serializers.ModelSerializer):
    class Meta:
        model = SLRModel
        fields = '__all__'


#User SignUp View
UserModel = get_user_model()

class UserSignUpSerializer(serializers.ModelSerializer):
    password = serializers.CharField(style= {'input_type':'password'}, write_only= True)
    password2 = serializers.CharField(style= {'input_type':'password'}, write_only= True)
    def validate(self, attrs):
        email = attrs.get("email","")
        if(UserModel.objects.filter(email= email).exists()):
            raise serializers.ValidationError({"Email" : "Email is already in use!"})
        return super().validate(attrs)


    def create(self, validated_data):
        user = UserModel.objects.create(
            #here we include the fields we want to be showing with its info, when making a POST request
            first_name = self.validated_data['first_name'],
            last_name = self.validated_data['last_name'],
            email = self.validated_data['email'],
            birthday = self.validated_data['birthday'],
            birthmonth = self.validated_data['birthmonth'],
            birthyear = self.validated_data['birthyear'],
        )
        #We don't include the password field above, because we don't want it to be show when making a POST request
        password = self.validated_data['password']
        password2 = self.validated_data['password2']
        if(password != password2):
            raise serializers.ValidationError("Passwords do not match!")
        user.set_password(password)
        user.save()
        return user
    class Meta:
        model = UserModel
        fields = ('first_name', 'last_name', 'email', 'birthday', 'birthmonth' , 'birthyear','password', 'password2',)



#Login
class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()
    def validate(self, data):
        email = data.get("email", "")
        password = data.get("password", "")

        if(email and password):
            user = authenticate(email= email, password= email)

            #If user credintials are correct
            if(user):
                if(user.is_active):
                    data["user"] = user
                else:
                    raise serializers.ValidationError("User Is Deactivated!")
            else:
                raise serializers.ValidationError("Unable To Login with given Email/Password!")
        else:
            raise serializers.ValidationError("Must Provide Email / Password!")
        return data