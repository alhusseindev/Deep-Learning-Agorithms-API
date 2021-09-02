from django.contrib.auth.base_user import AbstractBaseUser
from django.db import models
from django.contrib.auth.models import BaseUserManager, PermissionsMixin
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
import os
import uuid
from MyWebsite.settings import AUTH_USER_MODEL
from django.utils import timezone
from pathlib import Path
import datetime

class MyUserManager(BaseUserManager):
    def create_user(self, first_name, last_name, email, password, **extra_fields):
        if(not email):
            raise ValueError("User must have an Email!")
        if(not password):
            raise ValueError("User must have a Password!")
        if(not first_name):
            raise ValueError("User must have a First Name!")
        if(not last_name):
            raise ValueError("User must have a Last Name!")

        user = self.model(
            first_name = first_name,
            last_name = last_name,
            email = self.normalize_email(email),
            **extra_fields
        )
        user.set_password(password) #Hashes the password
        user.is_admin = False
        user.is_superuser = False
        user.is_staff = False
        user.is_active = True
        user.save(using= self._db)
        return user

    def create_superuser(self, email, password, **extra_fields):
        if (not email):
            raise ValueError("User must have an Email!")
        if (not password):
            raise ValueError("User must have a Password!")
        user = self.model(
            email = self.normalize_email(email)
        )
        user.set_password(password)
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.is_active = True
        user.save(using= self._db)
        return user

class MyUser(AbstractBaseUser):
    first_name = models.CharField(max_length= 250)
    last_name = models.CharField(max_length= 250)
    email = models.EmailField(unique= True)

    day_dropdown = []
    month_dropdown = []
    year_dropdown = []
    for z in range(1930, (datetime.datetime.now().year)):
        year_dropdown.append((z,z))
    for y in range(1, (datetime.datetime.now().day)):
        day_dropdown.append((y,y))
    for x in range(1, (datetime.datetime.now().month)):
        month_dropdown.append((x,x))

    birthday = models.IntegerField(('Day'), choices= day_dropdown, null= False, default= datetime.datetime.now().day)
    birthmonth = models.IntegerField(('Month'), choices= month_dropdown, null= False, default= datetime.datetime.now().month)
    birthyear = models.IntegerField(('Year'), choices= year_dropdown,null= False, default= datetime.datetime.now().year)
    date_joined= models.DateTimeField(auto_now_add= True)
    # we have to include the following models, and we have overridden them above int he MyUserManager class
    # to make a distinction between a regular user and a superuser
    is_staff = models.BooleanField(default= False) #True for admin only
    is_superuser = models.BooleanField(default= False) #True for admin only
    is_admin = models.BooleanField(default= False) #True for admin only
    is_active = True

    USERNAME_FIELD = 'email'
    EMAIL_FIELD = 'email'
    REQUIRED_FIELDS=[]

    #Instantiating a MyUserManager object
    objects = MyUserManager()

    def get_absolute_url(self):
        return "/users/%i/" % (self.pk)

    def __str__(self):
        return "{} {}".format(self.first_name, self.last_name)

    def has_perm(self, perm, obj = None):
        #Does the user have a specific permission
        #simplest possible answer: yes, always
        return True

    def has_module_perms(self, app_label):
        #Does this user have permissions to view the app
        #simplest possible answer: yes, always
        return True

#Creating Folders

#creating an experience folder to save the weights
def create_weights_folder(instance, filename):
    folder = Path("users/user_{0}/AI Experience/{1}".format(instance.user, filename)).mkdir(parents= True, exist_ok= True)
    return folder

#creating a data folder to save data uploaded, such as excel files, etc...
def create_data_folder(instance, filename):
    folder = Path("users/user_{0}/Data/{1}".format(instance.user, filename)).mkdir(parents= True, exist_ok= True)
    return folder



# Create your models here.
class SLRModel(models.Model):
    user = models.OneToOneField(AUTH_USER_MODEL, on_delete= models.CASCADE, null= True)
    #user_folder = models.FileField(upload_to= create_user_folder, null= True)
    ai_weights = models.FileField(upload_to= create_weights_folder, null= True)
    data_folder = models.FileField(upload_to= create_data_folder, null= True)
    uploaded_at = models.DateTimeField(auto_now_add= True)

    def __str__(self):
        return self.user

    def save(self, *args, **kwargs):
        if (os.path.isfile("users/user_{0}/".format(self.user_id))):
            Path("users/user_{0}/".format(self.user_id)).mkdir(parents=True, exist_ok=True)
        super(SLRModel, self).save(*args, **kwargs)


@receiver(post_save, sender= AUTH_USER_MODEL)
def create_auth_token(sender, instance= None, created= False, **kwargs):
    #if a user registered and an account object is created
    if(created):
        Token.objects.create(user= instance) #this creates a token and stores it in the database
