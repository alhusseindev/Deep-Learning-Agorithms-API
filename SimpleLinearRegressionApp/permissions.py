from rest_framework import permissions
from .models import SLRModel



class IsOwner(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        #Allowing READ-Only permission for users who are not owners
        if(request.method in permissions.SAFE_METHODS):
            return True
        #WRITE permissions are only allowed for the user
        #obj.user -> user which is a field in SLRModel in models.py
        return obj.user == request.user

