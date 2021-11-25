from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.core.validators import RegexValidator
from django.forms import ModelForm



# Create your models here.
class Online(models.Model):
    file = models.FileField(upload_to='online/')
    email = models.EmailField(max_length=254)

class Offline(models.Model):
    file = models.FileField(upload_to='media/')

class UploadForm(ModelForm):
    class Meta:
        model = Online
        fields = ['file']

class QuickForm(ModelForm):
    class Meta:
        model = Offline
        fields = ['file']