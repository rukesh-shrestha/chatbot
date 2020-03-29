from django.db import models

# Create your models here.
class chat(models.Model):
    name = models.CharField(max_length=30) #required
    email = models.EmailField(max_length=50, default='someone@gmail.com')
    query = models.TextField(blank=False,max_length=30)
    description = models.TextField(null=False,max_length=60)