from django.db import models

# Create your models here.


class UserQuery(models.Model):
    query = models.CharField(max_length=150)
    response = models.CharField()
    created_at = models.TimeField(auto_now_add=True)
