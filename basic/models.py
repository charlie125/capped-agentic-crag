from django.db import models

# Create your models here.


class UserQuery(models.Model):
    user_query = models.CharField(max_length=150, null=True)
    ai_response = models.CharField(max_length=150, null=True)
    created_at = models.TimeField(auto_now_add=True)
