from django.db import models

# Create your models here.


class UserQuery(models.Model):
    user_query = models.TextField(null=True, blank=True)
    ai_response = models.TextField(null=True, blank=True)
    created_at = models.TimeField(auto_now_add=True)

