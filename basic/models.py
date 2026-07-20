from django.db import models

# Create your models here.


class Greeting(models.Model):
    sentences = models.CharField(max_length=200, null=True)


class UserQuery(models.Model):
    user_query = models.CharField(max_length=150, null=True)
    ai_response = models.CharField(max_length=150, null=True)
    created_at = models.TimeField(auto_now_add=True)


class ResourceMonitor(models.Model):
    mode = models.CharField(max_length=100, null=True)
    cpu_usage = models.FloatField(null=True)
    memory_rss = models.FloatField(null=True)
