from django.db import models


class Prediction(models.Model):
    input_data = models.JSONField()
    prediction = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f'Prediction #{self.pk} - {self.created_at:%Y-%m-%d %H:%M:%S}'
