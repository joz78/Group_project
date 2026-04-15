from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'notes')
    list_filter = ('created_at',)
    search_fields = ('notes',)
