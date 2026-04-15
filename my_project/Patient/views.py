import json
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Prediction

MODEL = None
MODEL_FILE = None
ML_MODELS_DIR = settings.BASE_DIR / 'ml_models'

for path in ML_MODELS_DIR.iterdir():
    if path.suffix in {'.keras', '.h5', '.hdf5'}:
        MODEL_FILE = path
        break


def load_ml_model():
    global MODEL
    if MODEL is None:
        if MODEL_FILE is None or not MODEL_FILE.exists():
            raise RuntimeError('ML model file not found in ml_models/')

        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            from keras.models import load_model

        MODEL = load_model(str(MODEL_FILE))
    return MODEL


def home(request):
    return render(request, 'home.html')


@csrf_exempt
def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        payload = json.loads(request.body.decode('utf-8'))
        features = payload.get('features') or payload.get('data')

        if features is None:
            return JsonResponse({'error': 'Missing features or data in JSON body'}, status=400)

        if not isinstance(features, list):
            raise ValueError('features must be a list of numbers')

        import numpy as np

        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim > 2:
            raise ValueError('Input shape not supported')

        model = load_ml_model()
        prediction = model.predict(array).tolist()

        record = Prediction.objects.create(input_data=features, prediction=prediction)
        return JsonResponse({'prediction': prediction, 'record_id': record.id})
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=400)
