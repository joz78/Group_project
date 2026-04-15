import json
from io import BytesIO
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
        image_file = request.FILES.get('image')
        features = None
        prediction = None

        if image_file is not None:
            if not image_file.content_type.startswith('image/'):
                raise ValueError('Uploaded file must be an image')

            try:
                from PIL import Image
            except ImportError:
                return JsonResponse({'error': 'Pillow is required for image uploads'}, status=500)

            image = Image.open(image_file).convert('RGB')
            model = load_ml_model()
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            shape = [dim for dim in input_shape[1:] if dim is not None]
            if len(shape) not in (2, 3):
                raise ValueError('Unsupported model input shape for image uploads')

            if len(shape) == 2:
                height, width = shape
                channels = 3
                channels_first = False
            else:
                if shape[0] in (1, 3) and shape[-1] not in (1, 3):
                    channels_first = True
                    channels, height, width = shape
                else:
                    channels_first = False
                    height, width, channels = shape

            if height is None or width is None:
                raise ValueError('Model input shape cannot contain unknown height or width')

            image = image.resize((width, height))
            import numpy as np
            array = np.asarray(image, dtype=np.float32) / 255.0
            if channels == 1:
                array = np.mean(array, axis=2, keepdims=True)
            if channels_first:
                array = array.transpose(2, 0, 1)
            array = np.expand_dims(array, axis=0)

            prediction = model.predict(array).tolist()
            features = {
                'type': 'image',
                'name': image_file.name,
                'content_type': image_file.content_type,
            }
        else:
            if not request.body:
                return JsonResponse({'error': 'No image file or JSON payload provided'}, status=400)

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
