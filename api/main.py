from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../saved_models/2")
# MODEL = tf.keras.models.load_model("E:/Pea_Plant_disease/saved_models/pea_model.h5")
# MODEL = tf.keras.models.load_model("E:/Pea_Plant_disease/saved_models/2")
# MODEL = tf.saved_model.load("E:/Pea_Plant_disease/saved_models/2")

MODEL = tf.saved_model.load("E:/Pea_Plant_disease/saved_models/2")
infer = MODEL.signatures["serving_default"]


CLASS_NAMES = ['DOWNY_MILDEW_LEAF', 'FRESH_LEAF', 'LEAFMINNER_LEAF', 'POWDER_MILDEW_LEAF']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0).astype(np.float32)

    # Convert numpy to tensor and call the model
    input_tensor = tf.convert_to_tensor(img_batch)
    output = infer(input_tensor)

    # Extract predictions (depends on export signature)
    predictions = list(output.values())[0].numpy()
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8002)













# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # CORS configuration - more permissive for development
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model with error handling
# try:
#     MODEL_PATH = r"E:\Pea_Plant_disease\saved_models\2"  # Use raw string for Windows paths
#     MODEL = tf.saved_model.load(MODEL_PATH)
#     infer = MODEL.signatures["serving_default"]
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {e}")
#     raise

# CLASS_NAMES = ['DOWNY_MILDEW_LEAF', 'FRESH_LEAF', 'LEAFMINNER_LEAF', 'POWDER_MILDEW_LEAF']

# @app.get("/ping")
# async def ping():
#     return {"message": "Hello, I am alive", "status": "healthy"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Validate file type
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read and process image
#         image_data = await file.read()
#         image = np.array(Image.open(BytesIO(image_data)))
        
#         # Validate image shape
#         if len(image.shape) not in [2, 3]:
#             raise HTTPException(status_code=400, detail="Invalid image format")
        
#         # Preprocess
#         img_batch = np.expand_dims(image, 0).astype(np.float32)
#         input_tensor = tf.convert_to_tensor(img_batch)
        
#         # Predict
#         output = infer(input_tensor)
#         predictions = list(output.values())[0].numpy()
        
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))
        
#         return {
#             'class': predicted_class,
#             'confidence': confidence
#         }
        
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host='0.0.0.0', port=8002)

