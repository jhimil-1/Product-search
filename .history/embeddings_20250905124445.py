from sentence_transformers import SentenceTransformer
from PIL import Image
import io

# Text embedding model
text_model = SentenceTransformer('all-MiniLM-L6-v2')
# CLIP image embedding model
clip_model = SentenceTransformer('clip-ViT-B-32')

def embed_text(text):
    return text_model.encode([text])[0].tolist()

def embed_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return clip_model.encode([image])[0].tolist()

