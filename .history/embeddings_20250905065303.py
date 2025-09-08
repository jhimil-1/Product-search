import torch
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer
import io

text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = SentenceTransformer('clip-ViT-B-32')

def embed_text(text):
    return text_model.encode([text])[0].tolist()

def embed_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return clip_model.encode([image])[0].tolist()
