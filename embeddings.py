from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np

# CLIP model for both text and image embeddings
model = SentenceTransformer('clip-ViT-B-32')

def embed_text(text):
    """Generate text embedding using CLIP"""
    try:
        # CLIP returns a numpy array, convert to list for JSON serialization
        embedding = model.encode([text])[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Error in embed_text: {str(e)}")
        return None

def embed_image_bytes(image_bytes):
    """Generate image embedding using CLIP from image bytes"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Generate embedding (CLIP returns numpy array)
        embedding = model.encode([image])[0]  # CLIP returns 512-dim vector
        
        # Ensure we have the right dimensions
        if len(embedding) != 512:
            print(f"Warning: Unexpected embedding dimension: {len(embedding)}")
            
        return embedding.tolist()
        
    except Exception as e:
        print(f"Error in embed_image_bytes: {str(e)}")
        return None
