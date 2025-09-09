from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Use CLIP model for both text and image embeddings
model = SentenceTransformer('clip-ViT-B-32')
EMBEDDING_DIM = 512  # CLIP produces 512-dimensional vectors

def embed_text(text):
    """Generate text embedding using CLIP model"""
    try:
        # CLIP returns a numpy array, convert to list for JSON serialization
        embedding = model.encode([text], convert_to_tensor=False)[0].tolist()
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(f"Unexpected text embedding dimension: {len(embedding)}, expected {EMBEDDING_DIM}")
        return embedding
    except Exception as e:
        logger.error(f"Error in embed_text: {str(e)}", exc_info=True)
        return None

def embed_image_bytes(image_bytes):
    """Generate image embedding using CLIP model"""
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        # Initialize CLIP model and processor if not already loaded
        if not hasattr(embed_image_bytes, 'model'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embed_image_bytes.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            embed_image_bytes.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            embed_image_bytes.device = device
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process image through CLIP
        inputs = embed_image_bytes.processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(embed_image_bytes.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = embed_image_bytes.model.get_image_features(**inputs)
            
        # Convert to numpy array and ensure correct shape
        embedding = image_features.cpu().numpy().astype('float32')
        
        # Ensure we have a 1D array of the correct dimension
        if len(embedding.shape) > 1:
            embedding = embedding.reshape(-1)
        
        # Check dimension
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(f"Unexpected image embedding dimension: {len(embedding)}, expected {EMBEDDING_DIM}")
            # Truncate or pad if necessary
            if len(embedding) > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            else:
                padding = np.zeros(EMBEDDING_DIM - len(embedding), dtype='float32')
                embedding = np.concatenate([embedding, padding])
        
        # Normalize the embedding (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"Error in embed_image_bytes: {str(e)}", exc_info=True)
        return None
