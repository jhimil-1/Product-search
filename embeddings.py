from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Use a smaller model to avoid memory issues
model = SentenceTransformer('all-MiniLM-L6-v2')
TEXT_EMBEDDING_DIM = 384  # This model produces 384-dimensional vectors
IMAGE_EMBEDDING_DIM = 512 # CLIP model produces 512-dimensional vectors

def embed_text(text):
    """Generate text embedding using CLIP model"""
    try:
        # CLIP returns a numpy array, convert to list for JSON serialization
        embedding = model.encode([text], convert_to_tensor=False)[0].tolist()
        if len(embedding) != TEXT_EMBEDDING_DIM:
            logger.warning(f"Unexpected text embedding dimension: {len(embedding)}, expected {TEXT_EMBEDDING_DIM}")
        return embedding
    except Exception as e:
        logger.error(f"Error in embed_text: {str(e)}", exc_info=True)
        return None

def preprocess_image(image, target_size=224):
    """Preprocess image with resizing and normalization"""
    # Resize maintaining aspect ratio
    width, height = image.size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize with high-quality downsampling
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop to target size
    left = (new_width - target_size) / 2
    top = (new_height - target_size) / 2
    right = (new_width + target_size) / 2
    bottom = (new_height + target_size) / 2
    
    return image.crop((left, top, right, bottom))

def embed_image_bytes(image_bytes, basic_mode=False):
    """
    Generate image embedding using CLIP model
    
    Args:
        image_bytes: Binary image data
        basic_mode: If True, use faster but less accurate processing
        
    Returns:
        List of floats: Image embedding vector
    """
    try:
        # Log the image bytes length for debugging
        logger.info(f"Processing image with {len(image_bytes)} bytes")
        
        import torch
        from torchvision import transforms
        from transformers import CLIPProcessor, CLIPModel
        
        # Initialize CLIP model and processor if not already loaded
        if not hasattr(embed_image_bytes, 'model'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP model on {device}")
            embed_image_bytes.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float16 if 'cuda' in device else torch.float32
            ).to(device)
            embed_image_bytes.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            embed_image_bytes.device = device
            embed_image_bytes.model.eval()  # Set to evaluation mode
        
        # Validate image bytes
        if not image_bytes or len(image_bytes) == 0:
            logger.error("Empty image bytes received")
            # Return a default embedding instead of None
            logger.warning("Returning default embedding due to empty image bytes")
            default_embedding = np.zeros(IMAGE_EMBEDDING_DIM, dtype='float32')
            default_embedding[0] = 1.0  # Set first value to 1.0 to ensure non-zero vector
            logger.info(f"Returning default embedding with length: {len(default_embedding.tolist())}")
            return default_embedding.tolist()
            
        # Convert bytes to PIL Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.info(f"Successfully opened image with size {image.size}")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            # Return a default embedding instead of None
            logger.warning("Returning default embedding due to image opening error")
            default_embedding = np.zeros(IMAGE_EMBEDDING_DIM, dtype='float32')
            default_embedding[0] = 1.0  # Set first value to 1.0 to ensure non-zero vector
            logger.info(f"Returning default embedding with length: {len(default_embedding.tolist())}")
            return default_embedding.tolist()
            
        # Enhanced preprocessing for better results
        try:
            if not basic_mode and min(image.size) > 224:  # Only preprocess if image is large enough
                image = preprocess_image(image)
                logger.info(f"Image preprocessed to size {image.size}")
        except Exception as e:
            logger.error(f"Error during image preprocessing: {str(e)}")
            # Fall back to basic mode if preprocessing fails
            basic_mode = True
        
        # Process image through CLIP with different strategies based on mode
        try:
            if basic_mode:
                # Basic processing - faster but less accurate
                logger.info("Using basic image processing mode")
                inputs = embed_image_bytes.processor(
                    images=image, 
                    return_tensors="pt", 
                    padding=True,
                    do_rescale=True,
                    do_normalize=True,
                    do_center_crop=True,
                    size={"height": 224, "width": 224}
                )
            else:
                # Enhanced processing with better preprocessing
                logger.info("Using enhanced image processing mode")
                inputs = embed_image_bytes.processor(
                    images=image,
                    return_tensors="pt",
                    do_rescale=True,
                    do_normalize=True,
                    do_resize=True,
                    size={"height": 224, "width": 224}
                )
                
            # Move inputs to the correct device
            inputs = {k: v.to(embed_image_bytes.device) for k, v in inputs.items()}
            logger.info("Successfully moved inputs to device")
            
            # Generate embedding with gradient disabled for inference
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = embed_image_bytes.model.get_image_features(**inputs)
                logger.info("Successfully generated image features")
                
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}")
            # Return a default embedding instead of None
            logger.warning("Returning default embedding due to image processing error")
            default_embedding = np.zeros(IMAGE_EMBEDDING_DIM, dtype='float32')
            default_embedding[0] = 1.0  # Set first value to 1.0 to ensure non-zero vector
            logger.info(f"Returning default embedding with length: {len(default_embedding.tolist())}")
            return default_embedding.tolist()
            
        # Convert to numpy array and ensure correct shape
        embedding = image_features.cpu().numpy().astype('float32')
        logger.info(f"Raw embedding shape: {embedding.shape}")
        
        # Ensure we have a 1D array of the correct dimension
        if len(embedding.shape) > 1:
            embedding = embedding.reshape(-1)
            logger.info(f"Reshaped embedding length: {len(embedding)}")
        
        # Check dimension
        if len(embedding) != IMAGE_EMBEDDING_DIM:
            logger.warning(f"Unexpected image embedding dimension: {len(embedding)}, expected {IMAGE_EMBEDDING_DIM}")
            # Truncate or pad if necessary
            if len(embedding) > IMAGE_EMBEDDING_DIM:
                embedding = embedding[:IMAGE_EMBEDDING_DIM]
            else:
                padding = np.zeros(IMAGE_EMBEDDING_DIM - len(embedding), dtype='float32')
                embedding = np.concatenate([embedding, padding])
            logger.info(f"Adjusted embedding to correct dimension: {len(embedding)}")
        
        # Normalize the embedding (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            logger.info("Successfully normalized embedding")
            
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"Error in embed_image_bytes: {str(e)}", exc_info=True)
        # Return a default embedding of the correct dimension as a fallback
        logger.warning("Returning default embedding as fallback")
        default_embedding = np.zeros(IMAGE_EMBEDDING_DIM, dtype='float32')
        # Normalize the default embedding
        default_embedding[0] = 1.0  # Set first value to 1.0 to ensure non-zero vector
        return default_embedding.tolist()