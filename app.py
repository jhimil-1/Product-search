from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, make_response
from flask_cors import CORS
import os, uuid, json, numpy as np, base64
from dotenv import load_dotenv
from embeddings import embed_text, embed_image_bytes, TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM
from vectorstore import VectorStore
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
import logging
import requests
from urllib.parse import urlparse
import shutil
from werkzeug.utils import secure_filename
import tempfile
from functools import wraps
from sentence_transformers import SentenceTransformer
from qdrant_client import models
import re

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load environment variables
load_dotenv()

# Initialize the model when the module loads
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    """
    Generate embedding for the given text using the sentence transformer model.
    """
    try:
        # Generate embedding
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()  # Convert numpy array to list for JSON serialization
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

# ---------------------------
# Config
# ---------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-123')  # Change this in production
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "ai_ecom_chatbot")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------------------------
# Initialize
# ---------------------------
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)  # Session expires after 1 day

# MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.server_info()  # Test the connection
    db = mongo_client[DB_NAME]
    sessions_col = db["sessions"]
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

# Initialize VectorStore
vector_store = VectorStore(
    host=QDRANT_URL,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY
)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add a decorator to protect routes that require authentication
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            # Check if this is an API endpoint by looking at the request path
            if request.path.startswith(('/query', '/upload_products', '/query_image', '/ask_about_product', '/api/')):
                return jsonify({"error": "Authentication required", "authenticated": False}), 401
            else:
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Enhanced jewelry categorization
def detect_jewelry_category(text):
    """
    Enhanced jewelry category detection with better accuracy and single category focus
    """
    if not text or not isinstance(text, str):
        return []
        
    text_lower = text.lower().strip()
    
    # Define comprehensive jewelry categories with weighted patterns
    # Higher weights indicate stronger category indicators
    category_patterns = {
        'necklace': [
            # Exact matches - highest weight
            (r'\bnecklaces?\b', 1.0),
            (r'\bpendants?\b', 1.0),
            (r'\bchains?\b', 0.95),
            (r'\bchokers?\b', 1.0),
            (r'\blockets?\b', 1.0),
            # Compound patterns
            (r'\b(?:gold|silver|diamond|pearl)\s+necklaces?\b', 1.0),
            (r'\b(?:pendant|chain)\s+necklaces?\b', 1.0),
            (r'\bnecklace\s+(?:set|chain|pendant)\b', 1.0),
            # Descriptive patterns
            (r'\b(?:layered|layering|beaded|strand)\s+necklaces?\b', 0.9),
            (r'\b(?:statement|tennis|opera|princess)\s+necklaces?\b', 0.9)
        ],
        'ring': [
            (r'\brings?\b', 1.0),
            (r'\b(?:wedding|engagement)\s+rings?\b', 1.0),
            (r'\bbands?\b', 0.85),  # Lower weight as "band" can be ambiguous
            (r'\b(?:diamond|gold|silver|platinum)\s+rings?\b', 1.0),
            (r'\b(?:solitaire|eternity|promise|cocktail|signet)\s+rings?\b', 0.95),
            (r'\bring\s+(?:set|band|size)\b', 1.0),
            (r'\b(?:stackable|stacking)\s+rings?\b', 0.95)
        ],
        'earring': [
            (r'\bearrings?\b', 1.0),
            (r'\bstuds?\b', 0.95),
            (r'\bhoops?\b', 0.95),
            (r'\b(?:dangle|drop|chandelier)\s+earrings?\b', 1.0),
            (r'\b(?:gold|silver|diamond)\s+earrings?\b', 1.0),
            (r'\bearring\s+(?:set|pair)\b', 1.0),
            (r'\b(?:huggie|threader|jacket)\s+earrings?\b', 0.95),
            (r'\bear\s+(?:cuffs?|jackets?|threaders?)\b', 0.9)
        ],
        'bracelet': [
            (r'\bbracelets?\b', 1.0),
            (r'\bbangles?\b', 1.0),
            (r'\bcuffs?\b', 0.9),  # Can be ear cuffs too
            (r'\b(?:charm|tennis|chain|bead)\s+bracelets?\b', 1.0),
            (r'\b(?:gold|silver|diamond)\s+bracelets?\b', 1.0),
            (r'\bbracelet\s+(?:set|chain)\b', 1.0),
            (r'\b(?:leather|rope|cord)\s+bracelets?\b', 0.95)
        ],
        'anklet': [
            (r'\banklets?\b', 1.0),
            (r'\bankle\s+(?:chains?|bracelets?)\b', 1.0),
            (r'\bfoot\s+jewelry\b', 0.9)
        ],
        'watch': [
            (r'\bwatches?\b', 1.0),
            (r'\bwristwatch(?:es)?\b', 1.0),
            (r'\bsmartwatch(?:es)?\b', 1.0),
            (r'\btimepiece(?:s)?\b', 0.95),
            (r'\bwatch\s+(?:band|strap|face)\b', 1.0)
        ],
        'brooch': [
            (r'\bbrooches?\b', 1.0),
            (r'\bpins?\b', 0.8),  # "pins" can be ambiguous
            (r'\blapel\s+pins?\b', 1.0),
            (r'\bcorsage\s+pins?\b', 1.0)
        ]
    }
    
    # Calculate scores for each category
    category_scores = {}
    
    for category, patterns in category_patterns.items():
        max_score = 0.0
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Score based on number of matches and weight
                score = len(matches) * weight
                max_score = max(max_score, score)
        
        if max_score > 0:
            category_scores[category] = max_score
    
    # Sort categories by score and return only the highest scoring one
    if category_scores:
        sorted_categories = sorted(
            category_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Only return the top category if it has a strong score
        top_category, top_score = sorted_categories[0]
        if top_score >= 0.85:  # Threshold for confidence
            return [top_category]
    
    return []
    return [x for x in detected_categories if not (x in seen or seen.add(x))]

def extract_category_from_image_query(query_text=""):
    """Extract category from image search context"""
    if not query_text:
        return []
    
    # Check for category mentions in the query
    detected = detect_jewelry_category(query_text)
    return detected

# Add login route before other routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        # If user is already logged in, redirect to index
        if 'user' in session:
            return redirect(url_for('serve_index'))
        # Serve login page
        return send_from_directory('frontend', 'login.html')
    
    try:
        # Handle POST request
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password are required'}), 400
        
        # In a real application, you would validate the username and password here
        # For now, we'll just create a session for any valid credentials
        if data['username'] and data['password']:
            # Create user session
            user = {
                'username': data['username'],
                'name': data['username'].split('@')[0].capitalize()
            }
            
            # Store user in session
            session['user'] = user
            session.permanent = True  # Make the session persistent
            
            return jsonify({
                'message': 'Login successful',
                'user': user,
                'redirect': '/'  # Add redirect URL
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Serve index.html from the frontend directory
@app.route("/")
def serve_index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory('frontend', 'index.html')

# Serve static files from the frontend directory
@app.route('/<path:path>')
def serve_static(path):
    # Allow access to login page and static files without authentication
    if 'user' not in session and path != 'login.html' and not path.startswith(('js/', 'static/')):
        return redirect(url_for('login'))
    return send_from_directory('frontend', path)

# Serve JavaScript files
@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join('frontend', 'js'), filename)

# Serve CSS files
@app.route('/static/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join('frontend', 'static', 'css'), filename)

@app.route("/uploads/<filename>")
def serve_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Check if user is logged in
@app.route('/check_session')
def check_session():
    if 'user' in session:
        return jsonify({'authenticated': True, 'user': session['user']})
    return jsonify({'authenticated': False}), 401

# ---------------------------
# Create session
# ---------------------------
@app.route("/create_session", methods=["POST"])
@login_required
def create_session():
    try:
        # Test MongoDB connection
        mongo_client.server_info()
        
        session_id = str(uuid.uuid4())
        collection_name = f"session_{session_id}"

        try:
            # Create Qdrant collection with 512 dimensions for CLIP embeddings
            vector_store.create_collection(collection_name, recreate=True)
            logger.info(f"Created new Qdrant collection: {collection_name} with 512 dimensions")
        except Exception as qe:
            logger.error(f"Failed to create Qdrant collection: {str(qe)}")
            return jsonify({
                "error": f"Failed to create vector database collection. Error: {str(qe)}"
            }), 500

        # Save session in MongoDB
        try:
            sessions_col.insert_one({
                "_id": session_id,
                "collection_name": collection_name,
                "created_at": datetime.now(timezone.utc),
                "status": "active",
                "vector_size": 512  # Store the vector size for reference
            })
            return jsonify({
                "session_id": session_id,
                "collection_name": collection_name,
                "status": "success"
            })
        except Exception as e:
            # Clean up Qdrant collection if MongoDB insert fails
            try:
                vector_store.delete_collection(collection_name)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up Qdrant collection: {str(cleanup_error)}")
                
            logger.error(f"MongoDB error: {str(e)}")
            return jsonify({"error": "Failed to create session in database"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in create_session: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

def save_image_from_url(url, session_id):
    """Download and save image from URL or local path to uploads folder"""
    try:
        if not url:
            logger.error("No URL provided")
            return None
            
        logger.info(f"Processing image URL: {url}")
        
        # Handle local file paths
        if os.path.exists(url):
            # Generate a unique filename
            ext = os.path.splitext(url)[1].lower()
            if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                ext = '.jpg'  # Default extension
                
            filename = f"{uuid.uuid4()}{ext}"
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(url, dest_path)
            logger.info(f"Copied local file to {dest_path}")
            return f"/uploads/{filename}"
                
        # Handle file:// URLs
        if url.startswith('file://'):
            filepath = url[7:]  # Remove 'file://' prefix
            if os.path.exists(filepath):
                return save_image_from_url(filepath, session_id)  # Recursively handle as local path
            
        # Handle HTTP/HTTPS URLs
        if url.startswith(('http://', 'https://')):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, stream=True, headers=headers, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Get content type to determine file extension
                content_type = response.headers.get('content-type', '').lower()
                if 'image' not in content_type:
                    logger.error(f"URL does not point to an image: {content_type}")
                    return None
                
                # Determine file extension from content type or URL
                ext = None
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                else:
                    # Try to get extension from URL
                    url_path = urlparse(url).path
                    ext = os.path.splitext(url_path)[1]
                    if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif']:
                        ext = '.jpg'  # Default extension
                
                filename = f"{uuid.uuid4()}{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure upload directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save the image with proper error handling
                try:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # Filter out keep-alive chunks
                                f.write(chunk)
                    
                    logger.info(f"Saved image from URL to {filepath}")
                    return f"/uploads/{filename}"
                    
                except IOError as e:
                    logger.error(f"Error saving image to {filepath}: {str(e)}")
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading image from {url}: {str(e)}")
                return None
        
        logger.error(f"Unsupported URL or path: {url}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error in save_image_from_url: {str(e)}", exc_info=True)
        return None

# ---------------------------
# Debug Endpoints
# ---------------------------
@app.route("/debug/category_detection", methods=["POST"])
def debug_category_detection():
    """Debug endpoint to test category detection"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        detected = detect_jewelry_category(query)
        return jsonify({
            "query": query,
            "detected_categories": detected,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in debug_category_detection: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/debug/products', methods=['GET'])
def debug_products():
    """Debug endpoint to list all products and their categories"""
    try:
        # Get all collections
        collections = vector_store.client.get_collections().collections
        if not collections:
            return jsonify({
                'error': 'No collections found',
                'success': False
            }), 404
        
        # Get products from all collections
        all_products = []
        for collection in collections:
            collection_name = collection.name
            try:
                # Get total count
                count = vector_store.client.count(collection_name=collection_name, exact=True).count
                if count == 0:
                    continue
                    
                # Get sample of products
                results = vector_store.client.scroll(
                    collection_name=collection_name,
                    limit=min(50, count),
                    with_payload=True,
                    with_vectors=False
                )
                
                if results and len(results) > 0 and len(results[0]) > 0:
                    for point in results[0]:
                        payload = point.payload or {}
                        all_products.append({
                            'collection': collection_name,
                            'id': str(point.id),
                            'name': payload.get('name', 'Unnamed'),
                            'category': payload.get('category', 'No category'),
                            'description': payload.get('description', 'No description')
                        })
            except Exception as e:
                logger.error(f"Error reading collection {collection_name}: {str(e)}")
        
        if not all_products:
            return jsonify({
                'message': 'No products found in any collection',
                'success': False
            }), 404
            
        return jsonify({
            'products': all_products,
            'total_products': len(all_products),
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error retrieving products: {str(e)}',
            'success': False
        }), 500

@app.route("/debug/collection/<session_id>")
def debug_collection(session_id):
    """Debug endpoint to check collection contents"""
    try:
        # Verify session exists
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "Session not found or invalid"}), 404
            
        collection_name = session_data['collection_name']
        
        # Get collection info from Qdrant
        try:
            collection_info = vector_store.client.get_collection(collection_name)
        except Exception as e:
            return jsonify({
                "error": f"Error getting collection info: {str(e)}",
                "session_id": session_id,
                "collection_name": collection_name,
                "exists": False
            }), 404
        
        # Get count of points in the collection
        count_result = vector_store.client.count(
            collection_name=collection_name,
            exact=True
        )
        
        # Get sample points (first 5)
        points, _ = vector_store.client.scroll(
            collection_name=collection_name,
            limit=5,
            with_vectors=False,
            with_payload=True
        )
        
        # Clean up points for JSON serialization
        sample_points = []
        for point in points:
            sample_points.append({
                "id": str(point.id) if hasattr(point, 'id') else None,
                "payload": {k: str(v) if isinstance(v, (bytes, bytearray)) else v 
                           for k, v in (point.payload or {}).items()}
            })
        
        return jsonify({
            "session_id": session_id,
            "collection_name": collection_name,
            "exists": True,
            "vectors_count": getattr(count_result, 'count', 0) if hasattr(count_result, 'count') else 0,
            "collection_info": str(collection_info),
            "sample_points": sample_points
        })
        
    except Exception as e:
        logger.error(f"Error in debug_collection: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# ---------------------------
# Upload products
# ---------------------------
@app.route("/upload_products/<session_id>", methods=["POST"])
@login_required
def upload_products(session_id):
    if 'user' not in session:
        return jsonify({"error": "Not authenticated"}), 401
        
    try:
        logger.info(f"Received upload_products request for session: {session_id}")
        
        # Get the JSON data from form data
        products_json = request.form.get('products_json')
        if not products_json:
            logger.error("No products_json in form data")
            return jsonify({"error": "No products data provided"}), 400
            
        # Parse the products JSON
        try:
            products = json.loads(products_json)
            logger.info(f"Successfully parsed {len(products)} products from JSON")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return jsonify({"error": "Invalid JSON format for products"}), 400
            
        if not isinstance(products, list):
            return jsonify({"error": "Products data should be an array"}), 400

        # Process file uploads if any
        uploaded_files = []
        if 'images' in request.files:
            uploaded_files = request.files.getlist('images')
            logger.info(f"Received {len(uploaded_files)} image files")
            
            # Create a mapping of filenames to their secure filenames
            file_mapping = {}
            for file in uploaded_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        file.save(filepath)
                        file_mapping[filename] = filename  # Store mapping of original to saved filename
                        logger.info(f"Saved uploaded file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error saving file {filename}: {str(e)}")
        else:
            logger.info("No images received in the request")
        
        # Get the collection name for this session
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "Session not found"}), 404
            
        collection_name = session_data['collection_name']
        
        # Process products
        points = []
        for idx, product in enumerate(products):
            try:
                # Generate a new UUID for each point
                point_id = str(uuid.uuid4())
                product_id = str(product.get('product_id', str(uuid.uuid4())))
                name = str(product.get('name', ''))
                description = str(product.get('description', ''))
                
                # Get image URL from the product data
                image_url = product.get('image_url', '')
                image_data = None
                
                # If we have an image URL, download it and store the data
                if image_url and image_url.startswith('http'):
                    try:
                        # Download the image directly
                        response = requests.get(image_url, stream=True, timeout=10)
                        response.raise_for_status()
                        # Store the image data as base64
                        image_data = base64.b64encode(response.content).decode('utf-8')
                        logger.info(f"Downloaded image from {image_url}")
                    except Exception as e:
                        logger.warning(f"Failed to download image from {image_url}: {str(e)}")
                        image_url = ''  # Clear the URL if download fails
                
                # Enhanced category detection and normalization
                category_raw = str(product.get('category', '')).strip().lower()
                detected_categories = detect_jewelry_category(f"{name} {description} {category_raw}")
                
                # Use detected categories or fallback to raw category
                # Only use the primary/first category to avoid cross-category search issues
                if detected_categories:
                    # Use only the first/primary category for more accurate search results
                    category = detected_categories[0]
                else:
                    # If no category detected, use the raw category or default to 'jewelry'
                    # If raw category contains multiple comma-separated values, only use the first one
                    category = category_raw.split(',')[0].strip() if category_raw else 'jewelry'
                
                # Ensure category is properly formatted for indexing
                category = ' '.join(word.strip() for word in category.split() if word.strip())
                
                # Generate text embedding with category emphasis
                text_to_embed = f"{category} {name} {description}".strip()
                
                # Add product type emphasis based on detected category
                if category in ['necklace', 'ring', 'bracelet', 'earring', 'anklet', 'watch']:
                    text_to_embed = f"{category} jewelry {text_to_embed}"
                
                if not text_to_embed:
                    logger.warning(f"Skipping product with empty data: {product_id}")
                    continue
                
                # Generate text embedding
                text_embedding = embed_text(text_to_embed)
                if not text_embedding:
                    logger.error(f"Failed to generate text embedding for product: {product_id}")
                    continue
                
                # Generate image embedding if we have image data
                image_embedding = None
                if image_data:
                    try:
                        # Generate image embedding from the downloaded image data
                        image_bytes = base64.b64decode(image_data)
                        image_embedding = embed_image_bytes(image_bytes)
                        if image_embedding and len(image_embedding) == 512:
                            logger.info("Generated valid image embedding")
                        else:
                            logger.warning("Failed to generate valid image embedding")
                            image_embedding = None
                    except Exception as e:
                        logger.warning(f"Failed to generate image embedding: {str(e)}")
                        image_embedding = None
                
                # Only add the product if we have at least a text embedding
                if text_embedding and len(text_embedding) == TEXT_EMBEDDING_DIM:
                    # Create vector dictionary with text and optional image embedding
                    vector_dict = {
                        'text': text_embedding
                    }
                    
                    # Add image vector if available
                    if image_embedding and len(image_embedding) == 512:
                        vector_dict['image'] = image_embedding
                    else:
                        logger.info(f"No image embedding for product {product_id}, proceeding with text embedding only")
                    
                    # Add product to points list
                    points.append({
                        'id': point_id,
                        'payload': {
                            'product_id': product_id,
                            'name': name,
                            'description': description,
                            'category': category,  # Store normalized category
                            'original_category': category_raw,  # Store original category for reference
                            'price': str(product.get('price', 'N/A')),
                            'image_url': image_url,
                            'image_data': image_data if image_data else ''
                        },
                        'vector': vector_dict
                    })
                    logger.info(f"Added product {product_id} with normalized category '{category}' and valid embeddings")
                else:
                    logger.warning(f"Skipping product {product_id} due to invalid text embedding")
                
            except Exception as e:
                logger.error(f"Error processing product {idx}: {str(e)}")
                continue
        
        # Create collection with both text and image vectors
        try:
            vector_store.create_collection(collection_name=collection_name, recreate=True)
            logger.info(f"Created collection {collection_name} with text vectors of size {TEXT_EMBEDDING_DIM} and image vectors of size {IMAGE_EMBEDDING_DIM}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return jsonify({"error": f"Failed to create collection: {str(e)}"}), 500
        
        # Process products in batches
        # Use smaller batch size for uploads with many products and no images
        batch_size = 3 if len(points) > 5 and all('image' not in p['vector'] for p in points) else 5
        logger.info(f"Using batch size of {batch_size} for {len(points)} products")
        
        success_count = 0
        error_count = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                # Log batch details for debugging
                for point in batch:
                    logger.info(f"Point ID: {point['id']}")
                    logger.info(f"Text vector length: {len(point['vector']['text'])}")
                    if 'image' in point['vector']:
                        logger.info(f"Image vector length: {len(point['vector']['image'])}")
                    else:
                        logger.info(f"No image vector for point {point['id']}")
                
                # Add batch to Qdrant
                success = vector_store.upsert_points(collection_name, batch)
                if success:
                    success_count += len(batch)
                    logger.info(f"Successfully added batch {i//batch_size + 1}")
                else:
                    error_count += len(batch)
                    logger.error(f"Failed to add batch {i//batch_size + 1}")
            except Exception as e:
                error_msg = f"Error adding batch {i//batch_size + 1}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                error_count += len(batch)
        
        # Return success/error counts
        return jsonify({
            "message": f"Successfully processed {success_count} products. {error_count} failed.",
            "success_count": success_count,
            "error_count": error_count
        })
        
    except Exception as e:
        logger.error(f"Error in upload_products: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
@login_required
def query_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        query = data.get('query')
        session_id = data.get('session_id')
        
        if not query or not session_id:
            return jsonify({"error": "Missing query or session_id"}), 400
            
        logger.info(f"Received text query: '{query}' for session: {session_id}")
        
        # Get session data
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "No products found for this session"}), 404
            
        collection_name = session_data['collection_name']
        
        # Enhanced query classification
        query_type = classify_query(query)
        logger.info(f"Query classified as: {query_type}")
        
        if query_type == "price_query":
            return handle_price_query(query, session_id, collection_name)
        elif query_type == "specific_product":
            return handle_specific_product_query(query, session_id, collection_name)
        elif query_type == "question":
            return handle_product_question(query, session_id, collection_name)
        else:  # general_search
            return handle_product_search(query, session_id, collection_name)
            
    except Exception as e:
        logger.error(f"Error in query_text endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e)
        }), 500

@app.route('/debug', methods=['GET'])
def debug_endpoint():
    try:
        # Add debug logic here
        return jsonify({"message": "Debug endpoint reached successfully"}), 200
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def classify_query(query):
    """
    Enhanced query classification with better handling of different query types
    """
    query_lower = query.lower().strip()
    
    # Handle browsing queries first
    if any(word in query_lower for word in ['show me', 'display', 'list', 'browse', 'all']):
        return "general_search"
    
    # Enhanced price query patterns - more comprehensive
    price_patterns = [
        r'\b(?:tell me |what\'s |what is |show me )?(?:the )?price of\b',
        r'\b(?:tell me |what\'s |what is )?(?:the )?cost of\b',
        r'\bhow much (?:is|are|does|do)\b',
        r'\bwhat (?:does|do) (?:the |this |these )?.*cost\b',
        r'\bwhat (?:is|are) (?:the )?price(?:s)?\b',
        r'\bhow much (?:for|is) (?:the |this |these )?\b',
        r'\bcost for (?:the |this |these )?\b',
        r'\bprice for (?:the |this |these )?\b'
    ]
    
    # Check if it matches any price pattern
    for pattern in price_patterns:
        if re.search(pattern, query_lower):
            logger.info(f"Query classified as price query using pattern: {pattern}")
            return "price_query"
    
    # Simple price indicators (fallback)
    simple_price_words = ['price', 'cost', 'expensive', 'cheap', 'money', 'dollar', 'rupee']
    if any(word in query_lower for word in simple_price_words):
        # Additional check to avoid false positives
        if not any(word in query_lower for word in ['show', 'find', 'search', 'browse', 'all']):
            logger.info(f"Query classified as price query using simple indicators")
            return "price_query"
    
    # Question patterns
    question_words = ["what", "where", "when", "how", "why", "which", "who"]
    if any(query_lower.startswith(word) for word in question_words) or "?" in query:
        return "question"
    
    # Specific product patterns (usually 2-4 words, specific items)
    words = query_lower.split()
    
    if 2 <= len(words) <= 4:
        # Look for specific product indicators
        specific_indicators = [
            r'\b(?:diamond|ruby|emerald|sapphire|pearl|gold|silver|platinum)\s+(?:stud|ring|necklace|bracelet|earring|pendant)\b',
            r'\b(?:classic|vintage|modern|antique|designer)\s+(?:stud|ring|necklace|bracelet|earring|pendant)\b',
            r'\b(?:stud|ring|necklace|bracelet|earring|pendant)\s+(?:set|pair|collection)\b'
        ]
        
        for pattern in specific_indicators:
            if re.search(pattern, query_lower):
                return "specific_product"
        
        # If it's just a jewelry category + descriptor, treat as specific
        categories = ['stud', 'studs', 'ring', 'rings', 'necklace', 'necklaces', 
                     'bracelet', 'bracelets', 'earring', 'earrings', 'pendant', 'pendants']
        
        if any(cat in words for cat in categories):
            return "specific_product"
    
    return "general_search"


def handle_price_query(query, session_id, collection_name):
    """Enhanced price query handler with better matching strategies"""
    try:
        # Extract product name from the price query
        query_lower = query.lower().strip()
        
        # Remove price indicators to extract the product name
        price_indicators = [
            "tell me the price of", "what's the price of", "what is the price of",
            "price of", "cost of", "how much is", "what does the", "cost for",
            "how much does", "price for", "how much are", "what do the"
        ]
        
        product_query = query
        for indicator in price_indicators:
            if indicator in query_lower:
                # Find the position and extract everything after the indicator
                pos = query_lower.find(indicator)
                product_query = query[pos + len(indicator):].strip()
                break
        
        # Clean up the extracted product name
        product_query = product_query.replace('?', '').replace('the ', '').strip()
        
        logger.info(f"Price query - original: '{query}', extracted: '{product_query}'")
        
        # Multiple search strategies for better matching
        
        # Strategy 1: Direct text search with multiple variations
        search_variations = [
            product_query,
            product_query.lower(),
            product_query.title(),
            # Try with common word variations
            product_query.replace('earring', 'earrings'),
            product_query.replace('earrings', 'earring'),
            product_query.replace('ring', 'rings'),
            product_query.replace('rings', 'ring'),
            product_query.replace('necklace', 'necklaces'),
            product_query.replace('necklaces', 'necklace')
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        search_variations = [x for x in search_variations if not (x in seen or seen.add(x))]
        
        best_match = None
        best_score = 0.0
        
        for variation in search_variations:
            try:
                logger.info(f"Trying search variation: '{variation}'")
                
                # Generate embedding for this variation
                query_embedding = embed_text(variation)
                if not query_embedding:
                    continue
                
                # Search with a lower threshold to cast a wider net
                results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    vector_name="text",  # Specify the vector name
                    top_k=30,  # Get more results for better matching
                    score_threshold=0.1  # Very low threshold
                )
                
                if results:
                    logger.info(f"Found {len(results)} results for variation '{variation}'")
                    
                    # Check for exact matches first
                    for result in results:
                        payload = getattr(result, 'payload', {}) or {}
                        result_name = str(payload.get('name', '')).strip()
                        result_score = float(getattr(result, 'score', 0.0))
                        
                        # Multiple exact match strategies
                        exact_match_found = False
                        
                        # Strategy A: Exact case-insensitive match
                        if result_name.lower() == variation.lower():
                            logger.info(f"Exact match found: '{result_name}' matches '{variation}'")
                            exact_match_found = True
                        
                        # Strategy B: Check if the search term is contained in the product name
                        elif variation.lower() in result_name.lower():
                            # Additional validation: ensure it's not just a partial word match
                            words_in_query = set(variation.lower().split())
                            words_in_result = set(result_name.lower().split())
                            
                            # Check if all words from query are in the result
                            if words_in_query.issubset(words_in_result):
                                logger.info(f"Subset match found: '{result_name}' contains all words from '{variation}'")
                                exact_match_found = True
                        
                        # Strategy C: Check if result name is contained in search term (for longer queries)
                        elif result_name.lower() in variation.lower():
                            logger.info(f"Reverse subset match found: '{variation}' contains '{result_name}'")
                            exact_match_found = True
                        
                        if exact_match_found:
                            if result_score > best_score or best_match is None:
                                best_match = result
                                best_score = result_score
                                logger.info(f"New best match: '{result_name}' with score {result_score}")
                
                # If we found a good match, no need to try more variations
                if best_match and best_score > 0.8:
                    break
                    
            except Exception as e:
                logger.warning(f"Error searching with variation '{variation}': {str(e)}")
                continue
        
        # Strategy 2: If no exact match found, try fuzzy matching with high similarity
        if not best_match:
            logger.info("No exact match found, trying high similarity matching")
            
            query_embedding = embed_text(product_query)
            if query_embedding:
                results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    vector_name="text",  # Specify the vector name
                    top_k=20,
                    score_threshold=0.2
                )
                
                if results:
                    # Look for very high similarity matches (85%+)
                    for result in results:
                        score = float(getattr(result, 'score', 0.0))
                        if score >= 0.85:
                            payload = getattr(result, 'payload', {}) or {}
                            result_name = str(payload.get('name', '')).strip()
                            logger.info(f"High similarity match: '{result_name}' with score {score}")
                            
                            if score > best_score or best_match is None:
                                best_match = result
                                best_score = score
        
        # Return the best match if found
        if best_match:
            formatted_result = format_product_results([best_match], session_id)
            if formatted_result:
                product = formatted_result[0]
                product_name = product.get('name', 'this item')
                product_price = product.get('price', 'not available')
                
                logger.info(f"Returning price for '{product_name}': {product_price}")
                
                return jsonify({
                    "message": f"The price of {product_name} is {product_price}.",
                    "results": [product],
                    "exact_match": best_score > 0.8,
                    "query_type": "price",
                    "confidence_score": round(best_score * 100, 1)
                })
        
        # No match found - provide helpful suggestion
        logger.warning(f"No match found for price query: '{product_query}'")
        
        # Try to suggest similar products by doing a broader search
        query_embedding = embed_text(product_query)
        if query_embedding:
            similar_results = vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                vector_name="text",  # Specify the vector name
                top_k=3,
                score_threshold=0.3
            )
            
            if similar_results:
                formatted_results = format_product_results(similar_results, session_id)
                if formatted_results:
                    suggestion_names = [p.get('name', '') for p in formatted_results[:2]]
                    suggestions = ', '.join([f"'{name}'" for name in suggestion_names if name])
                    
                    return jsonify({
                        "message": f"I couldn't find the exact price for '{product_query}'. Did you mean: {suggestions}? Please use the exact product name.",
                        "results": formatted_results[:2],  # Show top 2 suggestions
                        "exact_match": False,
                        "query_type": "price",
                        "suggestions": True
                    })
        
        return jsonify({
            "message": f"I couldn't find '{product_query}' in the catalog. Please check the spelling or browse the products to find the exact name.",
            "results": [],
            "exact_match": False,
            "query_type": "price"
        })
        
    except Exception as e:
        logger.error(f"Error in handle_price_query: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your price query"}), 500


def handle_specific_product_query(query, session_id, collection_name):
    """Handle specific product queries with exact matching priority"""
    try:
        logger.info(f"Specific product query: '{query}'")
        
        # Generate embedding for the query
        query_embedding = embed_text(query)
        if not query_embedding:
            return jsonify({"error": "Failed to process your query"}), 500
        
        # Get more results for better exact matching
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=25,
            score_threshold=0.3
        )
        
        if not results:
            return jsonify({
                "message": f"No products found matching '{query}'. Try browsing by category or using different keywords.",
                "results": [],
                "exact_match": False,
                "query_type": "specific_product"
            })
        
        query_lower = query.lower().strip()
        best_matches = []
        
        # Stage 1: Look for exact name matches
        for result in results:
            payload = getattr(result, 'payload', {}) or {}
            result_name = str(payload.get('name', '')).lower().strip()
            
            if result_name == query_lower:
                best_matches.append(result)
                break  # Found exact match, no need to continue
        
        # Stage 2: Look for very high similarity matches if no exact match
        if not best_matches:
            for result in results:
                if hasattr(result, 'score') and result.score >= 0.85:
                    best_matches.append(result)
                    if len(best_matches) >= 3:  # Limit to top 3 high-confidence matches
                        break
        
        # Stage 3: If still no good matches, check for partial name matches
        if not best_matches:
            query_words = set(query_lower.split())
            for result in results:
                payload = getattr(result, 'payload', {}) or {}
                result_name = str(payload.get('name', '')).lower().strip()
                result_words = set(result_name.split())
                
                # Check if all query words are in the result name
                if query_words.issubset(result_words) or result_words.issubset(query_words):
                    best_matches.append(result)
                    if len(best_matches) >= 2:  # Limit for partial matches
                        break
        
        # Stage 4: Last resort - category matching with high scores
        if not best_matches:
            detected_categories = detect_jewelry_category(query)
            if detected_categories:
                category = detected_categories[0].lower()
                for result in results[:10]:  # Only check top 10
                    payload = getattr(result, 'payload', {}) or {}
                    result_category = str(payload.get('category', '')).lower().strip()
                    
                    if category in result_category and hasattr(result, 'score') and result.score >= 0.6:
                        best_matches.append(result)
                        if len(best_matches) >= 5:  # More results for category fallback
                            break
        
        if best_matches:
            formatted_results = format_product_results(best_matches, session_id)
            
            if len(formatted_results) == 1:
                # Single result - likely what they were looking for
                product = formatted_results[0]
                return jsonify({
                    "message": f"Found: {product.get('name', 'Product')} - {product.get('price', 'Price not available')}",
                    "results": formatted_results,
                    "exact_match": True,
                    "query_type": "specific_product"
                })
            else:
                # Multiple results - show them but indicate it's a specific search
                return jsonify({
                    "message": f"Found {len(formatted_results)} products matching '{query}':",
                    "results": formatted_results,
                    "exact_match": False,
                    "query_type": "specific_product"
                })
        
        return jsonify({
            "message": f"No products found matching '{query}'. Try different keywords or browse by category.",
            "results": [],
            "exact_match": False,
            "query_type": "specific_product"
        })
        
    except Exception as e:
        logger.error(f"Error in handle_specific_product_query: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your query"}), 500


def detect_jewelry_category(text):
    """
    Enhanced jewelry category detection with better accuracy
    """
    if not text or not isinstance(text, str):
        return []
        
    text_lower = text.lower().strip()
    
    # Define comprehensive jewelry categories with weighted patterns
    category_patterns = {
        'necklace': [
            (r'\bnecklaces?\b', 1.0),
            (r'\bpendants?\b', 1.0),
            (r'\bchains?\b', 0.95),
            (r'\bchokers?\b', 1.0),
            (r'\blockets?\b', 1.0),
            (r'\b(?:gold|silver|diamond|pearl)\s+necklaces?\b', 1.0),
        ],
        'ring': [
            (r'\brings?\b', 1.0),
            (r'\b(?:wedding|engagement)\s+rings?\b', 1.0),
            (r'\bbands?\b', 0.85),
            (r'\b(?:diamond|gold|silver|platinum)\s+rings?\b', 1.0),
        ],
        'earring': [
            (r'\bearrings?\b', 1.0),
            (r'\bstuds?\b', 0.95),
            (r'\bhoops?\b', 0.95),
            (r'\b(?:dangle|drop|chandelier)\s+earrings?\b', 1.0),
        ],
        'bracelet': [
            (r'\bbracelets?\b', 1.0),
            (r'\bbangles?\b', 1.0),
            (r'\bcuffs?\b', 0.9),
            (r'\b(?:charm|tennis|chain|bead)\s+bracelets?\b', 1.0),
        ],
        'anklet': [
            (r'\banklets?\b', 1.0),
            (r'\bankle\s+(?:chains?|bracelets?)\b', 1.0),
        ],
        'watch': [
            (r'\bwatches?\b', 1.0),
            (r'\bwristwatch(?:es)?\b', 1.0),
        ],
        'brooch': [
            (r'\bbrooches?\b', 1.0),
            (r'\bpins?\b', 0.8),
        ]
    }
    
    # Calculate scores for each category
    category_scores = {}
    
    for category, patterns in category_patterns.items():
        max_score = 0.0
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                score = len(matches) * weight
                max_score = max(max_score, score)
        
        if max_score > 0:
            category_scores[category] = max_score
    
    # Return only the highest scoring category
    if category_scores:
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_category, top_score = sorted_categories[0]
        if top_score >= 0.85:  # Confidence threshold
            return [top_category]
    
    return []


def handle_product_search(query: str, session_id: str, collection_name: str):
    """
    Enhanced product search with STRICT category filtering
    """
    try:
        # Verify session exists
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            logger.error(f"Session not found or invalid: {session_id}")
            return jsonify({"error": "Session not found", "success": False}), 404
            
        # Check if collection exists and has points
        try:
            collection_info = vector_store.client.get_collection(collection_name)
            if collection_info.points_count == 0:
                logger.warning(f"Collection {collection_name} is empty")
                return jsonify({
                    "message": "No products found in your catalog. Please upload products first.",
                    "results": [],
                    "success": False
                })
        except Exception as e:
            logger.error(f"Error checking collection {collection_name}: {str(e)}")
            return jsonify({
                "error": f"Collection error: {str(e)}",
                "success": False
            }), 500
            
        logger.info(f"Processing search query: '{query}' for session: {session_id}")
        
        # Clean the query and remove browsing words
        clean_query = query.lower().strip()
        browsing_words = ['show', 'display', 'list', 'find', 'search', 'browse', 'get', 'give', 'me', 'all', 'the', 'some']
        query_words = clean_query.split()
        content_words = [word for word in query_words if word not in browsing_words]
        search_term = ' '.join(content_words) if content_words else query
        logger.info(f"Search term after cleaning: '{search_term}'")
        
        # Detect category with improved accuracy
        detected_categories = detect_jewelry_category(search_term)
        logger.info(f"Detected categories: {detected_categories}")
        
        if not detected_categories:
            logger.warning(f"No specific category detected for query: '{query}'")
            return jsonify({
                "message": "Please specify a jewelry category (e.g., rings, necklaces, earrings, bracelets) for better search results.",
                "results": [],
                "success": False
            })
        
        primary_category = detected_categories[0]
        logger.info(f"Primary category: {primary_category}")
        
        # Generate embedding with heavy category emphasis
        enhanced_query = f"{primary_category} {primary_category} jewelry {search_term}"
        
        # Generate embedding for the search query
        query_embedding = embed_text(enhanced_query)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return jsonify({"error": "Failed to process search query"}), 500
        
        # STRICT CATEGORY SEARCH - Single approach with strict filtering
        results = []
        
        try:
            # First, try to get exact matches for the category
            logger.info(f"Searching for products in category: {primary_category}")
            
            # Search with strict category filtering
            search_results = vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=50,  # Get more results for filtering
                score_threshold=0.1,  # Lower threshold to catch more potential matches
                filter_conditions={"category": primary_category}
            )
            
            logger.info(f"Initial search returned {len(search_results) if search_results else 0} results")
            
            # If no results, try with a more lenient search
            if not search_results:
                logger.info("No results with exact category match, trying more lenient search")
                search_results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    top_k=100,  # Get even more results
                    score_threshold=0.0,  # No score threshold
                    filter_conditions={"category": {"$regex": f"{primary_category}", "$options": "i"}}
                )
                logger.info(f"Lenient search returned {len(search_results) if search_results else 0} results")
            
            # Strict category filtering
            filtered_results = []
            seen_products = set()
            
            for result in search_results or []:
                try:
                    payload = getattr(result, 'payload', {}) or {}
                    if not payload:
                        continue
                    
                    # Get result details
                    result_category = str(payload.get('category', '')).lower().strip()
                    result_name = str(payload.get('name', '')).lower()
                    product_id = str(payload.get('product_id', ''))
                    
                    # Skip if missing essential data
                    if not all([result_category, result_name, product_id]):
                        logger.debug(f"Skipping result with missing data: {payload}")
                        continue
                    
                    # Avoid duplicates
                    if product_id in seen_products:
                        continue
                    
                    # STRICT CATEGORY MATCHING - Only accept exact or singular/plural matches
                    is_correct_category = (
                        result_category == primary_category or  # Exact match
                        result_category == f"{primary_category}s" or  # Plural form
                        result_category == primary_category[:-1] or  # Singular form if primary is plural
                        (primary_category.endswith('s') and result_category == primary_category[:-1])  # Handle plural/singular
                    )
                    
                    # If category doesn't match, REJECT the result
                    if not is_correct_category:
                        logger.debug(f"✗ REJECTED - Category mismatch: '{result_category}' != '{primary_category}' for product '{result_name}'")
                        continue
                    
                    # Category matches - add to results
                    seen_products.add(product_id)
                    filtered_results.append(result)
                    logger.debug(f"✓ ACCEPTED - '{result_name}' (category: '{result_category}')")
                    
                    # Stop when we have enough results
                    if len(filtered_results) >= 15:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing search result: {str(e)}")
                    continue
            
            results = filtered_results
            logger.info(f"Strict filtering kept {len(results)} results out of {len(search_results)}")
            
        except Exception as e:
            logger.error(f"Error in product search: {str(e)}", exc_info=True)
            results = []
        
        # Format and return results
        if results:
            formatted_results = format_product_results(results, session_id)
            
            # Sort by relevance score
            formatted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Create response message
            category_name = primary_category.title()
            if any(word in query.lower() for word in ['show', 'display', 'list', 'browse']):
                message = f"Here are {len(formatted_results)} {category_name}s from our collection:"
            else:
                message = f"Found {len(formatted_results)} {category_name}s matching your search:"
            
            logger.info(f"Returning {len(formatted_results)} strictly filtered results")
            
            return jsonify({
                "results": formatted_results,
                "message": message,
                "detected_categories": [primary_category],
                "query_type": "category_search",
                "success": True
            })
        else:
            # No results found
            logger.warning(f"No {primary_category}s found")
            
            # Check if the collection has any products at all
            try:
                total_count = vector_store.client.count(collection_name=collection_name, exact=True)
                total_products = getattr(total_count, 'count', 0)
                
                if total_products == 0:
                    message = "No products found in your catalog. Please upload products first."
                else:
                    message = f"No {primary_category}s found in your catalog. Available categories might include rings, necklaces, earrings, or bracelets."
                    
            except Exception as e:
                logger.error(f"Error checking total product count: {str(e)}")
                message = f"No {primary_category}s found. Please check if you have uploaded products in this category."
            
            return jsonify({
                "message": message,
                "results": [],
                "detected_categories": [primary_category],
                "success": False
            })
            
    except Exception as e:
        logger.error(f"Error in handle_product_search: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"An error occurred while processing your search: {str(e)}",
            "success": False
        }), 500

def handle_product_question(question: str, session_id: str, collection_name: str):
    """Handle questions about products with category awareness"""
    try:
        # Verify session exists
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return {"error": "Session not found or invalid"}, 404
            
        # Get collection name from session if not provided
        if not collection_name:
            collection_name = session_data.get('collection_name')
            if not collection_name:
                return {"error": "No collection associated with this session"}, 400
        
        # Process the question
        question_lower = question.lower()
        
        # Check for common questions
        if any(q in question_lower for q in ["how many products", "total products", "number of products"]):
            # Count total products in the collection
            count = vector_store.count_points(collection_name=collection_name)
            return {"message": f"There are {count} products in the catalog."}
        
        # Check for price queries
        price_indicators = ["price of", "cost of", "how much is", "what does the", "cost for", "what is the price of"]
        if any(indicator in question_lower for indicator in price_indicators):
            # Extract product name from question by removing price indicators and question marks
            product_query = question
            for indicator in price_indicators:
                if indicator in question_lower:
                    product_query = question[question_lower.find(indicator) + len(indicator):].strip(' .,?!')
                    break
            
            logger.info(f"Processing price query for: '{product_query}'")
            
            # Clean up the query - remove any remaining question marks and extra spaces
            product_query = product_query.replace('?', '').strip()
            
            # Search for products and look for exact name matches
            search_results = vector_store.search(
                collection_name=collection_name,
                query_vector=embed_text(product_query),
                top_k=20,  # Get more results to find exact matches
                score_threshold=0.3  # Lower threshold to cast a wider net
            )
            
            if search_results:
                # First, look for exact name matches (case-insensitive)
                query_lower = product_query.lower().strip()
                
                for result in search_results:
                    payload = getattr(result, 'payload', {}) or {}
                    result_name = str(payload.get('name', '')).lower().strip()
                    
                    # Check for exact match (case-insensitive)
                    if result_name == query_lower:
                        formatted_results = format_product_results([result], session_id)
                        if formatted_results:
                            product = formatted_results[0]
                            logger.info(f"Found exact name match for '{product_query}': {product.get('name')}")
                            return jsonify({
                                "message": f"The price of {product.get('name', 'this item')} is {product.get('price', 'not available')}.",
                                "results": [product],  # Only return the exact match
                                "exact_match": True
                            })
                
                # If no exact match, check for very high similarity (95%+) for price queries only
                for result in search_results:
                    if hasattr(result, 'score') and result.score >= 0.95:
                        formatted_results = format_product_results([result], session_id)
                        if formatted_results:
                            product = formatted_results[0]
                            logger.info(f"Found high-confidence match for '{product_query}': {product.get('name')} (score: {result.score:.3f})")
                            return jsonify({
                                "message": f"The price of {product.get('name', 'this item')} is {product.get('price', 'not available')}.",
                                "results": [product],  # Only return the high-confidence match
                                "exact_match": True
                            })
            
            # If we get here, no exact or high-confidence matches were found
            logger.warning(f"No exact matches found for price query: '{product_query}'")
            
            # For price queries, don't show multiple products - it's confusing
            return jsonify({
                "message": f"I couldn't find an exact match for '{product_query}'. Please check the spelling or try browsing the catalog to find the exact product name.",
                "results": [],
                "exact_match": False
            })
        
        # Detect categories from the question for more specific searches
        detected_categories = detect_jewelry_category(question)
        
        # Generate embedding for semantic search if needed
        question_embedding = embed_text(question)
        if not question_embedding:
            return {"error": "Failed to process your question"}, 500
            
        # Search for relevant products with category filtering if applicable
        if detected_categories:
            results = vector_store.search_with_category_filter(
                collection_name=collection_name,
                query_vector=question_embedding,
                categories=detected_categories,
                top_k=5,
                score_threshold=0.4
            )
            
            if results:
                formatted_results = format_product_results(results, session_id)
                return {
                    "message": f"Here are some {', '.join(detected_categories)} that might match your question:",
                    "results": formatted_results,
                    "detected_categories": detected_categories,
                    "exact_match": False
                }
        
        # Check if this is a query for a specific product (not a category or general question)
        # If it's a specific product query, try to find an exact match first
        specific_product_query = not detected_categories and not any(word in question_lower for word in ["what", "where", "when", "how", "why", "?"])
        
        if specific_product_query:
            # Try exact name match first
            exact_match = vector_store.search_with_name(
                collection_name=collection_name,
                product_name=question,
                exact_match=True
            )
            
            if exact_match:
                formatted_results = format_product_results(exact_match, session_id)
                if formatted_results:
                    product = formatted_results[0]
                    logger.info(f"Found exact match for '{question}': {product.get('name')}")
                    return {
                        "message": f"Here is the {product.get('name', 'product')} you asked for.",
                        "results": [product],  # Only return the exact match
                        "exact_match": True
                    }
        
        # If no exact match or it's not a specific product query, try a general search
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            top_k=5,
            score_threshold=0.4
        )
        
        if results:
            # For specific product queries, only return the top result
            if specific_product_query:
                formatted_results = format_product_results([results[0]], session_id)
                if formatted_results and len(formatted_results) > 0:
                    product = formatted_results[0]
                    return {
                        "message": f"Here is the closest match for '{question}'.",
                        "results": [product],  # Only return the top match
                        "exact_match": False
                    }
            else:
                # For general queries, return multiple results
                formatted_results = format_product_results(results, session_id)
                if formatted_results and len(formatted_results) > 0:
                    top_result = formatted_results[0]
                    answer = top_result.get('name', 'I found some products')
                    if top_result.get('description'):
                        answer += f" - {top_result['description'][:100]}..."
                    
                    return {
                        "message": answer,
                        "results": formatted_results,
                        "detected_categories": detected_categories or [],
                        "exact_match": False
                    }
        
        # Default response if no products found
        return {
            "message": "I couldn't find any products matching your question. " \
                      "Please try asking about specific products or categories.",
            "results": [],
            "detected_categories": detected_categories or [],
            "exact_match": False
        }
        
    except Exception as e:
        logger.error(f"Error handling product question: {str(e)}", exc_info=True)
        return {
            "error": "An error occurred while processing your question",
            "results": []
        }, 500

def format_product_results(results, session_id):
    """
    Format product results for the frontend with proper image URLs from Qdrant
    """
    if not results:
        return []
        
    formatted_results = []
    seen_products = set()  # Track seen products for deduplication
    
    for result in results:
        try:
            # Handle different result formats
            if hasattr(result, 'payload') and hasattr(result, 'score'):
                # Qdrant result object
                payload = result.payload or {}
                score = float(getattr(result, 'score', 0.0))
                result_id = str(getattr(result, 'id', ''))
            elif isinstance(result, dict):
                # Dictionary format
                if 'payload' in result and 'score' in result:
                    payload = result.get('payload', {})
                    score = float(result.get('score', 0.0))
                    result_id = str(result.get('id', ''))
                else:
                    # If result is already in product format
                    if all(k in result for k in ['name', 'price', 'category']):
                        # Add session_id to image URL if it exists
                        if 'image_url' in result and result['image_url']:
                            if not result['image_url'].startswith('http') and not result['image_url'].startswith('/api/'):
                                result['image_url'] = f"/api/images/{result.get('id', result.get('product_id', ''))}?session_id={session_id}"
                        formatted_results.append(result)
                        continue
                    payload = result
                    score = 1.0  # Default score for direct product dicts
                    result_id = str(result.get('id', str(uuid.uuid4())))
            else:
                logger.warning(f"Unexpected result format: {type(result)}")
                continue
            
            if not isinstance(payload, dict):
                logger.warning(f"Invalid payload type: {type(payload)}")
                continue
            
            # Create more specific unique identifier for deduplication
            product_key = f"{payload.get('name', '')}_{payload.get('category', '')}_{payload.get('price', '')}"
            if not product_key.strip('_') or product_key in seen_products:
                continue
                
            seen_products.add(product_key)
            
            # Get and normalize category
            category = str(payload.get('category', '')).lower().strip()
            if not category:
                category = 'uncategorized'
            
            # Convert score to percentage format
            percentage_score = min(100.0, max(0.0, score * 100))
            
            # Extract product details with proper defaults
            product_id = str(payload.get('product_id', result_id))
            
            product = {
                'id': product_id,
                'name': str(payload.get('name', 'Unnamed Product')).strip(),
                'description': str(payload.get('description', '')).strip(),
                'price': str(payload.get('price', 'N/A')).strip(),
                'category': category,  # Use normalized category
                'score': round(percentage_score, 2),  # Round to 2 decimal places
                'original_category': str(payload.get('category', '')).strip()  # Keep original for reference
            }
            
            # Handle image URL - prioritize Qdrant image data over external URLs
            image_data = payload.get('image_data', '')
            if image_data and image_data.strip():
                # Image is stored in Qdrant, use our API endpoint
                product['image_url'] = f"/api/images/{product_id}?session_id={session_id}"
                product['has_image'] = True
            else:
                # Fallback to external image URL if available
                external_url = str(payload.get('image_url', '')).strip()
                if external_url and external_url.startswith(('http://', 'https://')):
                    product['image_url'] = external_url
                    product['has_image'] = True
                else:
                    product['image_url'] = ''
                    product['has_image'] = False
                
            formatted_results.append(product)
            
        except Exception as e:
            logger.error(f"Error formatting product result: {str(e)}\nResult: {result}", exc_info=True)
            continue
    
    # Sort by score in descending order, then by name for consistent ordering
    formatted_results.sort(key=lambda x: (-x.get('score', 0), x.get('name', '').lower()))
    
    return formatted_results

@app.route("/query_image/<session_id>", methods=["POST"])
@login_required
def query_image(session_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the file directly into memory instead of saving to disk
        image_data = file.read()
        if not image_data:
            raise Exception("File is empty")
            
        # Get session and collection
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "No products found for this session"}), 404
            
        collection_name = session_data['collection_name']
        
        # Enhanced category detection from multiple sources with priority
        filename = file.filename or ""
        user_context = request.form.get('context', '').lower()
        query_text = request.form.get('query', '').lower()  # Additional query text if provided
        
        # Priority 1: Check explicit category in the query text first
        detected_categories = []
        
        # Check for explicit category mentions in the query
        category_keywords = {
            'ring': ['\bring\b', '\bbands?\b', '\bengagement\b', '\bwedding band\b', '\bsignet\b'],
            'earring': ['\bearrings?\b', '\bstuds?\b', '\bhoops?\b', '\bdangle\b', '\bear rings?\b', '\bear-rings?\b'],
            'necklace': ['\bnecklaces?\b', '\bpendants?\b', '\bchains?\b', '\bchokers?\b', '\blockets?\b', '\bcollars?\b'],
            'bracelet': ['\bbracelets?\b', '\bbangles?\b', '\bcuffs?\b', '\bcharms?\b', '\btennis\b', '\bchain bracelets?\b'],
            'set': ['\bsets?\b', '\bmatching sets?\b', '\bjewelry sets?\b', '\bpairs?\b']
        }
        
        # Check query text first (highest priority)
        for category, keywords in category_keywords.items():
            if any(re.search(rf'\b{kw}\b', query_text) for kw in keywords):
                detected_categories = [category]
                logger.info(f"Detected category '{category}' from query text")
                break
        
        # If no category in query, check filename
        if not detected_categories and filename:
            filename_lower = filename.lower()
            for category, keywords in category_keywords.items():
                if any(kw in filename_lower for kw in [k.replace('\\b', '') for k in keywords]):
                    detected_categories = [category]
                    logger.info(f"Detected category '{category}' from filename")
                    break
        
        # If still no category, check user context
        if not detected_categories and user_context:
            for category, keywords in category_keywords.items():
                if any(re.search(rf'\b{kw}\b', user_context) for kw in keywords):
                    detected_categories = [category]
                    logger.info(f"Detected category '{category}' from user context")
                    break
        
        # If still no category, use the extract_category_from_image_query as fallback
        if not detected_categories:
            all_context = f"{filename} {user_context} {query_text}".strip()
            if all_context:
                detected_categories = extract_category_from_image_query(all_context)
        
        logger.info(f"Image search context: filename='{filename}', query='{query_text}', "
                  f"user_context='{user_context}', detected_categories={detected_categories}")
        
        # Generate embedding for the image
        try:
            # Generate image embedding
            image_embedding = embed_image_bytes(image_data)
            
            if image_embedding is None or len(image_embedding) != 512:
                raise ValueError("Failed to generate valid image embedding")
            
            # Multi-stage search approach for better results
            results = None
            
            # Stage 1: Always try category-filtered search first if we detected categories
            results = None
            if detected_categories:
                try:
                    logger.info(f"Stage 1: Strict category filtering for: {detected_categories}")
                    # First try with a higher threshold for better precision
                    # Log the exact categories we're searching for
                    logger.info(f"Searching with strict category filter for: {detected_categories}")
                    
                    # First try with a higher threshold for better precision
                    results = vector_store.search_with_category_filter(
                        collection_name=collection_name,
                        query_vector=image_embedding,
                        vector_name="image",
                        categories=detected_categories,
                        top_k=15,  # Get more results to ensure we have enough after filtering
                        score_threshold=0.6  # Higher threshold for better precision
                    )
                    
                    if results:
                        logger.info(f"Found {len(results)} results with strict category filter")
                        
                        # Additional client-side filtering for extra safety
                        filtered_results = []
                        for result in results:
                            payload = getattr(result, 'payload', {}) or {}
                            result_category = str(payload.get('category', '')).lower()
                            
                            # Skip if no category is available
                            if not result_category:
                                logger.debug(f"Skipping result with no category: {payload.get('name', 'Unnamed')}")
                                continue
                            
                            # Split categories and clean them up
                            result_categories = [cat.strip().lower() for cat in result_category.split(',') if cat.strip()]
                            
                            # Check if any of the result's categories exactly match any of the target categories
                            is_category_match = any(
                                any(
                                    detected_cat.lower() == result_cat
                                    for result_cat in result_categories
                                )
                                for detected_cat in detected_categories
                            )
                            
                            if is_category_match:
                                filtered_results.append(result)
                                logger.debug(f"Added product to results: {payload.get('name', 'Unnamed')} with categories: {result_categories}")
                                if len(filtered_results) >= 10:  # Limit to top 10 matches
                                    break
                            else:
                                logger.debug(f"Excluded product due to category mismatch: {payload.get('name', 'Unnamed')} with categories: {result_categories}")
                        
                        results = filtered_results
                        logger.info(f"Kept {len(results)} results after exact category matching")
                    else:
                        logger.info("No results with strict category filter")
                
                except Exception as e:
                    logger.warning(f"Category filter search failed: {str(e)}", exc_info=True)
                    results = None
            
            # Stage 2: Broader image search with relaxed category filtering
            if not results and detected_categories:
                logger.info(f"Stage 2: Relaxed category filtering for: {detected_categories}")
                try:
                    # Try with a lower threshold but still with category filtering
                    results = vector_store.search_with_category_filter(
                        collection_name=collection_name,
                        query_vector=image_embedding,
                        vector_name="image",
                        categories=detected_categories,
                        top_k=10,
                        score_threshold=0.4  # Slightly lower threshold
                    )
                    logger.info(f"Found {len(results) if results else 0} results with relaxed category filter")
                except Exception as e:
                    logger.warning(f"Relaxed category filter search failed: {str(e)}", exc_info=True)
            
            # Stage 3: Last resort - search without any category filtering
            if not results:
                logger.info("Stage 3: Fallback search without any category filtering")
                results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=image_embedding,
                    vector_name="image",
                    top_k=10,
                    score_threshold=0.1  # Even lower threshold for broader search
                )
            
            # Stage 3: Final filtering and validation of results
            if results:
                logger.info("Stage 3: Final result validation and filtering")
                
                valid_results = []
                seen_products = set()
                
                for result in results:
                    try:
                        payload = getattr(result, 'payload', {}) or {}
                        if not payload:
                            continue
                            
                        # Extract and validate product details
                        product_id = str(payload.get('product_id', '')).strip()
                        name = str(payload.get('name', '')).strip()
                        category = str(payload.get('category', '')).lower().strip()
                        
                        # Skip if missing critical fields
                        if not all([name, category, product_id]):
                            logger.debug(f"Skipping result - missing required fields: {payload}")
                            continue
                        
                        # Create a unique key for deduplication
                        product_key = f"{name}_{category}"
                        if product_key in seen_products:
                            continue
                            
                        seen_products.add(product_key)
                        
                        # Strict category validation if we have detected categories
                        if detected_categories:
                            # Normalize the result's category for comparison
                            result_categories = [c.strip().lower() for c in category.split(',')]
                            
                            # Check if any detected category EXACTLY matches the result's categories
                            # Only allow exact matches, not partial matches
                            category_match = False
                            for detected_cat in detected_categories:
                                for result_cat in result_categories:
                                    if detected_cat == result_cat:  # Exact match only
                                        category_match = True
                                        break
                                if category_match:
                                    break
                            
                            if not category_match:
                                logger.debug(f"Skipping result - category mismatch: {category} not in {detected_categories}")
                                continue
                        
                        # Enhanced image validation
                        has_image = False
                        image_url = ''
                        
                        # Check for base64 image data first (highest priority)
                        if payload.get('image_data'):
                            try:
                                # Basic validation of base64 data
                                if isinstance(payload['image_data'], str) and len(payload['image_data']) > 100:
                                    has_image = True
                                    image_url = f"/api/images/{product_id}?session_id={session_id}"
                            except Exception as e:
                                logger.warning(f"Invalid image data for product {product_id}: {str(e)}")
                        
                        # Fall back to URL if no embedded image
                        if not has_image and payload.get('image_url'):
                            img_url = str(payload['image_url']).strip()
                            if img_url.startswith(('http://', 'https://', '/api/images/')):
                                has_image = True
                                image_url = img_url if img_url.startswith(('http://', 'https://')) else f"{img_url}?session_id={session_id}"
                        
                        if not has_image:
                            logger.debug(f"Skipping result - no valid image: {product_key}")
                            continue
                        
                        # Add the image URL to the result for easy access
                        if hasattr(result, 'payload'):
                            result.payload['_image_url'] = image_url
                        
                        valid_results.append(result)
                        
                        if len(valid_results) >= 6:  # Limit to top 6 matches
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error processing result: {str(e)}", exc_info=True)
                        continue
                
                results = valid_results
                logger.info(f"Kept {len(results)} valid results after filtering")
                
                if not results:
                    return jsonify({
                        "message": "No valid products found with matching images. Please try a different search.",
                        "results": []
                    })
            else:
                # No category filtering, just take top results
                results = results[:6] if results else []
            
            if not results:
                return jsonify({
                    "message": "No visually similar products found. Try uploading a clearer image or search by text.",
                    "results": []
                })
                
            # Format the results for the response
            formatted_results = format_product_results(results, session_id)
            
            # Create response message based on detected categories
            if detected_categories:
                category_text = ', '.join(detected_categories)
                message = f"Here are some visually similar {category_text}s:"
            else:
                message = "Here are some visually similar products:"
                
            logger.info(f"Returning {len(formatted_results)} formatted results")
            
            return jsonify({
                "message": message,
                "results": formatted_results,
                "detected_categories": detected_categories  # Include this for debugging
            })
            
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}", exc_info=True)
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in query_image: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# ---------------------------
# Image serving endpoint - FIXED VERSION
# ---------------------------
@app.route('/api/images/<product_id>')
@login_required
def get_image(product_id):
    """Serve image data directly from Qdrant for a given product ID"""
    try:
        logger.info(f"Fetching image for product ID: {product_id}")
        
        # Get the session ID from the query parameters
        session_id = request.args.get('session_id')
        if not session_id:
            logger.warning("No session_id provided in request")
            return jsonify({"error": "Session ID is required"}), 400
            
        # Get the collection name for this session
        logger.debug(f"Looking up session: {session_id}")
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            logger.warning(f"Invalid or missing session data for session_id: {session_id}")
            return jsonify({"error": "Invalid or expired session"}), 404
            
        collection_name = session_data['collection_name']
        logger.debug(f"Using collection: {collection_name}")
        
        # First try to get the point directly by ID (as string)
        try:
            logger.debug(f"Attempting direct lookup of product ID: {product_id}")
            point = vector_store.client.retrieve(
                collection_name=collection_name,
                ids=[str(product_id)],  # Ensure ID is string
                with_payload=True,
                with_vectors=False
            )
            
            if point and len(point) > 0:
                payload = point[0].payload or {}
                if not payload and hasattr(point[0], 'get'):
                    payload = {k: v for k, v in point[0].items() if k != 'payload'}
                if payload:
                    logger.debug(f"Found product with direct lookup: {product_id}")
                    return _serve_image(payload, product_id)
                else:
                    logger.warning(f"Empty payload for product ID: {product_id}")
            else:
                logger.debug(f"No direct match found for product ID: {product_id}")
                
        except Exception as e:
            logger.error(f"Error in direct product lookup: {str(e)}", exc_info=True)
        
        # Try searching in the payload with different field names
        search_fields = ["product_id", "id", "_id", "name"]
        
        for field in search_fields:
            try:
                logger.debug(f"Searching by field '{field}' for value: {product_id}")
                results = vector_store.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=field,
                                match=models.MatchValue(value=product_id)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                
                if results and len(results[0]) > 0:
                    payload = results[0][0].payload or {}
                    if payload:
                        logger.debug(f"Found product by {field} lookup: {product_id}")
                        return _serve_image(payload, product_id)
                    else:
                        logger.warning(f"Empty payload in search by {field} for ID: {product_id}")
                else:
                    logger.debug(f"No match found for {field}={product_id}")
                    
            except Exception as e:
                logger.error(f"Search by {field} failed: {str(e)}", exc_info=True)
                continue
        
        # If we get here, no product was found with any search method
        logger.warning(f"Product {product_id} not found in collection {collection_name}")
        return jsonify({
            "error": "Product not found",
            "details": f"No product found with ID: {product_id} in collection: {collection_name}"
        }), 404
            
    except Exception as e:
        logger.error(f"Error in get_image: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500


def _serve_image(payload, product_id):
    """Helper method to serve image from payload"""
    if not payload:
        logger.warning(f"No payload found for product {product_id}")
        return jsonify({"error": "Product data not found"}), 404
        
    # Get the base64-encoded image data
    image_data = payload.get('image_data', '')
    if not image_data:
        logger.warning(f"No image data found for product {product_id}")
        return jsonify({"error": "No image data available"}), 404
        
    # Clean up the base64 data if it contains data URL prefix
    if image_data.startswith('data:image/'):
        # Extract just the base64 part
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
    
    # Decode the base64 data
    try:
        image_bytes = base64.b64decode(image_data)
        if not image_bytes:
            raise ValueError("Decoded image data is empty")
    except Exception as e:
        logger.error(f"Error decoding base64 image data for product {product_id}: {str(e)}")
        return jsonify({"error": "Invalid image data"}), 400
    
    # Determine content type based on image signature
    content_type = 'image/jpeg'  # Default
    if image_bytes.startswith(b'\xFF\xD8\xFF'):
        content_type = 'image/jpeg'
    elif image_bytes.startswith(b'\x89PNG\r\n\x1A\n'):
        content_type = 'image/png'
    elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
        content_type = 'image/gif'
    elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
        content_type = 'image/webp'
    
    # Create response with proper headers
    response = make_response(image_bytes)
    response.headers.set('Content-Type', content_type)
    response.headers.set('Content-Length', len(image_bytes))
    response.headers.set('Cache-Control', 'public, max-age=31536000')  # Cache for 1 year
    response.headers.set('Accept-Ranges', 'bytes')
    
    logger.info(f"Successfully served image for product {product_id}, size: {len(image_bytes)} bytes, type: {content_type}")
    return response

# ---------------------------
# Global error handlers
# ---------------------------
@app.errorhandler(404)
def not_found(error):
    # Return JSON for API endpoints, HTML for others
    if request.path.startswith(('/query', '/upload_products', '/query_image', '/ask_about_product', '/api/')):
        return jsonify({"error": "Endpoint not found", "status": 404}), 404
    return error, 404

@app.errorhandler(500)
def internal_error(error):
    # Return JSON for API endpoints, HTML for others
    if request.path.startswith(('/query', '/upload_products', '/query_image', '/ask_about_product', '/api/')):
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return error, 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Handle any unhandled exceptions
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    if request.path.startswith(('/query', '/upload_products', '/query_image', '/ask_about_product', '/api/')):
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
    return e, 500

# ---------------------------
# Graceful shutdown handler
# ---------------------------
def handle_shutdown(signum, frame):
    logger.info("Shutting down server...")
    # Add any cleanup code here if needed
    import sys
    sys.exit(0)

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import signal
    from werkzeug.serving import is_running_from_reloader
    
    # Only run this once, not in the reloader process
    if not is_running_from_reloader():
        # Ensure upload directory exists
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        # Print startup message with the URL
        print("\n" + "="*50)
        print(f"Starting server at http://localhost:5000")
        print("Images will be served directly from Qdrant via /api/images/<product_id>")
        print("="*50 + "\n")
    
    # Run the app with threaded=True to handle multiple requests
    app.run(host='0.0.0.0', 
            port=5000, 
            debug=True, 
            use_reloader=True, 
            threaded=True,
            use_debugger=True,
            passthrough_errors=True)