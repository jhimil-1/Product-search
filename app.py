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
    Detect jewelry category from text with high accuracy.
    Returns a list of matched categories in order of confidence.
    """
    if not text or not isinstance(text, str):
        return []
        
    text_lower = text.lower().strip()
    
    # Define comprehensive jewelry categories with variations and synonyms
    # Each pattern has a weight that affects the final confidence
    category_patterns = {
        'necklace': [
            (r'\b(?:gold|silver|diamond|pearl|choker|pendant|chain)\s+(?:necklace|chain|pendant)\b', 1.0),
            (r'\bnecklaces?\b', 0.9),
            (r'\b(?:chokers?|pendants?|lockets?|collars?)\b', 0.85),
            (r'\b(?:beads?|strand|layered|layering)\s+(?:necklace|chain)\b', 0.8),
            (r'\b(?:y[oó]?u?k?i?|opera|rope|princess|matinee|bib|lariat|tassel|locket)\b', 0.7)
        ],
        'ring': [
            (r'\b(?:engagement|wedding|diamond|gold|silver|platinum|band|stacking|cocktail|eternity|promise|signet|claddagh|solitare)\s+(?:ring|bands?)\b', 1.0),
            (r'\brings?\b', 0.9),
            (r'\b(?:bands?|anniversary|diamond|gemstone|birthstone|cluster|halo|three[\s-]?stone)\b', 0.8)
        ],
        'earring': [
            (r'\b(?:stud|hoop|dangle|drop|huggy|threader|chandelier|ear\s*cuff|ear\s*jacket|ear\s*threader|ear\s*threaders|ear\s*threading|ear\s*threads)\s+(?:earrings?|ear\s*studs?|ear\s*hoops?|ear\s*drops?)\b', 1.0),
            (r'\bearrings?\b', 0.9),
            (r'\b(?:studs?|hoops?|dangles?|drops?|huggies?|threaders?|chandeliers?|ear\s*cuffs?|ear\s*jackets?)\b', 0.85)
        ],
        'bracelet': [
            (r'\b(?:charm|bangle|cuff|tennis|chain|bead|leather|stainless\s*steel|gold|silver|diamond|gemstone)\s+(?:bracelets?|bangles?|cuffs?|bands?)\b', 1.0),
            (r'\bbracelets?\b', 0.9),
            (r'\b(?:bangles?|cuffs?|charms?|wristbands?|wrist\s*chains?|wristlets?|wrist\s*accessories?)\b', 0.8)
        ],
        'anklet': [
            (r'\b(?:anklets?|ankle\s*chains?|ankle\s*bracelets?|foot\s*jewelry|toe\s*rings?)\b', 1.0)
        ],
        'brooch': [
            (r'\b(?:brooches?|pins?|corsage|lapel\s*pins?|hat\s*pins?|tie\s*pins?|stick\s*pins?|enamel\s*pins?|vintage\s*brooches?)\b', 1.0)
        ],
        'watch': [
            (r'\b(?:watches?|timepieces?|chronographs?|wristwatches?|smartwatches?|analog\s*watches?|digital\s*watches?|automatic\s*watches?|mechanical\s*watches?|quartz\s*watches?|diving\s*watches?|dress\s*watches?|sport\s*watches?|luxury\s*watches?|fashion\s*watches?|pocket\s*watches?)\b', 1.0)
        ]
    }
    
    # Score categories based on pattern matches
    category_scores = {category: 0.0 for category in category_patterns}
    
    for category, patterns in category_patterns.items():
        for pattern, weight in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                category_scores[category] = max(category_scores[category], weight)
    
    # Filter out categories with score below threshold and sort by score
    threshold = 0.7
    detected_categories = [
        category for category, score in sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ) if score >= threshold
    ]
    
    # If no strong matches found, try a more lenient approach
    if not detected_categories:
        for category, patterns in category_patterns.items():
            if any(re.search(p[0], text_lower, re.IGNORECASE) for p in patterns):
                detected_categories.append(category)
    
    # Remove duplicates while preserving order
    seen = set()
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
        query = data.get('query')
        session_id = data.get('session_id')
        
        if not query or not session_id:
            return jsonify({"error": "Missing query or session_id"}), 400
            
        # Get session data
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "No products found for this session"}), 404
            
        collection_name = session_data['collection_name']
        
        # Check if it's a question or search query
        is_question = any(word in query.lower() for word in ["what", "where", "when", "how", "why", "?"])
        
        if is_question:
            return handle_product_question(query, session_id, collection_name)
        else:
            return handle_product_search(query, session_id, collection_name)
            
    except Exception as e:
        logger.error(f"Error in query_text: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500

def handle_product_search(query: str, session_id: str, collection_name: str):
    """Enhanced product search with strict category filtering"""
    try:
        # Verify session exists
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "Session not found"}), 404
        
        # Detect categories from the query
        detected_categories = detect_jewelry_category(query)
        
        logger.info(f"Search query: '{query}' - Detected categories: {detected_categories}")
        
        # Create enhanced query for embedding
        if detected_categories:
            # Boost the category in the query
            enhanced_query = f"{' '.join(detected_categories)} {query}"
        else:
            enhanced_query = query
        
        # Generate embedding for the search query
        query_embedding = embed_text(enhanced_query)
        if not query_embedding:
            return jsonify({"error": "Failed to process search query"}), 500
        
        # Always use category filtering if categories were detected
        if detected_categories:
            logger.info(f"Searching with category filter: {detected_categories}")
            results = vector_store.search_with_category_filter(
                collection_name=collection_name,
                query_vector=query_embedding,
                categories=detected_categories,
                top_k=10,
                score_threshold=0.5  # Slightly lower threshold to get more results for filtering
            )
            
            # If no results with category filter, try a broader search but still within the category
            if not results:
                logger.info("No results with category filter, trying broader search within category")
                results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    top_k=20,
                    score_threshold=0.3
                )
                
                # Filter results to only include matching categories
                if results:
                    filtered_results = []
                    for result in results:
                        payload = getattr(result, 'payload', {}) or {}
                        result_category = str(payload.get('category', '')).lower()
                        
                        # Split categories if multiple are present (comma-separated)
                        result_categories = [cat.strip() for cat in result_category.split(',')]
                        
                        # Check if result category EXACTLY matches any detected category
                        is_category_match = False
                        for detected_cat in detected_categories:
                            detected_cat_lower = detected_cat.lower()
                            # Check for exact category match
                            if detected_cat_lower in result_categories:
                                is_category_match = True
                                break
                        
                        if is_category_match:
                            filtered_results.append(result)
                            if len(filtered_results) >= 10:  # Limit to top 10
                                break
                    
                    results = filtered_results
        else:
            # No specific categories detected, use regular search with higher threshold
            logger.info("No specific category detected, using regular search")
            results = vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=10,
                score_threshold=0.6  # Higher threshold for non-category searches
            )
        
        # Format and return results
        if results:
            formatted_results = format_product_results(results, session_id)
            logger.info(f"Returning {len(formatted_results)} results for query: '{query}'")
            return jsonify({"results": formatted_results})
        else:
            logger.info(f"No results found for query: '{query}'")
            return jsonify({"message": "No matching products found", "results": []})
            
    except Exception as e:
        logger.error(f"Error in handle_product_search: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500

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
                    "detected_categories": detected_categories
                }
        
        # If no results with category filter, try a general search
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            top_k=5,
            score_threshold=0.4
        )
        
        if results:
            formatted_results = format_product_results(results, session_id)
            if formatted_results and len(formatted_results) > 0:
                top_result = formatted_results[0]
                answer = top_result.get('name', 'I found some products')
                if top_result.get('description'):
                    answer += f" - {top_result['description'][:100]}..."
                
                return {
                    "message": answer,
                    "results": formatted_results,
                    "detected_categories": detected_categories or []
                }
        
        # Default response if no products found
        return {
            "message": "I couldn't find any products matching your question. " \
                      "Please try asking about specific products or categories.",
            "results": [],
            "detected_categories": detected_categories or []
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
                    results = vector_store.search_with_category_filter(
                        collection_name=collection_name,
                        query_vector=image_embedding,
                        vector_name="image",
                        categories=detected_categories,
                        top_k=15,  # Get more results to ensure we have enough after filtering
                        score_threshold=0.5
                    )
                    
                    if results:
                        logger.info(f"Found {len(results)} results with strict category filter")
                        
                        # Ensure results match the detected categories exactly
                        filtered_results = []
                        for result in results:
                            payload = getattr(result, 'payload', {}) or {}
                            result_category = str(payload.get('category', '')).lower().strip()
                            
                            # Check if result category matches any detected category exactly
                            # Use exact matching only - don't allow partial matches
                            is_category_match = False
                            for detected_cat in detected_categories:
                                detected_cat_lower = detected_cat.lower()
                                for result_cat in [c.strip() for c in result_category.split(',')]:
                                    if detected_cat_lower == result_cat:
                                        is_category_match = True
                                        break
                                if is_category_match:
                                    break
                            
                            if is_category_match:
                                filtered_results.append(result)
                                if len(filtered_results) >= 10:  # Limit to top 10 matches
                                    break
                        
                        results = filtered_results
                        logger.info(f"Kept {len(results)} results after exact category matching")
                    else:
                        logger.info("No results with strict category filter")
                
                except Exception as e:
                    logger.warning(f"Category filter search failed: {str(e)}", exc_info=True)
                    results = None
            
            # Stage 2: Broader image search without strict category filtering
            if not results:
                logger.info("Stage 2: Broader image search without category filter")
                results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=image_embedding,
                    vector_name="image",
                    top_k=15,
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