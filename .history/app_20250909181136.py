from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for 
from flask_cors import CORS
import os, uuid, json, numpy as np
from dotenv import load_dotenv
from embeddings import embed_text, embed_image_bytes
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
            if request.path.startswith('/query') or request.path.startswith('/upload_products') or request.path.startswith('/query_image') or request.path.startswith('/ask_about_product'):
                return jsonify({"error": "Authentication required", "authenticated": False}), 401
            else:
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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
        if not data or 'email' not in data or 'name' not in data:
            return jsonify({'error': 'Email and name are required'}), 400
        
        # Create user session
        user = {
            'email': data['email'],
            'name': data['name']
        }
        
        # Store user in session
        session['user'] = user
        session.permanent = True  # Make the session persistent
        
        return jsonify({
            'message': 'Login successful',
            'user': user,
            'redirect': '/'  # Add redirect URL
        })
        
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
    if 'user' not in session and path != 'login.html':
        return redirect(url_for('login'))
    return send_from_directory('frontend', path)

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
            vector_store.create_collection(collection_name, vector_size=512, recreate=True)
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
                
                # Check for image file in the uploaded files
                image_filename = product.get('image_filename') or f"product_{product_id}_{idx}.jpg"
                image_url = product.get('image_url', '')
                
                # If we have an uploaded file for this product, use it
                if 'file_mapping' in locals() and image_filename in file_mapping:
                    image_url = f"/uploads/{file_mapping[image_filename]}"
                # If we have an image URL, download and save it
                elif image_url:
                    saved_image_url = save_image_from_url(image_url, session_id)
                    if saved_image_url:
                        image_url = saved_image_url
                    else:
                        logger.warning(f"Failed to download image from {image_url}")
                
                # Generate text embedding from product name, category, and description
                category = str(product.get('category', '')).lower()
                text_to_embed = f"{name} {category} {description}".strip()
                
                # Add product type to the text if it can be inferred
                product_types = ["ring", "necklace", "bracelet", "earring", "watch"]
                for ptype in product_types:
                    if ptype in name.lower() or ptype in description.lower() or ptype in category:
                        text_to_embed = f"{ptype} {text_to_embed}"  # Emphasize product type
                        break
                
                if not text_to_embed:
                    logger.warning(f"Skipping product with empty data: {product_id}")
                    continue
                
                # Generate text embedding
                text_embedding = embed_text(text_to_embed)
                if not text_embedding:
                    logger.error(f"Failed to generate text embedding for product: {product_id}")
                    continue
                
                # Generate image embedding if image URL is available
                image_embedding = None
                if image_url and image_url.startswith('http'):
                    try:
                        # Download the image
                        response = requests.get(image_url, stream=True, timeout=10)
                        response.raise_for_status()
                        
                        # Generate embedding from image bytes
                        image_embedding = embed_image_bytes(response.content)
                        if not image_embedding:
                            logger.warning(f"Failed to generate image embedding for product: {product_id}")
                    except Exception as e:
                        logger.warning(f"Error processing image for product {product_id}: {str(e)}")
                
                if not image_embedding and os.path.exists(image_url.lstrip('/')):
                    try:
                        # Try to read local file
                        with open(image_url.lstrip('/'), 'rb') as f:
                            image_embedding = embed_image_bytes(f.read())
                    except Exception as e:
                        logger.warning(f"Error reading local image for product {product_id}: {str(e)}")
                
                if not image_embedding:
                    # If we couldn't get image embedding, use text embedding as fallback
                    image_embedding = text_embedding
                
                # Prepare product payload with image_url
                payload = {
                    "product_id": product_id,
                    "name": name,
                    "description": description,
                    "price": str(product.get('price', '')),
                    "category": category,
                    "image_url": image_url,  # This will be either the original URL or the uploaded file URL
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Create point with both text and image vectors
                point = {
                    "id": point_id,  # Use the generated UUID as the point ID
                    "vector": {
                        "text": text_embedding,
                        "image": image_embedding
                    },
                    "payload": payload
                }
                points.append(point)
                
            except Exception as e:
                logger.error(f"Error processing product {product.get('product_id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        if not points:
            return jsonify({"error": "No valid products to upload"}), 400
            
        # Insert points into Qdrant
        try:
            success = vector_store.upsert_points(
                collection_name=collection_name,
                points=points
            )
            
            if not success:
                raise Exception("Failed to insert points into vector store")
                
            # Update session with product count
            sessions_col.update_one(
                {"_id": session_id},
                {"$inc": {"product_count": len(points)}},
                upsert=True
            )
            
            return jsonify({
                "message": f"Successfully uploaded {len(points)} products",
                "count": len(points)
            })
            
        except Exception as e:
            logger.error(f"Error storing products in vector store: {str(e)}", exc_info=True)
            return jsonify({"error": f"Failed to store products in vector database: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in upload_products: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500

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
    """Handle product search by text query with exact matching and category filtering"""
    try:
        # Verify session exists
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "Session not found"}), 404
        
        # Define jewelry categories and their variations
        jewelry_categories = {
            'necklace': ['necklace', 'necklaces'],
            'ring': ['ring', 'rings'],
            'bracelet': ['bracelet', 'bracelets'],
            'earring': ['earring', 'earrings'],
            'anklet': ['anklet', 'anklets'],
            'pendant': ['pendant', 'pendants'],
            'bangle': ['bangle', 'bangles'],
            'brooch': ['brooch', 'brooches'],
            'choker': ['choker', 'chokers']
        }
        
        # Check if query contains any jewelry category
        query_lower = query.lower()
        matched_categories = []
        
        # Find all matching categories in the query
        for category, terms in jewelry_categories.items():
            if any(re.search(r'\b' + re.escape(term) + r'\b', query_lower) for term in terms):
                matched_categories.append(category)
        
        # If we're looking for a specific category, search within that category
        if matched_categories:
            # Create a query that emphasizes the category
            boosted_query = " ".join([f"{c}" for c in matched_categories])
            query_embedding = embed_text(boosted_query)
            
            # First try with strict category matching
            results = vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                filter_categories=matched_categories,
                top_k=10,
                score_threshold=0.7  # Higher threshold for strict matching
            )
            
            # If no results with strict matching, try with a lower threshold
            if not results:
                results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    filter_categories=matched_categories,
                    top_k=10,
                    score_threshold=0.4
                )
            
            # Filter results to ensure they match at least one of the categories
            if results:
                filtered_results = []
                for result in results:
                    payload = getattr(result, 'payload', {}) or {}
                    result_categories = str(payload.get('category', '')).lower().split(',')
                    # Check if any of the matched categories are in the result categories
                    if any(cat.strip() in ' '.join(result_categories) for cat in matched_categories):
                        filtered_results.append(result)
                
                if filtered_results:
                    return format_product_results(filtered_results[:10])  # Return top 10 filtered results
        
        # If no category match or no results from category search, try exact match
        exact_matches = vector_store.search(
            collection_name=collection_name,
            query_text=query,
            exact_match=True,
            top_k=1
        )
        
        # If we found an exact match, return only that product
        if exact_matches and hasattr(exact_matches[0], 'score') and exact_matches[0].score > 0.9:
            return format_product_results([exact_matches[0]])
            
        # If no exact match, proceed with regular semantic search
        if 'query_embedding' not in locals():
            query_embedding = embed_text(query)
            if not query_embedding:
                return jsonify({"error": "Failed to process search query"}), 500
            
            results = vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=5,
                score_threshold=0.5
            )
        
        # If we have results, return them
        if results:
            return format_product_results(results)
            
        return jsonify({"message": "No matching products found"}), 404
        
    except Exception as e:
        logger.error(f"Error in handle_product_search: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while searching for products"}), 500

def handle_product_question(question: str, session_id: str, collection_name: str):
    """Handle questions about products"""
    try:
        # Verify session exists
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "Session not found"}), 404
        
        # Generate embedding for the question
        question_embedding = embed_text(question)
        if not question_embedding:
            return jsonify({"error": "Failed to process your question"}), 500
            
        # Search for relevant products
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            top_k=3,
            score_threshold=0.4
        )
        
        if not results:
            return jsonify({
                "message": "I couldn't find any relevant products to answer your question.",
                "results": []
            })
            
        # Format the results
        formatted_results = format_product_results(results)
        
        # For now, just return the top result as the answer
        if formatted_results and len(formatted_results) > 0:
            top_result = formatted_results[0]
            answer = f"I found a product that might help: {top_result['name']}"
            if top_result.get('description'):
                answer += f" - {top_result['description'][:100]}..."
            return jsonify({
                "message": answer,
                "results": formatted_results
            })
            
        return jsonify({"results": formatted_results})
        
    except Exception as e:
        logger.error(f"Error in handle_product_question: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your question"}), 500

def format_product_results(results):
    """Format product results for the frontend"""
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
            
            # Create unique identifier for deduplication
            product_key = f"{payload.get('name', '')}_{payload.get('description', '')}"
            if not product_key.strip('_') or product_key in seen_products:
                continue
                
            seen_products.add(product_key)
            
            # Convert score to percentage format
            percentage_score = min(100.0, max(0.0, score * 100))
            
            # Extract product details with proper defaults
            product = {
                'id': str(payload.get('product_id', result_id)),
                'name': str(payload.get('name', 'Unnamed Product')).strip(),
                'description': str(payload.get('description', '')).strip(),
                'price': str(payload.get('price', 'N/A')).strip(),
                'category': str(payload.get('category', 'Uncategorized')).strip(),
                'image_url': str(payload.get('image_url', '')).strip(),
                'score': round(percentage_score, 2)  # Round to 2 decimal places
            }
            
            # Ensure image path is in the correct format
            if (product['image_url'] and 
                not product['image_url'].startswith(('http://', 'https://', '/'))):
                product['image_url'] = f"/{product['image_url'].lstrip('/')}"
                
            formatted_results.append(product)
            
        except Exception as e:
            logger.error(f"Error formatting product result: {str(e)}\nResult: {result}", exc_info=True)
            continue
    
    # Sort by score in descending order
    formatted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return formatted_results

# ---------------------------
# Image query
# ---------------------------
@app.route("/query_image/<session_id>", methods=["POST"])
@login_required
def query_image(session_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Create a unique filename to prevent collisions
    temp_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, temp_filename)
    
    try:
        # Save the file temporarily
        try:
            file.save(filepath)
            logger.info(f"Saved uploaded file to {filepath}")
            if not os.path.exists(filepath):
                raise Exception("File was not saved correctly")
        except Exception as e:
            logger.error(f"Error saving temporary file: {str(e)}")
            return jsonify({"error": "Failed to save uploaded file"}), 500
        
        try:
            # Get the collection name for this session
            session_data = sessions_col.find_one({"_id": session_id})
            if not session_data or 'collection_name' not in session_data:
                return jsonify({"error": "No products found for this session"}), 404
                
            collection_name = session_data['collection_name']
            
            # Load CLIP model for visual similarity search
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
                from PIL import Image
                
                # Initialize CLIP model and processor (this will be cached after first load)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if not hasattr(query_image, 'model'):
                    query_image.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                    query_image.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    query_image.device = device
                
                # Process the uploaded image
                image = Image.open(filepath).convert("RGB")
                
                # STEP 1: Detect product category from the image using CLIP text-image similarity
                jewelry_categories = [
                    "necklace", "ring", "bracelet", "earring", "anklet", 
                    "pendant", "bangle", "brooch", "choker", "watch"
                ]
                
                # Create text prompts for category detection
                category_texts = [f"a {category} jewelry" for category in jewelry_categories]
                
                # Process both image and text through CLIP
                inputs = query_image.processor(
                    text=category_texts, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(query_image.device) for k, v in inputs.items()}
                
                # Get similarity scores between image and category texts
                with torch.no_grad():
                    outputs = query_image.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Find the most likely category
                best_category_idx = probs.argmax().item()
                best_category = jewelry_categories[best_category_idx]
                confidence = probs[0][best_category_idx].item()
                
                logger.info(f"Detected category: {best_category} with confidence: {confidence:.3f}")
                
                # STEP 2: Get image embedding for visual similarity
                image_inputs = query_image.processor(images=image, return_tensors="pt", padding=True)
                image_inputs = {k: v.to(query_image.device) for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    image_features = query_image.model.get_image_features(**image_inputs)
                
                # Convert to numpy array and ensure correct shape
                query_embedding = image_features.cpu().numpy().astype('float32')
                
                # Ensure we have a 1D array of the correct dimension
                if len(query_embedding.shape) > 1:
                    query_embedding = query_embedding.reshape(-1)
                
                # Check dimension
                if len(query_embedding) != 512:  # CLIP embedding dimension
                    logger.warning(f"Unexpected query embedding dimension: {len(query_embedding)}")
                    # Truncate or pad if necessary
                    if len(query_embedding) > 512:
                        query_embedding = query_embedding[:512]
                    else:
                        padding = np.zeros(512 - len(query_embedding), dtype='float32')
                        query_embedding = np.concaten([query_embedding, padding])
                
                # Normalize the embedding
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = (query_embedding / norm).tolist()
                else:
                    query_embedding = query_embedding.tolist()
                
                # STEP 3: Search for visually similar products (using existing VectorStore search method)
                logger.info(f"Searching for similar images with detected category: {best_category}")
                
                # Use the existing search method without filter_categories parameter
                search_results = vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    vector_name="image",
                    top_k=20,  # Get more results for better filtering
                    score_threshold=0.2  # Lower threshold for initial search
                )
                
                # Log the search results for debugging
                if search_results:
                    logger.info(f"Found {len(search_results)} potential matches")
                else:
                    logger.warning("No visually similar products found")
                
                if not search_results:
                    return jsonify({
                        "message": "No visually similar products found. Try adjusting your search criteria.", 
                        "results": []
                    })
                
                # STEP 4: Post-process results with STRICT category filtering
                formatted_results = []
                category_matches = []
                seen_products = set()  # Track seen products for deduplication

                # Define related categories for more flexible matching
                related_categories = {
                    'necklace': ['necklace', 'pendant', 'chain'],
                    'pendant': ['pendant', 'necklace', 'chain'], 
                    'ring': ['ring', 'band'],
                    'bracelet': ['bracelet', 'bangle'],
                    'earring': ['earring', 'stud'],
                    'anklet': ['anklet'],
                    'watch': ['watch'],
                    'brooch': ['brooch'],
                    'choker': ['choker', 'necklace']
                }

                # Get related categories for the detected category
                target_categories = related_categories.get(best_category.lower(), [best_category.lower()])
                logger.info(f"Looking for products in categories: {target_categories}")

                for result in search_results:
                    try:
                        if hasattr(result, 'score'):
                            # Qdrant ScoredPoint object
                            payload = getattr(result, 'payload', {}) or {}
                            base_score = float(getattr(result, 'score', 0))
                            result_id = str(getattr(result, 'id', ''))
                        else:
                            # Dictionary format
                            payload = result.get('payload', {})
                            base_score = float(result.get('score', 0))
                            result_id = str(result.get('id', ''))
                        
                        if not payload:
                            continue
                        
                        # Create unique identifier for deduplication
                        product_key = f"{payload.get('name', '')}_{payload.get('product_id', '')}"
                        if product_key in seen_products:
                            continue
                        seen_products.add(product_key)
                        
                        # Get product details for category matching
                        product_category = str(payload.get('category', '')).lower().strip()
                        product_name = str(payload.get('name', '')).lower().strip()
                        product_desc = str(payload.get('description', '')).lower().strip()
                        
                        # STRICT category matching - only include if it matches target categories
                        is_category_match = False
                        
                        # Check if any target category appears in the product details
                        for target_cat in target_categories:
                            if (target_cat in product_category or 
                                target_cat in product_name or 
                                target_cat in product_desc or
                                # Check for plural forms
                                target_cat + 's' in product_category or
                                target_cat + 's' in product_name):
                                is_category_match = True
                                break
                        
                        # If confidence is high, ONLY include category matches
                        if confidence > 0.6 and not is_category_match:
                            continue
                        
                        # If confidence is medium, be more selective
                        elif confidence > 0.4 and not is_category_match and base_score < 0.7:
                            continue
                            
                        # If confidence is low, include high-scoring visual matches
                        elif confidence <= 0.4 and not is_category_match and base_score < 0.8:
                            continue
                        
                        # Calculate final score with category boost
                        final_score = base_score
                        if is_category_match:
                            final_score *= (1.5 + confidence)  # Higher boost for higher confidence
                        
                        # Convert score to percentage format
                        percentage_score = min(100.0, max(0.0, final_score * 100))
                        
                        formatted_result = {
                            'id': str(payload.get('product_id', result_id)),
                            'name': str(payload.get('name', 'Unnamed Product')),
                            'description': str(payload.get('description', '')),
                            'price': str(payload.get('price', 'N/A')),
                            'category': str(payload.get('category', 'Uncategorized')),
                            'image_url': str(payload.get('image_url', '')),
                            'score': round(percentage_score, 2),
                            'is_category_match': is_category_match
                        }
                        
                        category_matches.append(formatted_result)
                        
                        # Stop once we have enough good matches
                        if len(category_matches) >= 8:
                            break
                            
                    except Exception as e:
                        logger.error(f"Error formatting result: {str(e)}", exc_info=True)
                        continue

                # Sort by score
                category_matches.sort(key=lambda x: x.get('score', 0), reverse=True)

                # Take top 5 results
                formatted_results = category_matches[:5]

                if not formatted_results:
                    logger.warning("No valid products found after strict filtering")
                    return jsonify({
                        "message": f"No {best_category} products found in your catalog. Try with a different image or check if you have {best_category} products uploaded.",
                        "detected_category": best_category,
                        "confidence": round(confidence * 100, 1),
                        "results": []
                    })

                # Create response message
                category_count = len([r for r in formatted_results if r.get('is_category_match')])
                total_count = len(formatted_results)

                if confidence > 0.6:
                    if category_count == total_count:
                        message = f"Found {total_count} {best_category}(s) matching your image"
                    else:
                        message = f"Found {category_count} {best_category}(s) and {total_count - category_count} similar items"
                elif confidence > 0.4:
                    message = f"Found {total_count} products similar to your {best_category} image"
                else:
                    message = f"Found {total_count} visually similar products"

                # Remove the is_category_match flag from final results (internal use only)
                for result in formatted_results:
                    result.pop('is_category_match', None)

                logger.info(f"Returning {len(formatted_results)} strictly filtered results with {category_count} category matches")
                return jsonify({
                    "message": message,
                    "detected_category": best_category,
                    "confidence": round(confidence * 100, 1),
                    "results": formatted_results
                })
                    
            except Exception as e:
                logger.error(f"Error processing image with CLIP: {str(e)}", exc_info=True)
                return jsonify({
                    "error": "Failed to process image",
                    "details": str(e),
                    "results": []
                }), 500
            
        except ValueError as ve:
            logger.error(f"Validation error in query_image: {str(ve)}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logger.error(f"Error in query_image: {str(e)}", exc_info=True)
            return jsonify({"error": "An error occurred while processing your request"}), 500
            
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Error removing temporary file {filepath}: {str(e)}")

# ---------------------------
# Product Question Answering
# ---------------------------
@app.route("/ask_about_product", methods=["POST"])
@login_required
def ask_about_product():
    try:
        data = request.get_json()
        question = data.get("question")
        product_id = data.get("product_id")
        session_id = data.get("session_id")
        
        if not all([question, product_id, session_id]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get session and collection
        session_data = sessions_col.find_one({"_id": session_id})
        if not session_data or 'collection_name' not in session_data:
            return jsonify({"error": "No products found for this session"}), 404
            
        collection_name = session_data['collection_name']
        
        # Generate embedding for the question
        question_embedding = generate_embedding(question)
        
        # Search for relevant products
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            top_k=3,
            score_threshold=0.3
        )
        
        if not results:
            return jsonify({"error": "No relevant products found to answer your question"}), 404
            
        # Format the results for the response
        formatted_results = format_product_results(results)
        
        # For now, just return the top relevant products
        # In a real app, you might want to use a language model to generate a more detailed answer
        return jsonify({
            "message": "Here are some products that might help answer your question:",
            "results": formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error in ask_about_product: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# ---------------------------
# Global error handlers
# ---------------------------
@app.errorhandler(404)
def not_found(error):
    # Return JSON for API endpoints, HTML for others
    if request.path.startswith('/query') or request.path.startswith('/upload_products') or request.path.startswith('/query_image') or request.path.startswith('/ask_about_product'):
        return jsonify({"error": "Endpoint not found", "status": 404}), 404
    return error, 404

@app.errorhandler(500)
def internal_error(error):
    # Return JSON for API endpoints, HTML for others
    if request.path.startswith('/query') or request.path.startswith('/upload_products') or request.path.startswith('/query_image') or request.path.startswith('/ask_about_product'):
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return error, 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Handle any unhandled exceptions
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    if request.path.startswith('/query') or request.path.startswith('/upload_products') or request.path.startswith('/query_image') or request.path.startswith('/ask_about_product'):
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
    return e, 500

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    # Print startup message with the URL
    print("\n" + "="*50)
    print(f"Starting server at http://localhost:5000")
    print("="*50 + "\n")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)