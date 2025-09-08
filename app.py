from flask import Flask, request, jsonify, send_from_directory
import os, uuid, json
from dotenv import load_dotenv
from embeddings import embed_text, embed_image_bytes
from vectorstore import VectorStore
from pymongo import MongoClient
from datetime import datetime, timezone
import logging
import requests
from urllib.parse import urlparse
import shutil
from werkzeug.utils import secure_filename
import tempfile

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load environment variables
load_dotenv()

# ---------------------------
# Config
# ---------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

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
app = Flask(__name__, static_folder="frontend", static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

# ---------------------------
# Serve frontend & images
# ---------------------------
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/uploads/<filename>")
def serve_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------------------------
# Create session
# ---------------------------
@app.route("/create_session", methods=["POST"])
def create_session():
    try:
        # Test MongoDB connection
        mongo_client.server_info()
        
        session_id = str(uuid.uuid4())
        collection_name = f"session_{session_id}"  # Changed to match upload_products

        try:
            # Create Qdrant collection
            vector_store.create_collection_if_not_exists(collection_name)
        except Exception as qe:
            return jsonify({
                "error": f"Failed to connect to Qdrant vector database. Please ensure Qdrant is running. Error: {str(qe)}"
            }), 500

        # Save session in MongoDB
        try:
            sessions_col.insert_one({
                "_id": session_id,
                "collection": collection_name,  # Store the exact collection name
                "products": [],
                "created_at": datetime.now(timezone.utc),  # Updated to use timezone-aware datetime
                "status": "active"
            })
            return jsonify({
                "session_id": session_id,
                "collection": collection_name,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return jsonify({"error": f"Failed to create session: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in create_session: {str(e)}")
        return jsonify({"error": f"Failed to create session: {str(e)}"}), 500

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
def upload_products(session_id):
    try:
        logger.info(f"Received upload_products request for session: {session_id}")
        
        # Verify session exists
        session = sessions_col.find_one({"_id": session_id})
        if not session:
            logger.error(f"Session not found: {session_id}")
            return jsonify({"error": "Invalid session"}), 400
            
        # Get collection name from session or create one
        collection_name = session.get("collection")
        if not collection_name:
            collection_name = f"session_{session_id}"
            sessions_col.update_one(
                {"_id": session_id},
                {"$set": {"collection": collection_name}}
            )
        
        # Get products JSON from form data
        products_json = request.form.get("products_json")
        if not products_json:
            logger.error("No products_json in form data")
            return jsonify({"error": "Missing products JSON"}), 400
            
        try:
            products = json.loads(products_json)
            logger.info(f"Successfully parsed {len(products)} products from JSON")
            
            # Validate products is a list
            if not isinstance(products, list):
                return jsonify({"error": "Products data should be an array of product objects"}), 400
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
            
        logger.info(f"Processing {len(products)} products")
        
        # Get uploaded files if any
        uploaded_files = request.files.getlist('images') if 'images' in request.files else []
        logger.info(f"Found {len(uploaded_files)} uploaded files")
        
        saved_images = []
        points = []
        processed_count = 0
        
        for i, product in enumerate(products):
            try:
                logger.info(f"Processing product {i+1}/{len(products)}: {product.get('name', 'Unnamed')}")
                
                # Skip if no product_id
                if 'product_id' not in product:
                    logger.error(f"Product {i} is missing required field 'product_id'")
                    continue
                
                image_path = ""
                
                # First try to use uploaded file if available
                if i < len(uploaded_files) and uploaded_files[i].filename:
                    file = uploaded_files[i]
                    logger.info(f"Processing uploaded file {i}: {file.filename}")
                    if file and allowed_file(file.filename):
                        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        file.save(filepath)
                        image_path = f"uploads/{filename}"
                        saved_images.append(image_path)
                        logger.info(f"Saved uploaded file to {filepath}")
                
                # If no uploaded file, try to use image_url from product
                if not image_path and 'image_url' in product and product['image_url']:
                    image_url = product['image_url']
                    try:
                        logger.info(f"Downloading image from URL: {image_url}")
                        filename = f"{uuid.uuid4()}.jpg"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        
                        # Set a user agent to avoid 403 errors
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        response = requests.get(image_url, headers=headers, stream=True, timeout=10)
                        response.raise_for_status()
                        
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:  # filter out keep-alive new chunks
                                    f.write(chunk)
                        
                        image_path = f"uploads/{filename}"
                        saved_images.append(image_path)
                        logger.info(f"Downloaded image from {image_url} to {filepath}")
                        
                    except Exception as e:
                        logger.error(f"Error downloading image from {image_url}: {str(e)}")
                        # Continue with the next product if image download fails
                        continue
                
                # If we have an image, process it
                if image_path:
                    # Generate embedding for the image
                    try:
                        # Read the image file as bytes
                        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))
                        with open(full_image_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # Generate embedding from image bytes
                        image_embedding = embed_image_bytes(image_bytes)
                        if image_embedding is None:
                            raise Exception("Failed to generate image embedding")
                            
                        # Create point for Qdrant
                        point = {
                            "id": str(uuid.uuid4()),
                            "vector": image_embedding,
                            "payload": {
                                "product_id": product.get('product_id'),
                                "name": product.get('name', ''),
                                "category": product.get('category', ''),
                                "price": product.get('price', ''),
                                "description": product.get('description', ''),
                                "image_path": image_path,
                                "session_id": session_id,
                                "created_at": datetime.utcnow().isoformat()
                            }
                        }
                        points.append(point)
                        processed_count += 1
                        logger.info(f"Successfully processed product {product.get('product_id')}")
                        
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True)
                        continue
                else:
                    logger.warning(f"No image available for product {product.get('product_id')}")
                    
            except Exception as e:
                logger.error(f"Error processing product {i}: {str(e)}", exc_info=True)
                continue
        
        # Save points to Qdrant if we have any
        if points:
            try:
                logger.info(f"Upserting {len(points)} points to Qdrant collection {collection_name}")
                vector_store.upsert_points(collection_name, points)
                logger.info(f"Successfully upserted {len(points)} points to Qdrant")
            except Exception as e:
                logger.error(f"Error upserting to Qdrant: {str(e)}")
                return jsonify({"error": f"Failed to save to vector database: {str(e)}"}), 500
        
        return jsonify({
            "status": "success",
            "processed_products": processed_count,
            "saved_images": len(saved_images)
        })
        
    except Exception as e:
        logger.error(f"Error in upload_products: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ---------------------------
# Text query
# ---------------------------
@app.route("/query_text", methods=["POST"])
def query_text():
    try:
        data = request.get_json()
        query = data.get("query")
        session_id = data.get("session_id")
        
        if not query or not session_id:
            return jsonify({"error": "Missing query or session_id"}), 400
        
        # Get session
        session = sessions_col.find_one({"_id": session_id})
        if not session:
            return jsonify({"error": "Invalid session"}), 400
            
        collection_name = session.get("collection")
        if not collection_name:
            return jsonify({"error": "No collection found for this session"}), 400
        
        # Check if this is a question or a search query
        is_question = "?" in query or any(word in query.lower() for word in ["what", "how", "when", "where", "why", "tell me", "show me"])
        
        if is_question:
            # Handle question-answering
            return handle_product_question(query, session_id, collection_name)
        else:
            # Handle product search
            return handle_product_search(query, session_id, collection_name)
            
    except Exception as e:
        logger.error(f"Error in query_text: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def handle_product_search(query, session_id, collection_name):
    """Handle product search by text query"""
    # Generate embedding for the query
    vec = embed_text(query)
    if vec is None:
        return jsonify({"error": "Failed to generate embedding for query"}), 500
    
    # Query similar products
    results = vector_store.query_similar(collection_name, vec, top_k=6)
    
    # Format response
    products = format_product_results(results)
    
    return jsonify({
        "search_query": query,
        "answer": f"Found {len(products)} products matching your search:" if products else "No matching products found.",
        "products": products
    })

def handle_product_question(question, session_id, collection_name):
    """Handle questions about products"""
    # First, find relevant products using the question as a search query
    vec = embed_text(question)
    if vec is None:
        return jsonify({"error": "Failed to process your question"}), 500
    
    # Get relevant products
    results = vector_store.query_similar(collection_name, vec, top_k=3)
    
    if not results:
        return jsonify({
            "answer": "I couldn't find any relevant products to answer your question.",
            "products": []
        })
    
    # Format the products
    products = format_product_results(results)
    
    # Generate a natural language response
    product_names = ", ".join([p["name"] for p in products if p.get("name")])
    answer = f"Based on your question about '{question}', here are some relevant products: {product_names}."
    
    return jsonify({
        "answer": answer,
        "products": products
    })

def format_product_results(results):
    """Format product results for the frontend"""
    products = []
    for result in results:
        payload = result.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"value": payload}
            
        image_path = payload.get("image_path", "")
        if not image_path:
            continue
            
        # Ensure the image path is a valid URL
        if not image_path.startswith(('http://', 'https://', '/')):
            if not image_path.startswith('uploads/'):
                image_path = f"uploads/{image_path}"
            image_path = f"{request.host_url.rstrip('/')}/{image_path}"
        
        products.append({
            "id": payload.get("product_id") or str(uuid.uuid4()),
            "name": payload.get("name", ""),
            "category": payload.get("category", ""),
            "price": payload.get("price", ""),
            "description": payload.get("description", ""),
            "url": image_path,  # Changed from "image_url" to "url" to match frontend
            "score": float(result.get("score", 0.0))
        })
    
    return products

# ---------------------------
# Image query
# ---------------------------
@app.route("/query_image/<session_id>", methods=["POST"])
def query_image(session_id):
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Please upload an image file (jpg, jpeg, png, gif)."}), 400
        
        # Get session
        session = sessions_col.find_one({"_id": session_id})
        if not session:
            return jsonify({"error": "Invalid session. Please create a new session."}), 400
            
        collection_name = session.get("collection")
        if not collection_name:
            return jsonify({"error": "No product catalog found. Please upload products first."}), 400
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            try:
                file.save(temp_file.name)
                
                # Generate embedding for the image
                with open(temp_file.name, "rb") as f:
                    image_data = f.read()
                    if not image_data:
                        return jsonify({"error": "The uploaded file appears to be empty."}), 400
                        
                    vec = embed_image_bytes(image_data)
                    if vec is None:
                        return jsonify({"error": "Failed to process the image. Please try another image."}), 500
                
                # Query similar products
                results = vector_store.query_similar(collection_name, vec, top_k=6)
                
                # Format response
                products = format_product_results(results)
                
                # Ensure image URLs are absolute
                for product in products:
                    if 'url' in product and not product['url'].startswith(('http://', 'https://')):
                        product['url'] = f"/{product['url'].replace(os.path.sep, '/').lstrip('/')}"
                
                return jsonify({
                    "search_type": "image",
                    "answer": f"Found {len(products)} similar products:" if products else "No similar products found.",
                    "images": products
                })
                
            finally:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"Error deleting temp file {temp_file.name}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in query_image: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing your image: {str(e)}"}), 500

# ---------------------------
# Product Question Answering
# ---------------------------
@app.route("/ask_about_product", methods=["POST"])
def ask_about_product():
    try:
        data = request.get_json()
        question = data.get("question")
        product_id = data.get("product_id")
        session_id = data.get("session_id")
        
        if not all([question, product_id, session_id]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get session and collection
        session = sessions_col.find_one({"_id": session_id})
        if not session:
            return jsonify({"error": "Invalid session"}), 400
            
        collection_name = session.get("collection")
        if not collection_name:
            return jsonify({"error": "No collection found"}), 400
        
        # Get the specific product
        # Note: This assumes the product ID is stored in the payload
        # You might need to adjust this based on how you're storing products
        results = vector_store.query_similar(
            collection_name, 
            [0] * 512,  # Dummy vector since we're filtering by ID
            filter_conditions={"must": [{"key": "payload.product_id", "match": {"value": product_id}}]}
        )
        
        if not results:
            return jsonify({"error": "Product not found"}), 404
            
        product = results[0].get("payload", {})
        
        # Here you would typically use an LLM to generate an answer based on the question and product details
        # For now, we'll return a simple response with the product details
        answer = f"Here's information about {product.get('name', 'the product')}: {product.get('description', 'No description available.')}"
        
        # Format the product for the response
        formatted_product = {
            "id": product.get("product_id"),
            "name": product.get("name", ""),
            "category": product.get("category", ""),
            "price": product.get("price", ""),
            "description": product.get("description", ""),
            "image_url": product.get("image_path", ""),
        }
        
        return jsonify({
            "answer": answer,
            "product": formatted_product
        })
        
    except Exception as e:
        logger.error(f"Error in ask_about_product: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

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
