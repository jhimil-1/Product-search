from flask import Flask, request, jsonify, send_from_directory
import os, uuid, json
from embeddings import embed_text, embed_image_bytes
import vectorstore
from qdrant_client.http.models import PointStruct
from pymongo import MongoClient

# ---------------------------
# Config
# ---------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "ai_ecom_chatbot"

# ---------------------------
# Initialize
# ---------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB client
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
sessions_col = db["sessions"]  # Stores session info + product metadata

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
    session_id = str(uuid.uuid4())
    collection_name = f"vectors_{session_id}"

    # Create Qdrant collection
    vectorstore.create_collection_if_not_exists(collection_name, vector_size=512)

    # Save session in MongoDB
    sessions_col.insert_one({
        "_id": session_id,
        "collection": collection_name,
        "products": []
    })
    return jsonify({"session_id": session_id})

# ---------------------------
# Upload products
# ---------------------------
@app.route("/upload_products/<session_id>", methods=["POST"])
def upload_products(session_id):
    session = sessions_col.find_one({"_id": session_id})
    if not session:
        return jsonify({"error": "Invalid session"}), 400

    products_json = request.form.get("products_json")
    images = request.files.getlist("images")

    saved_files = []
    points = []
    try:
        products = json.loads(products_json)
    except:
        return jsonify({"error": "Invalid JSON"}), 400

    for p in products:
        for img_file in images:
            filename = f"{uuid.uuid4()}_{img_file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img_file.save(filepath)
            saved_files.append(filename)

            # Embed image + upsert to Qdrant
            with open(filepath, "rb") as f:
                vec = embed_image_bytes(f.read())
            payload = {
                "product_id": p.get("product_id"),
                "name": p.get("name"),
                "category": p.get("category"),
                "price": p.get("price"),
                "image_url": f"/uploads/{filename}",
                "description": p.get("description"),
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

            # Add image info to product metadata
            p["image_url"] = f"/uploads/{filename}"

    # Upsert points to Qdrant
    vectorstore.upsert_points(session["collection"], points)

    # Update MongoDB session
    sessions_col.update_one(
        {"_id": session_id},
        {"$push": {"products": {"$each": products}}}
    )

    return jsonify({
        "status": "ok",
        "saved_images": [f"/uploads/{f}" for f in saved_files]
    })

# ---------------------------
# Text query
# ---------------------------
@app.route("/query_text", methods=["POST"])
def query_text():
    data = request.get_json()
    query = data.get("query")
    session_id = data.get("session_id")
    session = sessions_col.find_one({"_id": session_id})
    if not session:
        return jsonify({"error": "Invalid session"}), 400

    vec = embed_text(query)
    results = vectorstore.query_similar(session["collection"], vec, top_k=6)
    images = [{"url": r.payload["image_url"], "description": r.payload["description"]} for r in results]

    return jsonify({
        "answer": f"Found {len(images)} similar products:",
        "images": images
    })

# ---------------------------
# Image query
# ---------------------------
@app.route("/query_image/<session_id>", methods=["POST"])
def query_image(session_id):
    session = sessions_col.find_one({"_id": session_id})
    if not session:
        return jsonify({"error": "Invalid session"}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    vec = embed_image_bytes(open(filepath, "rb").read())
    results = vectorstore.query_similar(session["collection"], vec, top_k=6)
    images = [{"url": r.payload["image_url"], "description": r.payload["description"]} for r in results]

    return jsonify({
        "answer": f"Found {len(images)} similar products:",
        "images": images
    })

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
