from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
from embeddings import TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM
from typing import Optional, List, Dict, Any, Union, Tuple
import uuid
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, host: str, port: int = 6333, api_key: Optional[str] = None):
        """
        Initialize the VectorStore with Qdrant connection details.
        
        Args:
            host: Qdrant host URL (e.g., 'localhost' or cloud URL)
            port: Qdrant port (default: 6333)
            api_key: API key for Qdrant Cloud (required for cloud instances)
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> QdrantClient:
        """Initialize and return Qdrant client with connection test"""
        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            
            client = QdrantClient(
                url=self.host,
                port=self.port,
                api_key=self.api_key,
                prefer_grpc=True,  # gRPC is more efficient for cloud connections
                timeout=10.0,
                grpc_options={
                    "grpc.keepalive_time_ms": 30000,  # Send keepalive pings every 30 seconds
                    "grpc.keepalive_timeout_ms": 10000,  # Wait 10 seconds for keepalive response
                }
            )
            
            # Test connection
            client.get_collections()
            logger.info("Successfully connected to Qdrant server")
            return client
            
        except Exception as e:
            error_msg = f"Failed to connect to Qdrant at {self.host}:{self.port}"
            if self.api_key:
                error_msg += " (with API key)"
            error_msg += f". Error: {str(e)}"
            logger.error(error_msg)
            raise

    def create_collection(self, collection_name: str, recreate: bool = False) -> bool:
        """
        Create a new collection with the given name and vector size.
        
        Args:
            collection_name: Name of the collection to create
            vector_size: Dimensionality of the vectors (default: 512 for CLIP model)
            recreate: If True, delete existing collection with the same name
            
        Returns:
            bool: True if collection was created successfully, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            
            # Define the collection configuration with both vectors
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=TEXT_EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    ),
                    "image": models.VectorParams(
                        size=IMAGE_EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                }
            )
            logger.info(f"Created collection '{collection_name}' with text and image vectors")
            
            # Create payload index for category field
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="category",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True
                    )
                )
                logger.info(f"Created text index for 'category' field in collection '{collection_name}'")
            except Exception as e:
                logger.warning(f"Could not create category index (may already exist): {str(e)}")
            
            # Create payload indexes for faster filtering
            try:
                # Index for top-level fields
                for field, schema_type in [
                    ("product_id", models.PayloadSchemaType.KEYWORD),
                    ("category", models.PayloadSchemaType.KEYWORD),
                    ("name", models.PayloadSchemaType.TEXT)
                ]:
                    try:
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field,
                            field_schema=models.TextIndexParams(
                                type="text",
                                tokenizer=models.TokenizerType.WORD,
                                min_token_len=2,
                                max_token_len=20,
                                lowercase=True
                            ) if schema_type == models.PayloadSchemaType.TEXT else schema_type
                        )
                        logger.info(f"Created index for top-level field: {field}")
                    except Exception as e:
                        logger.warning(f"Could not create index for '{field}': {str(e)}")
                
                # Index for nested payload fields
                for field in ["product_id", "name", "category"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name=f"payload.{field}",
                            field_schema=models.TextIndexParams(
                                type="text",
                                tokenizer=models.TokenizerType.WORD,
                                min_token_len=2,
                                max_token_len=20,
                                lowercase=True
                            ) if field != "product_id" else models.PayloadSchemaType.KEYWORD
                        )
                        logger.info(f"Created index for nested field: payload.{field}")
                    except Exception as e:
                        logger.warning(f"Could not create index for 'payload.{field}': {str(e)}")
                
                logger.info(f"Successfully created all payload indexes for collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Error creating payload indexes: {str(e)}", exc_info=True)
                # Don't fail the whole operation if indexing fails
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}", exc_info=True)
            return False
    
    def upsert_points(self, collection_name: str, points: List[Dict]) -> bool:
        """
        Upsert points into the collection with proper vector validation
        
        Args:
            collection_name: Name of the collection
            points: List of point dictionaries with 'id', 'vector', and 'payload' keys
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not points:
            logger.warning("No points to upsert")
            return False
            
        try:
            # Convert points to Qdrant PointStruct format
            qdrant_points = []
            
            for point in points:
                try:
                    # Ensure vectors are the correct length (512)
                    vectors = {}
                    
                    # Process text vector
                    if 'text' in point['vector'] and len(point['vector']['text']) == TEXT_EMBEDDING_DIM:
                        vectors['text'] = point['vector']['text']
                    else:
                        logger.warning(f"Invalid text vector in point {point.get('id')} - expected {TEXT_EMBEDDING_DIM} dimensions, got {len(point['vector']['text']) if 'text' in point['vector'] else 'none'}")
                        continue
                        
                    # Process image vector if present
                    if 'image' in point['vector'] and point['vector']['image'] and len(point['vector']['image']) == 512:
                        vectors['image'] = point['vector']['image']
                    else:
                        logger.info(f"No image vector for point {point.get('id')}, proceeding with text vector only")
                    
                    # Create point with validated vectors
                    qdrant_point = models.PointStruct(
                        id=point['id'],
                        vector=vectors,
                        payload=point['payload']
                    )
                    qdrant_points.append(qdrant_point)
                    
                except Exception as e:
                    logger.error(f"Error processing point {point.get('id')}: {str(e)}", exc_info=True)
                    continue
            
            if not qdrant_points:
                logger.error("No valid points to upsert after validation")
                return False
            
            # Upsert the points in smaller batches for better error handling
            batch_size = 25  # Reduced batch size for better stability
            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True
                    )
                    logger.info(f"Upserted batch {i//batch_size + 1} with {len(batch)} points")
                except Exception as e:
                    logger.error(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
                    # Try with even smaller batch if this fails
                    if len(batch) > 5:
                        logger.info("Retrying with smaller batches")
                        small_batch_size = 5
                        for j in range(0, len(batch), small_batch_size):
                            small_batch = batch[j:j + small_batch_size]
                            try:
                                self.client.upsert(
                                    collection_name=collection_name,
                                    points=small_batch,
                                    wait=True
                                )
                                logger.info(f"Upserted small batch with {len(small_batch)} points")
                            except Exception as e2:
                                logger.error(f"Error upserting small batch: {str(e2)}")
                                return False
                    else:
                        return False
            
            logger.info(f"Successfully upserted {len(qdrant_points)} points to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error in upsert_points: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    def _image_to_base64(image_data: Union[bytes, Image.Image], max_size: int = 1024) -> str:
        """Convert image to base64 string with resizing"""
        try:
            if isinstance(image_data, bytes):
                img = Image.open(BytesIO(image_data)).convert('RGB')
            else:
                img = image_data.convert('RGB')
                
            # Resize image if it's too large
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return ""
    
    def add_products(
        self,
        collection_name: str,
        products: List[Dict[str, Any]],
        text_embeddings: List[List[float]] = None,
        image_embeddings: List[List[float]] = None
    ) -> Tuple[int, int]:
        """
        Add products to the vector store with image data
        
        Args:
            collection_name: Name of the collection
            products: List of product dictionaries
            text_embeddings: Optional list of text embeddings
            image_embeddings: Optional list of image embeddings
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        if not products:
            return 0, 0
            
        points = []
        success = 0
        
        for i, product in enumerate(products):
            try:
                # Convert image to base64 if it's a file path
                image_data = product.get('image_data', '')
                if image_data and not image_data.startswith('data:image'):
                    try:
                        with open(image_data, 'rb') as f:
                            image_data = self._image_to_base64(f.read())
                    except Exception as e:
                        logger.warning(f"Could not read image file {image_data}: {str(e)}")
                        image_data = ''
                
                # Prepare vectors with validation
                vectors = {}
                
                # Add text vector if available
                text_vector = text_embeddings[i] if text_embeddings and i < len(text_embeddings) else None
                if text_vector and len(text_vector) == 512:
                    vectors['text'] = text_vector
                else:
                    logger.warning(f"Invalid text embedding for product {i}")
                    continue
                
                # Add image vector if available
                image_vector = image_embeddings[i] if image_embeddings and i < len(image_embeddings) else None
                if image_vector and len(image_vector) == 512:
                    vectors['image'] = image_vector
                
                point = {
                    'id': product.get('id', str(uuid.uuid4())),
                    'vector': vectors,
                    'payload': {
                        'product_id': product.get('product_id', ''),
                        'name': product.get('name', ''),
                        'description': product.get('description', ''),
                        'price': str(product.get('price', '')),
                        'category': product.get('category', '').lower(),
                        'image_data': image_data,
                        'metadata': product.get('metadata', {})
                    }
                }
                points.append(point)
                success += 1
                
            except Exception as e:
                logger.error(f"Error processing product {i}: {str(e)}", exc_info=True)
                continue
        
        if points:
            try:
                # Use our new upsert_points method which handles validation
                if self.upsert_points(collection_name, points):
                    logger.info(f"Successfully added {success} products to {collection_name}")
                    return success, len(products) - success
                
            except Exception as e:
                logger.error(f"Error adding products to Qdrant: {str(e)}")
                return 0, len(products)
                
        return success, len(products) - success

    def create_collection_if_not_exists(self, name: str, vector_size: int = 512) -> bool:
        """Create collection if it doesn't exist
        
        Args:
            name: Name of the collection
            vector_size: Dimensionality of the vectors (default: 512 to match CLIP model)
        """
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if name not in collections:
                self.create_collection(name)
                logger.info(f"Created new collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            raise

    def upsert_points(
        self, 
        collection_name: str, 
        points: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert points into the collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert, each with id, vector, and payload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            point_objects = []
            
            for point in points:
                try:
                    point_id = point.get("id")
                    vector = point.get("vector")
                    payload = point.get("payload", {})
                    
                    if not all([point_id, vector]):
                        logger.warning(f"Skipping invalid point: {point}")
                        continue
                    
                    # Handle different vector formats
                    if isinstance(vector, dict):
                        # If vector is already a dict with a named vector, use it as is
                        vector_dict = vector
                    else:
                        # Otherwise, wrap the vector in a dict with a default name
                        vector_dict = {"text": vector}
                    
                    # Convert point_id to string if it's a number
                    point_id = str(point_id)
                    
                    point_objects.append(models.PointStruct(
                        id=point_id,
                        vector=vector_dict,
                        payload=payload
                    ))
                except Exception as e:
                    logger.error(f"Error processing point {point.get('id')}: {str(e)}")
                    continue
            
            if not point_objects:
                logger.error("No valid points to upsert")
                return False
                
            # Upsert points
            self.client.upsert(
                collection_name=collection_name,
                points=point_objects,
                wait=True
            )
            
            logger.info(f"Successfully upserted {len(point_objects)} points to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error in upsert_points: {str(e)}")
            logger.exception("Upsert points error details:")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection by name.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful or collection didn't exist, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True
        except Exception as e:
            # If collection doesn't exist, consider it a success
            if "not found" in str(e).lower():
                logger.info(f"Collection {collection_name} not found, nothing to delete")
                return True
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        vector_name: str = "text",
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_categories: Optional[List[str]] = None,
        with_vectors: bool = False,
        with_payload: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the specified collection.
        
        Args:
            collection_name: Name of the collection to search in
            query_vector: Query vector for similarity search
            vector_name: Name of the vector to search ('text' or 'image')
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            filter_categories: Optional list of categories to filter by
            with_vectors: Whether to include vector in the response
            with_payload: Whether to include payload in the response
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            # Build the filter
            query_filter = None
            if filter_categories:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchAny(any=[c.lower() for c in filter_categories])
                        )
                    ]
                )
            
            # Perform the search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=(vector_name, query_vector),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_vectors=with_vectors,
                with_payload=with_payload
            )
            
            # Convert to list of dicts for easier handling
            results = []
            for result in search_results:
                result_dict = {
                    'id': str(result.id),
                    'score': float(result.score),
                    'payload': result.payload or {}
                }
                
                # Include vector if requested
                if with_vectors and hasattr(result, 'vector'):
                    result_dict['vector'] = result.vector
                    
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            error_msg = f"Error searching in collection {collection_name}"
            logger.error(f"{error_msg}: {str(e)}")
            raise Exception(f"{error_msg}: {str(e)}")
            
    def get_product_by_id(self, collection_name: str, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a product by its ID
        
        Args:
            collection_name: Name of the collection
            product_id: ID of the product to retrieve
            
        Returns:
            Product data if found, None otherwise
        """
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[product_id],
                with_vectors=False,
                with_payload=True
            )
            
            if not result:
                return None
                
            return {
                'id': str(result[0].id),
                'payload': result[0].payload or {}
            }
            
        except Exception as e:
            logger.error(f"Error retrieving product {product_id}: {str(e)}")
            return None

    def query_similar(
        self, 
        collection_name: str, 
        vector: List[float], 
        top_k: int = 5, 
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors from the collection.
        
        Args:
            collection_name: Name of the collection to query
            vector: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar items with their scores and payloads
        """
        try:
            if not vector:
                logger.error("Empty query vector provided")
                return []
                
            logger.info(f"Querying {collection_name} with vector of length {len(vector)}")
            
            # Search with named vector "text"
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=("text", vector),  # Specify vector name as "text"
                limit=top_k,
                score_threshold=score_threshold,
                with_vectors=False,
                with_payload=True
            )
            
            logger.info(f"Found {len(search_result)} similar items")
            
            results = []
            for hit in search_result:
                try:
                    # Handle different payload formats
                    payload = dict(hit.payload or {})
                    if not payload and hasattr(hit, 'payload') and hit.payload:
                        payload = dict(hit.payload)
                    
                    # If payload is empty, try to get it from the point
                    if not payload and hasattr(hit, 'payload'):
                        payload = dict(hit.payload) if hit.payload else {}
                    
                    # If we still don't have a payload, create a minimal one
                    if not payload:
                        payload = {"id": str(hit.id) if hasattr(hit, 'id') else str(uuid.uuid4())}
                    
                    # Ensure all values in payload are JSON serializable
                    for key, value in payload.items():
                        if hasattr(value, 'isoformat'):  # Handle datetime
                            payload[key] = value.isoformat()
                        elif isinstance(value, (bytes, bytearray)):  # Handle binary data
                            payload[key] = str(value)
                    
                    results.append({
                        "id": str(hit.id) if hasattr(hit, 'id') else str(uuid.uuid4()),
                        "score": float(hit.score) if hasattr(hit, 'score') else 0.0,
                        "payload": payload
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing search hit: {str(e)}", exc_info=True)
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in query_similar: {str(e)}", exc_info=True)
            return []

    def search_by_filter(self, collection_name: str, filter_conditions: Dict, limit: int = 10) -> List[Dict]:
        """
        Search for points in the collection using filter conditions.
        
        Args:
            collection_name: Name of the collection to search in
            filter_conditions: Dictionary of field-value pairs to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching points with payloads
        """
        try:
            if not self.collection_exists(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []
                
            must_conditions = []
            
            # Convert filter conditions to Qdrant filter conditions
            if filter_conditions:
                for field, value in filter_conditions.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value)
                        )
                    )
            
            # Create a filter with all conditions
            filter_condition = models.Filter(must=must_conditions) if must_conditions else None
            
            # Create a dummy text vector for the search
            dummy_text_vector = [0.0] * TEXT_EMBEDDING_DIM
            
            # Perform the search
            search_results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=limit,
                with_vectors=False,
                with_payload=True
            )
            
            # Process results
            results = []
            for hit in search_results[0]:  # scroll returns a tuple of (points, offset)
                try:
                    payload = dict(hit.payload or {})
                    
                    # Ensure all values in payload are JSON serializable
                    for key, value in payload.items():
                        if hasattr(value, 'isoformat'):  # Handle datetime
                            payload[key] = value.isoformat()
                        elif isinstance(value, (bytes, bytearray)):  # Handle binary data
                            payload[key] = str(value)
                    
                    results.append({
                        "id": str(hit.id) if hasattr(hit, 'id') else str(uuid.uuid4()),
                        "score": float(hit.score) if hasattr(hit, 'score') else 0.0,
                        "payload": payload
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing search hit: {str(e)}", exc_info=True)
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_by_filter: {str(e)}", exc_info=True)
            return []

    def search(
        self, 
        collection_name: str, 
        query_vector: List[float] = None, 
        vector_name: str = None,
        filter_conditions: Dict = None, 
        query_text: str = None, 
        exact_match: bool = False,
        top_k: int = 5, 
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the specified collection with optional category filtering.
        
        Args:
            collection_name: Name of the collection to search in
            query_vector: Query vector for similarity search
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            vector_name: Name of the vector to search in ("text" or "image")
            filter_conditions: Dictionary of field-value pairs to filter by
            query_text: Text query for exact matching
            exact_match: If True, perform exact text matching instead of vector search
            
        Returns:
            List of search results with payload and score
        """
        try:
            if not query_vector:
                logger.error("Empty query vector provided")
                return []
                
            logger.info(f"Searching collection '{collection_name}' with {vector_name} vector of length {len(query_vector)}")
            
            # Build filter conditions if provided
            filter_condition = None
            if filter_conditions and isinstance(filter_conditions, dict):
                logger.info(f"Applying filter conditions: {filter_conditions}")
                conditions = []
                
                # Handle MongoDB-style operators like $in
                for field, value in filter_conditions.items():
                    if isinstance(value, dict):
                        for op, op_value in value.items():
                            if op == "$in" and isinstance(op_value, list):
                                # Handle $in operator for category filtering
                                conditions.append(
                                    models.FieldCondition(
                                        key=field,
                                        match=models.MatchAny(any=op_value)
                                    )
                                )
                    else:
                        # Handle simple equality match
                        conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                # Add more filter conditions as needed
                # Example for price range:
                # if 'min_price' in filter_conditions:
                #     conditions.append(...)
                
                if conditions:
                    filter_condition = models.Filter(
                        must=conditions
                    )
            
            try:
                # If exact match is requested, use full-text search
                if exact_match and query_text:
                    logger.info(f"Performing exact match search for: {query_text}")
                    search_results = self.client.query(
                        collection_name=collection_name,
                        query_text=query_text,
                        query_filter=filter_condition,
                        limit=top_k,
                        with_vectors=False,
                        with_payload=True
                    )
                else:
                    # Perform vector search with optional filters
                    search_results = self.client.search(
                        collection_name=collection_name,
                        query_vector=(vector_name, query_vector) if vector_name else query_vector,
                        query_filter=filter_condition,
                        limit=top_k * 2,  # Get more results to ensure we have enough after filtering
                        score_threshold=score_threshold,
                        with_vectors=False,
                        with_payload=True
                    )
                
                # Additional client-side filtering for categories
                if filter_condition and search_results:
                    filtered_results = []
                    for hit in search_results:
                        try:
                            payload = getattr(hit, 'payload', {}) or {}
                            hit_category = str(payload.get('category', '')).lower()
                            # Check if any of the matched categories are in the result's category
                            if any(cat.lower() in hit_category for cat in filter_categories):
                                filtered_results.append(hit)
                                if len(filtered_results) >= top_k:
                                    break
                        except Exception as e:
                            logger.warning(f"Error processing search result: {str(e)}")
                            continue
                    
                    if filtered_results:
                        return filtered_results
                
                return search_results
                
            except Exception as e:
                logger.error(f"Error in search: {str(e)}")
                return []
            
            if not search_results:
                logger.info("No search results found")
                return []
                
            # Process and format search results
            results = []
            for hit in search_results:
                try:
                    # Handle different payload formats
                    payload = dict(hit.payload or {})
                    
                    # Ensure all values in payload are JSON serializable
                    for key, value in payload.items():
                        if hasattr(value, 'isoformat'):  # Handle datetime
                            payload[key] = value.isoformat()
                        elif isinstance(value, (bytes, bytearray)):  # Handle binary data
                            payload[key] = str(value)
                    
                    results.append({
                        "id": str(hit.id) if hasattr(hit, 'id') else str(uuid.uuid4()),
                        "score": float(hit.score) if hasattr(hit, 'score') else 0.0,
                        "payload": payload
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing search hit: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}", exc_info=True)
            return []

    def search_with_category_filter(self, collection_name, query_vector, categories, vector_name="text", top_k=5, score_threshold=0.5):
        """
        Search for similar vectors with strict category filtering
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            categories: List of categories to filter by (case-insensitive)
            vector_name: Name of the vector to search ("text" or "image")
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
        
        Returns:
            List of search results matching the category filter
        """
        try:
            from qdrant_client import models
            
            if not categories:
                logger.warning("No categories provided for category filter")
                return []
                
            # Normalize categories to lowercase for consistent matching
            categories = [str(cat).lower().strip() for cat in categories if cat and str(cat).strip()]
            if not categories:
                logger.warning("No valid categories provided after normalization")
                return []
                
            logger.info(f"Searching with strict category filter for: {categories}")
            
            # Create exact match conditions for each category
            category_conditions = []
            for category in categories:
                if not category:
                    continue
                    
                # Strict exact match condition on category field only
                category_conditions.append(
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category.lower().strip())
                    )
                )
            
            if not category_conditions:
                logger.warning("No valid category conditions created")
                return []
                
            # Create the filter with OR condition for categories
            search_filter = models.Filter(
                must=[
                    models.Filter(
                        should=category_conditions,
                        min_should_match=1
                    )
                ]
            )
            
            # First try with a higher score threshold
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(
                    name=vector_name,
                    vector=query_vector
                ) if isinstance(query_vector, list) else query_vector,
                query_filter=search_filter,
                limit=top_k * 2,  # Get more results for post-filtering
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Post-filter results to ensure they match the category exactly
            filtered_results = []
            for result in search_result:
                payload = getattr(result, 'payload', {}) or {}
                result_category = str(payload.get('category', '')).lower()
                
                # Skip if no category is available
                if not result_category:
                    logger.debug(f"Skipping result with no category: {payload.get('name', 'Unnamed')}")
                    continue
                
                # Split categories if multiple are present (comma-separated) and clean them up
                result_categories = [cat.strip().lower() for cat in result_category.split(',') if cat.strip()]
                
                # Debug logging
                logger.debug(f"Checking categories for product: {payload.get('name', 'Unnamed')}")
                logger.debug(f"Result categories: {result_categories}")
                logger.debug(f"Target categories: {categories}")
                
                # Check if any of the result's categories exactly match any of the target categories
                is_category_match = any(
                    any(
                        target_cat == result_cat
                        for result_cat in result_categories
                    )
                    for target_cat in categories
                )
                
                if is_category_match:
                    filtered_results.append(result)
                    logger.debug(f"Added product to results: {payload.get('name', 'Unnamed')} with categories: {result_categories}")
                    if len(filtered_results) >= top_k:
                        break
                else:
                    logger.debug(f"Excluded product due to category mismatch: {payload.get('name', 'Unnamed')} with categories: {result_categories}")
            
            logger.info(f"Found {len(filtered_results)} results after strict category filtering")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in category-filtered search: {str(e)}", exc_info=True)
            # Fallback to regular search without category filter
            return self.search(
                collection_name=collection_name,
                query_vector=query_vector,
                vector_name=vector_name,
                top_k=top_k,
                score_threshold=score_threshold
            )