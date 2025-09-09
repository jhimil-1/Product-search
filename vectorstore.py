from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
from typing import Optional, List, Dict, Any
import uuid

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
                timeout=10.0
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

    def create_collection(self, collection_name: str, vector_size: int = 512, recreate: bool = False) -> None:
        """
        Create a new collection with the given name and vector size.
        
        Args:
            collection_name: Name of the collection to create
            vector_size: Dimensionality of the vectors (default: 512 for CLIP model)
            recreate: If True, delete existing collection with the same name
        """
        try:
            # Delete existing collection if recreate is True
            if recreate:
                self.delete_collection(collection_name)
            
            # Create new collection with both text and image vectors
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=vector_size,  # CLIP uses 512-dimensional vectors
                        distance=models.Distance.COSINE
                    ),
                    "image": models.VectorParams(
                        size=vector_size,  # Same size as text for CLIP
                        distance=models.Distance.COSINE
                    )
                }
            )
            logger.info(f"Created new collection: {collection_name} with text and image vectors of size {vector_size}")
            
        except Exception as e:
            error_msg = f"Failed to create collection {collection_name}"
            logger.error(f"{error_msg}: {str(e)}")
            raise Exception(f"{error_msg}: {str(e)}")

    def create_collection_if_not_exists(self, name: str, vector_size: int = 512) -> bool:
        """Create collection if it doesn't exist
        
        Args:
            name: Name of the collection
            vector_size: Dimensionality of the vectors (default: 512 to match CLIP model)
        """
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if name not in collections:
                self.create_collection(name, vector_size)
                logger.info(f"Created new collection: {name} with vector size {vector_size}")
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

    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        top_k: int = 5, 
        score_threshold: float = 0.0, 
        vector_name: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the specified collection.
        
        Args:
            collection_name: Name of the collection to search in
            query_vector: Query vector for similarity search
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            vector_name: Name of the vector to search in ("text" or "image")
            
        Returns:
            List of search results with payload and score
        """
        try:
            if not query_vector:
                logger.error("Empty query vector provided")
                return []
                
            logger.info(f"Searching collection '{collection_name}' with {vector_name} vector of length {len(query_vector)}")
            
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=(vector_name, query_vector),
                limit=top_k,
                score_threshold=score_threshold,
                with_vectors=False,
                with_payload=True
            )
            
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

# Note: We don't initialize the vectorstore here anymore
# It will be initialized in app.py with proper configuration
