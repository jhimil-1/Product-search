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

    def create_collection_if_not_exists(self, name: str, vector_size: int = 512) -> bool:
        """Create collection if it doesn't exist
        
        Args:
            name: Name of the collection
            vector_size: Dimensionality of the vectors (default: 512 to match CLIP model)
        """
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if name not in collections:
                self.client.recreate_collection(
                    collection_name=name,
                    vectors_config={
                        "image": models.VectorParams(
                            size=vector_size,  # Now using 512 to match CLIP
                            distance=models.Distance.COSINE
                        )
                    }
                )
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
        if not points:
            logger.warning("No points to upsert")
            return False
            
        try:
            self.create_collection_if_not_exists(collection_name)
            
            # Process points in batches to avoid timeouts
            batch_size = 50
            total_points = len(points)
            
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                point_objects = []
                
                for point in batch:
                    try:
                        point_id = point.get("id") or str(uuid.uuid4())
                        vector = point.get("vector")
                        payload = point.get("payload", {})
                        
                        if not vector:
                            logger.warning(f"Skipping point {point_id}: No vector provided")
                            continue
                            
                        # Ensure payload is a dictionary
                        if not isinstance(payload, dict):
                            payload = {"value": payload}
                            
                        # Convert datetime objects to ISO format strings
                        for key, value in payload.items():
                            if hasattr(value, 'isoformat') and callable(value.isoformat):
                                payload[key] = value.isoformat()
                            elif isinstance(value, (bytes, bytearray)):
                                payload[key] = str(value)
                        
                        point_objects.append(models.PointStruct(
                            id=point_id,
                            vector={"image": vector},  # Use named vector
                            payload=payload
                        ))
                        
                    except Exception as e:
                        logger.error(f"Error creating point: {str(e)}", exc_info=True)
                        continue
                
                if point_objects:
                    logger.info(f"Upserting batch of {len(point_objects)} points to {collection_name}")
                    try:
                        self.client.upsert(
                            collection_name=collection_name,
                            points=point_objects,
                            wait=True
                        )
                        logger.info(f"Successfully upserted {len(point_objects)} points")
                    except Exception as e:
                        logger.error(f"Error upserting batch: {str(e)}", exc_info=True)
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in upsert_points: {str(e)}", exc_info=True)
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
            
            # Search with named vector "image"
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=("image", vector),  # Use named vector
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

# Note: We don't initialize the vectorstore here anymore
# It will be initialized in app.py with proper configuration
