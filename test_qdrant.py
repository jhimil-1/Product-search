import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qdrant_connection():
    try:
        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)
        
        # Test connection by listing collections
        collections = client.get_collections()
        logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections.")
        
        # Test creating a collection
        test_collection = "test_collection_123"
        try:
            client.delete_collection(test_collection)
            logger.info(f"Deleted existing test collection: {test_collection}")
        except Exception as e:
            if "not found" not in str(e).lower():
                logger.warning(f"Error deleting test collection: {str(e)}")
        
        # Create a test collection
        client.create_collection(
            collection_name=test_collection,
            vectors_config={"text": models.VectorParams(size=512, distance=models.Distance.COSINE)}
        )
        logger.info(f"Created test collection: {test_collection}")
        
        # Verify collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if test_collection in collection_names:
            logger.info(f"Successfully verified test collection exists")
        else:
            logger.error(f"Test collection not found after creation")
            return False
            
        # Test adding a point
        point = {
            "id": 1,
            "vector": {"text": [0.1] * 512},
            "payload": {
                "product_id": "test123",
                "name": "Test Product",
                "category": "test"
            }
        }
        
        client.upsert(
            collection_name=test_collection,
            points=[point]
        )
        logger.info("Successfully added test point to collection")
        
        # Test searching
        results = client.search(
            collection_name=test_collection,
            query_vector=("text", [0.1] * 512),
            limit=1
        )
        
        if results and len(results) > 0:
            logger.info(f"Successfully searched collection. Found {len(results)} results.")
            return True
        else:
            logger.error("No results found in search")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Qdrant connection: {str(e)}")
        return False
    finally:
        # Clean up
        try:
            client.delete_collection(test_collection)
            logger.info(f"Cleaned up test collection: {test_collection}")
        except:
            pass

if __name__ == "__main__":
    success = test_qdrant_connection()
    if success:
        print("\n✅ Qdrant connection test PASSED")
    else:
        print("\n❌ Qdrant connection test FAILED")
