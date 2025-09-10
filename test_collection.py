import os
import sys
from vectorstore import VectorStore
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_collection_creation():
    # Initialize vector store
    vector_store = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        api_key=os.getenv("QDRANT_API_KEY", None)
    )
    
    test_collection = "test_collection_123"
    
    try:
        # Try to create a test collection
        logger.info(f"Creating test collection: {test_collection}")
        success = vector_store.create_collection(test_collection, recreate=True)
        
        if success:
            logger.info(f"Successfully created collection: {test_collection}")
            
            # Test searching by product_id
            logger.info("Testing search by product_id...")
            try:
                results = vector_store.search_by_filter(
                    collection_name=test_collection,
                    filter_conditions={"product_id": "test123"},
                    limit=1
                )
                logger.info(f"Search by product_id successful. Found {len(results)} results.")
            except Exception as e:
                logger.error(f"Error searching by product_id: {str(e)}", exc_info=True)
            
            return True
        else:
            logger.error("Failed to create collection")
            return False
            
    except Exception as e:
        logger.error(f"Error in test_collection_creation: {str(e)}", exc_info=True)
        return False
    finally:
        # Clean up
        try:
            vector_store.client.delete_collection(test_collection)
            logger.info(f"Deleted test collection: {test_collection}")
        except Exception as e:
            logger.warning(f"Could not delete test collection: {str(e)}")

if __name__ == "__main__":
    test_collection_creation()
