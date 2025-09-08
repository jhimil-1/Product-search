import requests
import os
import json
import logging
import time
from pathlib import Path
from PIL import Image, ImageDraw
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
BASE_URL = "http://localhost:5000"
UPLOAD_FOLDER = "test_uploads"
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Create test uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def wait_for_server(url, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Wait for the server to become available"""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/")
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            logger.info(f"Waiting for server to start... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    logger.error("Server did not start in time")
    return False

def create_test_image(filename, size=(100, 100), text="Test Image"):
    """Create a simple test image"""
    img = Image.new('RGB', size, color='red')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill='white')
    img.save(filename)
    return filename

class TestClient:
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()

    def create_session(self):
        try:
            response = self.session.post(f"{self.base_url}/create_session", timeout=10)
            response.raise_for_status()
            session_id = response.json().get("session_id")
            logger.info(f"Created session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise

    def upload_products(self, session_id, products, images):
        try:
            files = [('images', (os.path.basename(img), open(img, 'rb'), 'image/jpeg')) 
                    for img in images]
            
            data = {
                'products_json': json.dumps(products)
            }
            
            logger.info(f"Uploading {len(products)} products to session {session_id}")
            
            # Send request with increased timeout
            response = self.session.post(
                f"{self.base_url}/upload_products/{session_id}",
                data=data,
                files=files,
                timeout=30  # 30 seconds timeout
            )
            
            # Log response
            logger.info(f"Status code: {response.status_code}")
            logger.info(f"Response: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to upload products: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise

def test_invalid_session(client):
    """Test uploading with an invalid session ID"""
    logger.info("\n--- Testing invalid session ---")
    try:
        # This should return a 400 response, not raise an exception
        response = client.session.post(
            f"{client.base_url}/upload_products/invalid-session-123",
            data={'products_json': json.dumps([{"name": "Test"}])},
            files=[('images', ('test.jpg', open(create_test_image('test.jpg'), 'rb'), 'image/jpeg'))]
        )
        
        # Check if we got a 400 response as expected
        assert response.status_code == 400, f"Expected 400 for invalid session, got {response.status_code}"
        assert "error" in response.json(), "Response should contain an error message"
        logger.info("✅ Test invalid session: PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Test invalid session: FAILED - {str(e)}")
        raise
    finally:
        # Clean up the test image
        if os.path.exists('test.jpg'):
            os.remove('test.jpg')

def test_missing_products_json(client, session_id):
    """Test missing products JSON in request"""
    logger.info("\n--- Testing missing products JSON ---")
    try:
        files = [('images', (os.path.basename('test.jpg'), open(create_test_image('test.jpg'), 'rb'), 'image/jpeg'))]
        response = requests.post(
            f"{client.base_url}/upload_products/{session_id}",
            files=files
        )
        assert response.status_code == 400, "Expected 400 for missing products JSON"
        logger.info("✅ Test missing products JSON: PASSED")
    except Exception as e:
        logger.error(f"❌ Test missing products JSON: FAILED - {str(e)}")
        raise

def test_empty_products_list(client, session_id):
    """Test uploading empty products list"""
    logger.info("\n--- Testing empty products list ---")
    try:
        # Create a test image
        test_img = 'test_empty.jpg'
        create_test_image(test_img, text="Empty Products Test")
        
        # Make request with empty products list
        with open(test_img, 'rb') as img_file:
            files = [('images', (test_img, img_file, 'image/jpeg'))]
            data = {'products_json': json.dumps([])}
            
            response = client.session.post(
                f"{client.base_url}/upload_products/{session_id}",
                data=data,
                files=files
            )
        
        # Should return 400 with error message
        assert response.status_code == 400, "Expected 400 for empty products list"
        response_data = response.json()
        assert "error" in response_data, "Response should contain an error message"
        logger.info("✅ Test empty products list: PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Test empty products list: FAILED - {str(e)}")
        raise
    finally:
        # Clean up test image
        if os.path.exists(test_img):
            os.remove(test_img)

def test_large_image_upload(client, session_id):
    """Test uploading large images"""
    logger.info("\n--- Testing large image upload ---")
    large_img = os.path.join(UPLOAD_FOLDER, f"large_image_{uuid.uuid4()}.jpg")
    
    try:
        # Create a large but valid test image (2MB)
        width, height = 2000, 2000
        img = Image.new('RGB', (width, height), color='red')
        draw = ImageDraw.Draw(img)
        
        # Add some text and patterns to make it a valid image
        for i in range(0, 2000, 200):
            draw.line([(i, 0), (i, height)], fill='blue', width=5)
            draw.line([(0, i), (width, i)], fill='green', width=5)
        
        draw.text((width//2, height//2), "LARGE TEST IMAGE", fill='white', anchor='mm')
        
        # Save with high quality to increase size
        img.save(large_img, quality=95, subsampling=0)
        
        # Verify the image is large enough
        file_size = os.path.getsize(large_img) / (1024 * 1024)  # in MB
        logger.info(f"Created test image: {large_img} ({file_size:.2f} MB)")
        
        products = [{
            "product_id": f"large_img_test_{uuid.uuid4()}",
            "name": "Large Image Test",
            "category": "test",
            "price": 99.99,
            "description": "Testing large image upload"
        }]
        
        # Upload the large image
        response = client.upload_products(
            session_id=session_id,
            products=products,
            images=[large_img]
        )
        
        # Check if upload was successful
        assert isinstance(response, dict), f"Expected dict response, got {type(response)}"
        assert "status" in response, "Response should contain status"
        assert response.get("status") == "success", f"Upload failed: {response.get('message', 'Unknown error')}"
        assert "saved_images" in response, "Response should contain saved_images"
        assert len(response["saved_images"]) == 1, "Should have one saved image"
        
        logger.info("✅ Test large image upload: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test large image upload: FAILED - {str(e)}")
        if os.path.exists(large_img):
            logger.error(f"Test image size: {os.path.getsize(large_img) / (1024 * 1024):.2f} MB")
        raise
    finally:
        # Ensure the file handle is closed and file is deleted
        if 'img' in locals():
            del img
        if os.path.exists(large_img):
            try:
                os.remove(large_img)
            except Exception as e:
                logger.warning(f"Could not remove {large_img}: {str(e)}")

def test_special_characters(client, session_id):
    """Test special characters in product data"""
    logger.info("\n--- Testing special characters ---")
    img_path = os.path.join(UPLOAD_FOLDER, f"special_{uuid.uuid4()}.jpg")
    
    try:
        create_test_image(img_path, text="Special Chars Test")
        
        special_product = {
            "product_id": f"special_{uuid.uuid4()}",
            "name": "Product with special chars: !@#$%^&*()_+{}|:<>?",
            "category": "test-123",
            "price": 123.45,
            "description": "Testing special characters: áéíóú 你好 こんにちは 🚀"
        }
        
        response = client.upload_products(
            session_id=session_id,
            products=[special_product],
            images=[img_path]
        )
        
        # Check if upload was successful
        assert isinstance(response, dict), f"Expected dict response, got {type(response)}"
        assert "status" in response, "Response should contain status"
        assert response.get("status") == "success", f"Upload failed: {response.get('message', 'Unknown error')}"
        assert "saved_images" in response, "Response should contain saved_images"
        assert len(response["saved_images"]) == 1, "Should have one saved image"
        
        logger.info("✅ Test special characters: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test special characters: FAILED - {str(e)}")
        raise
    finally:
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                logger.warning(f"Could not remove {img_path}: {str(e)}")

def test_upload_with_image_urls(client, session_id):
    """Test uploading products with image URLs"""
    logger.info("\n--- Testing upload with image URLs ---")
    
    # Create test_uploads directory if it doesn't exist
    os.makedirs("test_uploads", exist_ok=True)
    
    # Create a test image in the test_uploads directory
    test_image_path = os.path.abspath(os.path.join("test_uploads", "test_url_image.jpg"))
    create_test_image(test_image_path, text="Test URL Image")
    
    try:
        # Create a test product with the local file path
        test_product = {
            "product_id": f"url_test_{uuid.uuid4()}",
            "name": "Test Product with URL",
            "category": "test",
            "price": 199.99,
            "description": "Testing product with image URL",
            "image_url": test_image_path  # Use direct absolute file path
        }
        
        logger.info(f"Using test image at: {test_image_path}")
        
        # Make request with just the product data (no file uploads)
        response = client.session.post(
            f"{client.base_url}/upload_products/{session_id}",
            data={"products_json": json.dumps([test_product])}
        )
        
        # Check response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        response_data = response.json()
        assert response_data.get("status") == "success", f"Upload failed: {response_data}"
        assert "saved_images" in response_data, "Response missing saved_images"
        assert len(response_data["saved_images"]) > 0, "No images were saved"
        
        # Verify the file was saved to the uploads directory
        saved_filename = os.path.basename(response_data["saved_images"][0])
        saved_path = os.path.join("uploads", saved_filename)
        assert os.path.exists(saved_path), f"Image file was not saved to {saved_path}"
        
        logger.info("✅ Test upload with image URL: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test upload with image URL: FAILED - {str(e)}")
        raise
    finally:
        # Clean up test files
        if os.path.exists(test_image_path):
            try:
                os.remove(test_image_path)
            except Exception as e:
                logger.warning(f"Could not remove {test_image_path}: {str(e)}")

def main():
    client = TestClient()
    
    try:
        # Wait for server
        if not wait_for_server(BASE_URL):
            raise Exception("Server not available")
        
        # Create a test session
        logger.info("\n--- Creating test session ---")
        session_id = client.create_session()
        
        # Run test cases
        test_invalid_session(client)
        test_missing_products_json(client, session_id)
        test_empty_products_list(client, session_id)
        test_large_image_upload(client, session_id)
        test_special_characters(client, session_id)
        test_upload_with_image_urls(client, session_id)
        
        logger.info("\n🎉 All tests completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}", exc_info=True)
        return 1
    finally:
        logger.info("\nTest execution completed")

if __name__ == "__main__":
    exit(main())
