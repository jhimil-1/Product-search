from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

client = QdrantClient(url="http://localhost:6333", api_key=None)

def create_collection_if_not_exists(name, vector_size):
    if name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(name=name, vector_size=vector_size, distance="Cosine")

def upsert_points(collection_name, points):
    client.upsert(collection_name=collection_name, points=[PointStruct(**p) for p in points])

def query_similar(collection_name, vector, top_k=6):
    return client.search(collection_name=collection_name, query_vector=vector, limit=top_k)
