#!/usr/bin/env python3
"""
Debug script to find products related to emerald studs
"""
import os
import sys
from vectorstore import VectorStore
from config import Config

def find_emerald_products():
    """Find all products related to emerald studs"""
    
    # Initialize vector store
    vector_store = VectorStore(
        host=Config.QDRANT_URL.replace('https://', '').replace('http://', ''),
        port=6333,
        api_key=Config.QDRANT_API_KEY
    )
    
    collection_name = "products"
    
    print("Searching for products with 'emerald' or 'studs'...")
    
    # Search for products containing "emerald"
    emerald_results = vector_store.search_with_name(
        collection_name=collection_name,
        product_name="emerald",
        exact_match=False,
        top_k=10
    )
    
    # Search for products containing "studs"
    studs_results = vector_store.search_with_name(
        collection_name=collection_name,
        product_name="studs",
        exact_match=False,
        top_k=10
    )
    
    # Search for "Emerald Studs" with fuzzy search
    emerald_studs_results = vector_store.search_with_name(
        collection_name=collection_name,
        product_name="Emerald Studs",
        exact_match=False,
        top_k=10
    )
    
    print(f"\n=== Products with 'emerald' ===")
    for product in emerald_results:
        print(f"Name: '{product.get('name')}'")
        print(f"Name Lower: '{product.get('name_lower')}'")
        print(f"Price: {product.get('price')}")
        print(f"Category: {product.get('category')}")
        print("-" * 40)
    
    print(f"\n=== Products with 'studs' ===")
    for product in studs_results:
        print(f"Name: '{product.get('name')}'")
        print(f"Name Lower: '{product.get('name_lower')}'")
        print(f"Price: {product.get('price')}")
        print(f"Category: {product.get('category')}")
        print("-" * 40)
    
    print(f"\n=== Products similar to 'Emerald Studs' ===")
    for product in emerald_studs_results:
        print(f"Name: '{product.get('name')}'")
        print(f"Name Lower: '{product.get('name_lower')}'")
        print(f"Price: {product.get('price')}")
        print(f"Category: {product.get('category')}")
        print("-" * 40)

if __name__ == "__main__":
    find_emerald_products()