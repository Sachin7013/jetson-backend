"""
Database utilities
------------------

Role:
- Provide a single entry-point to get the MongoDB collection that stores Agent tasks.

Config:
- Env vars: MONGO_URI (required), DB_NAME (default 'algo_vision'),
  COLLECTION_NAME (default 'Agents').

Used by:
- runner.py to read tasks and update statuses/timestamps.
- subscriber.py to read the task config, send heartbeat, and stop/complete tasks.

Data:
- This module does not define schemas; it just returns the collection handle.
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection


def get_collection(collection_name: str = None) -> Collection:
    """
    Connect to MongoDB and return a collection.
    
    Args:
        collection_name: Name of the collection. If None, uses COLLECTION_NAME env var (default: 'Agents')
    
    Returns:
        MongoDB Collection object
    """
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("❌ MONGO_URI not set. Please configure it in your .env file.")

    db_name = os.getenv("DB_NAME", "algo_vision")
    
    # Use provided collection name, or fall back to env var, or default to 'Agents'
    if collection_name:
        col_name = collection_name
    else:
        col_name = os.getenv("COLLECTION_NAME", "Agents")

    client = MongoClient(mongo_uri)
    print(f"✅ Connected to MongoDB: {db_name}.{col_name}")
    return client[db_name][col_name]


