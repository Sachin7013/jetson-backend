"""
MongoDB Client
==============

Singleton MongoDB client for database connections.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


class MongoClientManager:
    """
    Singleton MongoDB client manager.
    
    Manages MongoDB connections and provides access to collections.
    """
    _instance: Optional["MongoClientManager"] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize MongoDB client connection."""
        if self._client is not None:
            return  # Already initialized
        
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("❌ MONGO_URI not set. Please configure it in your .env file.")
        
        db_name = os.getenv("DB_NAME", "algo_vision")
        
        self._client = MongoClient(mongo_uri)
        self._database = self._client[db_name]
        print(f"✅ Connected to MongoDB: {db_name}")
    
    def get_database(self) -> Database:
        """Get MongoDB database instance."""
        if self._database is None:
            self._initialize_client()
        return self._database
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB Collection object
        """
        database = self.get_database()
        return database[collection_name]
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None


def get_mongo_client() -> MongoClientManager:
    """Get singleton MongoDB client manager."""
    return MongoClientManager()

