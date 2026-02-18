"""
Vector database service for efficient similarity search and storage.
Provides in-memory vector storage with indexing for fast retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from sklearn.neighbors import NearestNeighbors
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class VectorRecord:
    """Record storing vector embedding with metadata."""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float

class VectorDatabase:
    """In-memory vector database with similarity search capabilities."""
    
    def __init__(self, storage_path: str = "./vector_db", index_type: str = "knn"):
        """
        Initialize vector database.
        
        Args:
            storage_path: Path to store database files
            index_type: Type of index to use ('knn' for k-nearest neighbors)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.index_type = index_type
        self.records: Dict[str, VectorRecord] = {}
        self.vectors: Optional[np.ndarray] = None
        self.vector_ids: List[str] = []
        self.index = None
        self.is_indexed = False
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load existing data
        self._load_database()
    
    def add_vector(self, 
                   vector_id: str, 
                   vector: np.ndarray, 
                   metadata: Dict[str, Any] = None) -> bool:
        """
        Add a vector to the database.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata associated with the vector
            
        Returns:
            True if added successfully, False if ID already exists
        """
        if vector_id in self.records:
            logger.warning(f"Vector ID {vector_id} already exists")
            return False
        
        import time
        record = VectorRecord(
            id=vector_id,
            vector=vector.copy(),
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        self.records[vector_id] = record
        self.is_indexed = False  # Mark index as stale
        
        logger.debug(f"Added vector {vector_id} with dimension {len(vector)}")
        return True
    
    def add_vectors_batch(self, 
                         vectors_data: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> int:
        """
        Add multiple vectors in batch.
        
        Args:
            vectors_data: List of (vector_id, vector, metadata) tuples
            
        Returns:
            Number of vectors successfully added
        """
        added_count = 0
        
        for vector_id, vector, metadata in vectors_data:
            if self.add_vector(vector_id, vector, metadata):
                added_count += 1
        
        logger.info(f"Added {added_count}/{len(vectors_data)} vectors to database")
        return added_count
    
    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Get a vector record by ID."""
        return self.records.get(vector_id)
    
    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the database."""
        if vector_id in self.records:
            del self.records[vector_id]
            self.is_indexed = False
            logger.debug(f"Removed vector {vector_id}")
            return True
        return False
    
    def build_index(self, n_neighbors: int = 10, algorithm: str = 'auto'):
        """
        Build search index for efficient similarity queries.
        
        Args:
            n_neighbors: Number of neighbors for KNN index
            algorithm: Algorithm to use ('auto', 'ball_tree', 'kd_tree', 'brute')
        """
        if not self.records:
            logger.warning("No vectors to index")
            return
        
        try:
            # Prepare vectors and IDs
            self.vector_ids = list(self.records.keys())
            vectors_list = [self.records[vid].vector for vid in self.vector_ids]
            self.vectors = np.vstack(vectors_list)
            
            # Build KNN index
            if self.index_type == "knn":
                self.index = NearestNeighbors(
                    n_neighbors=min(n_neighbors, len(self.vector_ids)),
                    algorithm=algorithm,
                    metric='cosine'
                )
                self.index.fit(self.vectors)
            
            self.is_indexed = True
            logger.info(f"Built {self.index_type} index for {len(self.vector_ids)} vectors")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    async def search_similar(self, 
                           query_vector: np.ndarray, 
                           k: int = 10,
                           threshold: float = 0.0,
                           filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Enhanced search for similar vectors with metadata filtering.
        
        Args:
            query_vector: Query vector to search for
            k: Number of similar vectors to return
            threshold: Minimum similarity threshold
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of (vector_id, similarity_score, metadata) tuples
        """
        if not self.is_indexed:
            self.build_index()
        
        if not self.is_indexed or self.index is None:
            logger.warning("Index not available, performing brute force search")
            return await self._brute_force_search(query_vector, k, threshold, filter_metadata)
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._knn_search_with_filter,
                query_vector,
                k,
                threshold,
                filter_metadata
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _knn_search(self, 
                   query_vector: np.ndarray, 
                   k: int, 
                   threshold: float) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform KNN search using the built index."""
        query_vector = query_vector.reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = self.index.kneighbors(query_vector, n_neighbors=k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            # Convert cosine distance to similarity
            similarity = 1 - distance
            
            if similarity >= threshold:
                vector_id = self.vector_ids[idx]
                metadata = self.records[vector_id].metadata
                results.append((vector_id, float(similarity), metadata))
        
        return results
    
    def _knn_search_with_filter(self, 
                               query_vector: np.ndarray, 
                               k: int, 
                               threshold: float,
                               filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform KNN search with metadata filtering."""
        query_vector = query_vector.reshape(1, -1)
        
        # If no filter, use regular search
        if not filter_metadata:
            return self._knn_search(query_vector, k, threshold)
        
        # Find more neighbors to account for filtering
        search_k = min(k * 3, len(self.vector_ids))  # Search 3x more to account for filtering
        distances, indices = self.index.kneighbors(query_vector, n_neighbors=search_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            # Convert cosine distance to similarity
            similarity = 1 - distance
            
            if similarity >= threshold:
                vector_id = self.vector_ids[idx]
                metadata = self.records[vector_id].metadata
                
                # Apply metadata filters
                if self._matches_filter(metadata, filter_metadata):
                    results.append((vector_id, float(similarity), metadata))
                    
                    # Stop when we have enough results
                    if len(results) >= k:
                        break
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # If filter value is a list, check if metadata value is in the list
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict) and "not" in value:
                # Support for "not" filters
                if metadata[key] == value["not"]:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    async def _brute_force_search(self, 
                                query_vector: np.ndarray, 
                                k: int, 
                                threshold: float,
                                filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform brute force similarity search with optional filtering."""
        if not self.records:
            return []
        
        similarities = []
        
        for vector_id, record in self.records.items():
            # Apply metadata filter if provided
            if filter_metadata and not self._matches_filter(record.metadata, filter_metadata):
                continue
            
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, record.vector)
            norm_query = np.linalg.norm(query_vector)
            norm_record = np.linalg.norm(record.vector)
            
            if norm_query > 0 and norm_record > 0:
                similarity = dot_product / (norm_query * norm_record)
                
                if similarity >= threshold:
                    similarities.append((vector_id, float(similarity), record.metadata))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.records:
            return {"total_vectors": 0, "is_indexed": False}
        
        vector_dims = [len(record.vector) for record in self.records.values()]
        
        return {
            "total_vectors": len(self.records),
            "is_indexed": self.is_indexed,
            "vector_dimension": vector_dims[0] if vector_dims else 0,
            "index_type": self.index_type,
            "storage_path": str(self.storage_path)
        }
    
    def save_database(self):
        """Save database to disk."""
        try:
            # Save records
            records_file = self.storage_path / "records.pkl"
            with open(records_file, 'wb') as f:
                pickle.dump(self.records, f)
            
            # Save metadata
            metadata = {
                "total_vectors": len(self.records),
                "index_type": self.index_type,
                "is_indexed": self.is_indexed
            }
            
            metadata_file = self.storage_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved database with {len(self.records)} vectors")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def _load_database(self):
        """Load database from disk."""
        try:
            records_file = self.storage_path / "records.pkl"
            if records_file.exists():
                with open(records_file, 'rb') as f:
                    self.records = pickle.load(f)
                
                logger.info(f"Loaded database with {len(self.records)} vectors")
                
                # Rebuild index if we have vectors
                if self.records:
                    self.build_index()
            
        except Exception as e:
            logger.warning(f"Could not load database: {e}")
            self.records = {}
    
    def clear_database(self):
        """Clear all vectors from the database."""
        self.records.clear()
        self.vectors = None
        self.vector_ids = []
        self.index = None
        self.is_indexed = False
        
        # Remove files
        for file_path in self.storage_path.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        logger.info("Cleared vector database")
    
    async def find_duplicates(self, 
                            similarity_threshold: float = 0.85,
                            exclude_same_metadata: Dict[str, str] = None) -> List[Tuple[str, str, float]]:
        """
        Find potential duplicate vectors in the database with enhanced filtering.
        
        Args:
            similarity_threshold: Minimum similarity to consider as duplicate
            exclude_same_metadata: Metadata fields to exclude from duplicate detection
                                 (e.g., {"student_id": "same"} to exclude same student)
            
        Returns:
            List of (id1, id2, similarity) tuples for potential duplicates
        """
        if not self.is_indexed:
            self.build_index()
        
        duplicates = []
        processed_pairs = set()
        
        for vector_id in self.vector_ids:
            record = self.records[vector_id]
            
            # Search for similar vectors
            similar_vectors = await self.search_similar(
                record.vector, 
                k=20,  # Search more to account for filtering
                threshold=similarity_threshold
            )
            
            for similar_id, similarity, similar_metadata in similar_vectors:
                # Skip self-match
                if similar_id == vector_id:
                    continue
                
                # Apply exclusion filters
                if exclude_same_metadata:
                    should_exclude = False
                    for key, value in exclude_same_metadata.items():
                        if value == "same":
                            # Exclude if both records have the same value for this key
                            if (key in record.metadata and key in similar_metadata and
                                record.metadata[key] == similar_metadata[key]):
                                should_exclude = True
                                break
                        elif key in similar_metadata and similar_metadata[key] == value:
                            should_exclude = True
                            break
                    
                    if should_exclude:
                        continue
                
                # Avoid duplicate pairs
                pair = tuple(sorted([vector_id, similar_id]))
                if pair not in processed_pairs:
                    duplicates.append((vector_id, similar_id, similarity))
                    processed_pairs.add(pair)
        
        # Sort by similarity (descending)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates
    
    async def batch_similarity_search(self, 
                                    query_vectors: List[np.ndarray],
                                    k: int = 10,
                                    threshold: float = 0.0) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Perform batch similarity search for multiple query vectors.
        
        Args:
            query_vectors: List of query vectors
            k: Number of similar vectors to return per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query vector
        """
        if not self.is_indexed:
            self.build_index()
        
        results = []
        
        try:
            if self.is_indexed and self.index is not None:
                # Use vectorized search for efficiency
                query_matrix = np.vstack(query_vectors)
                distances, indices = self.index.kneighbors(query_matrix, n_neighbors=k)
                
                for i, (query_distances, query_indices) in enumerate(zip(distances, indices)):
                    query_results = []
                    
                    for distance, idx in zip(query_distances, query_indices):
                        similarity = 1 - distance
                        
                        if similarity >= threshold:
                            vector_id = self.vector_ids[idx]
                            metadata = self.records[vector_id].metadata
                            query_results.append((vector_id, float(similarity), metadata))
                    
                    results.append(query_results)
            else:
                # Fallback to individual searches
                for query_vector in query_vectors:
                    query_results = await self.search_similar(query_vector, k, threshold)
                    results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch similarity search: {e}")
            return [[] for _ in query_vectors]
    
    def get_vectors_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Get vectors that match specific metadata criteria.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match
            
        Returns:
            List of (vector_id, vector, metadata) tuples
        """
        matching_vectors = []
        
        for vector_id, record in self.records.items():
            if self._matches_filter(record.metadata, metadata_filter):
                matching_vectors.append((vector_id, record.vector, record.metadata))
        
        return matching_vectors


# Global instance
vector_db = VectorDatabase()