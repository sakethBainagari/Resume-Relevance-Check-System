import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """Vector search engine using ChromaDB and sentence transformers for semantic search."""

    def __init__(self, collection_name: str = "resume_analysis", persist_directory: str = "./chroma_db"):
        """Initialize the vector search engine."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient model
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it
            try:
                self.collection = self.client.create_collection(name=collection_name)
                logger.info(f"Created new collection: {collection_name}")
            except Exception as create_e:
                logger.error(f"Failed to create collection: {create_e}")
                raise

    def add_resume(self, resume_text: str, metadata: Dict[str, Any], resume_id: str = None) -> str:
        """Add a resume to the vector database."""
        if resume_id is None:
            resume_id = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(resume_text) % 10000}"

        # Generate embedding
        embedding = self.model.encode(resume_text).tolist()

        # Prepare metadata
        doc_metadata = {
            "type": "resume",
            "filename": metadata.get("filename", "Unknown"),
            "word_count": len(resume_text.split()),
            "added_at": datetime.now().isoformat(),
            **metadata
        }

        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[resume_text],
            metadatas=[doc_metadata],
            ids=[resume_id]
        )

        logger.info(f"Added resume {resume_id} to vector database")
        return resume_id

    def add_job_description(self, jd_text: str, metadata: Dict[str, Any], jd_id: str = None) -> str:
        """Add a job description to the vector database."""
        if jd_id is None:
            jd_id = f"jd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(jd_text) % 10000}"

        # Generate embedding
        embedding = self.model.encode(jd_text).tolist()

        # Prepare metadata
        doc_metadata = {
            "type": "job_description",
            "title": metadata.get("title", "Unknown Position"),
            "word_count": len(jd_text.split()),
            "added_at": datetime.now().isoformat(),
            **metadata
        }

        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[jd_text],
            metadatas=[doc_metadata],
            ids=[jd_id]
        )

        logger.info(f"Added job description {jd_id} to vector database")
        return jd_id

    def semantic_search(self, query: str, n_results: int = 10, doc_type: str = None,
                       metadata_filter: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform semantic search on the vector database."""
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()

        # Prepare where clause for filtering
        where_clause = {}
        if doc_type:
            where_clause["type"] = doc_type
        if metadata_filter:
            where_clause.update(metadata_filter)

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )

        # Format results
        formatted_results = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'id': results['ids'][0][i],
                    'document': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                formatted_results.append(result)

        return {
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results)
        }

    def find_similar_resumes(self, job_description: str, n_results: int = 10,
                           min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Find resumes most similar to a job description."""
        search_results = self.semantic_search(
            query=job_description,
            n_results=n_results * 2,  # Get more results to filter
            doc_type="resume"
        )

        # Filter and rank results
        candidates = []
        for result in search_results['results']:
            distance = result.get('distance', 1.0)
            similarity_score = 1.0 - distance  # Convert distance to similarity

            if similarity_score >= min_score:
                candidate = {
                    'resume_id': result['id'],
                    'filename': result['metadata'].get('filename', 'Unknown'),
                    'similarity_score': similarity_score,
                    'content_preview': result['document'][:500] + "..." if len(result['document']) > 500 else result['document'],
                    'metadata': result['metadata']
                }
                candidates.append(candidate)

        # Sort by similarity score
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

        return candidates[:n_results]

    def find_similar_job_descriptions(self, resume_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find job descriptions most similar to a resume."""
        search_results = self.semantic_search(
            query=resume_text,
            n_results=n_results,
            doc_type="job_description"
        )

        # Format results
        jobs = []
        for result in search_results['results']:
            distance = result.get('distance', 1.0)
            similarity_score = 1.0 - distance

            job = {
                'jd_id': result['id'],
                'title': result['metadata'].get('title', 'Unknown Position'),
                'similarity_score': similarity_score,
                'content_preview': result['document'][:300] + "..." if len(result['document']) > 300 else result['document'],
                'metadata': result['metadata']
            }
            jobs.append(job)

        return jobs

    def get_resume_by_id(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific resume by ID."""
        try:
            result = self.collection.get(ids=[resume_id])
            if result['documents']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Error retrieving resume {resume_id}: {e}")

        return None

    def get_job_description_by_id(self, jd_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific job description by ID."""
        try:
            result = self.collection.get(ids=[jd_id])
            if result['documents']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Error retrieving job description {jd_id}: {e}")

        return None

    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None):
        """Update an existing document."""
        try:
            # Get current document
            current = self.collection.get(ids=[doc_id])
            if not current['documents']:
                logger.error(f"Document {doc_id} not found")
                return False

            # Prepare update data
            update_data = {'ids': [doc_id]}

            if new_content:
                embedding = self.model.encode(new_content).tolist()
                update_data['embeddings'] = [embedding]
                update_data['documents'] = [new_content]

            if new_metadata:
                # Merge with existing metadata
                existing_metadata = current['metadatas'][0]
                updated_metadata = {**existing_metadata, **new_metadata}
                update_data['metadatas'] = [updated_metadata]

            # Update document
            self.collection.update(**update_data)
            logger.info(f"Updated document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector database."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            # Get all documents
            all_docs = self.collection.get()

            total_docs = len(all_docs['ids']) if all_docs['ids'] else 0

            # Count by type
            resume_count = 0
            jd_count = 0

            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    if doc_type == 'resume':
                        resume_count += 1
                    elif doc_type == 'job_description':
                        jd_count += 1

            return {
                'total_documents': total_docs,
                'resume_count': resume_count,
                'job_description_count': jd_count,
                'collection_name': self.collection_name
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.collection.delete()
            logger.info("Cleared all documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def export_collection(self, filename: str = None) -> str:
        """Export collection data to JSON."""
        try:
            all_docs = self.collection.get()

            export_data = {
                'collection_name': self.collection_name,
                'exported_at': datetime.now().isoformat(),
                'documents': []
            }

            if all_docs['documents']:
                for i, doc in enumerate(all_docs['documents']):
                    doc_data = {
                        'id': all_docs['ids'][i],
                        'document': doc,
                        'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                    }
                    export_data['documents'].append(doc_data)

            import json
            json_data = json.dumps(export_data, indent=2)

            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                return filename
            else:
                return json_data

        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return "{}"