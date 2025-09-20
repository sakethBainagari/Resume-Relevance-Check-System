import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeDatabase:
    """SQLite database for storing resume analysis results and metadata."""

    def __init__(self, db_path: str = "resume_analysis.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create analysis_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    relevance_score REAL,
                    verdict TEXT,
                    matching_skills TEXT,  -- JSON array
                    missing_skills TEXT,   -- JSON array
                    experience_match TEXT,
                    education_match TEXT,
                    improvement_suggestions TEXT,  -- JSON array
                    key_strengths TEXT,    -- JSON array
                    overall_feedback TEXT,
                    analysis_time TEXT,
                    job_description_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_description_id) REFERENCES job_descriptions(id)
                )
            ''')

            # Create job_descriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT NOT NULL,
                    source TEXT,  -- 'upload' or 'text'
                    filename TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create batch_analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_name TEXT,
                    total_resumes INTEGER,
                    high_match_count INTEGER,
                    medium_match_count INTEGER,
                    low_match_count INTEGER,
                    average_score REAL,
                    job_description_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_description_id) REFERENCES job_descriptions(id)
                )
            ''')

            # Create user_sessions table for tracking usage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    user_agent TEXT,
                    ip_address TEXT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    actions_count INTEGER DEFAULT 0
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully")

    def save_job_description(self, title: str, content: str, source: str = "text", filename: str = None) -> int:
        """Save job description and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job_descriptions (title, content, source, filename)
                VALUES (?, ?, ?, ?)
            ''', (title, content, source, filename))

            job_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved job description with ID: {job_id}")
            return job_id

    def save_analysis_result(self, result: Dict[str, Any], job_description_id: int = None) -> int:
        """Save a single analysis result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Convert lists to JSON strings
            matching_skills = json.dumps(result.get('matching_skills', []))
            missing_skills = json.dumps(result.get('missing_skills', []))
            improvement_suggestions = json.dumps(result.get('improvement_suggestions', []))
            key_strengths = json.dumps(result.get('key_strengths', []))

            cursor.execute('''
                INSERT INTO analysis_results (
                    filename, relevance_score, verdict, matching_skills, missing_skills,
                    experience_match, education_match, improvement_suggestions,
                    key_strengths, overall_feedback, analysis_time, job_description_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.get('filename', 'Unknown'),
                result.get('relevance_score', 0),
                result.get('verdict', 'Low'),
                matching_skills,
                missing_skills,
                result.get('experience_match', ''),
                result.get('education_match', ''),
                improvement_suggestions,
                key_strengths,
                result.get('overall_feedback', ''),
                result.get('analysis_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                job_description_id
            ))

            result_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved analysis result with ID: {result_id}")
            return result_id

    def save_batch_analysis(self, batch_name: str, results: List[Dict[str, Any]], job_description_id: int = None) -> int:
        """Save batch analysis results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Calculate batch statistics
            total_resumes = len(results)
            high_match = len([r for r in results if r.get('verdict') == 'High'])
            medium_match = len([r for r in results if r.get('verdict') == 'Medium'])
            low_match = len([r for r in results if r.get('verdict') == 'Low'])
            avg_score = sum([r.get('relevance_score', 0) for r in results]) / total_resumes if total_resumes > 0 else 0

            # Save batch summary
            cursor.execute('''
                INSERT INTO batch_analysis (
                    batch_name, total_resumes, high_match_count, medium_match_count,
                    low_match_count, average_score, job_description_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (batch_name, total_resumes, high_match, medium_match, low_match, avg_score, job_description_id))

            batch_id = cursor.lastrowid

            # Save individual results
            for result in results:
                self.save_analysis_result(result, job_description_id)

            conn.commit()
            logger.info(f"Saved batch analysis with ID: {batch_id}")
            return batch_id

    def get_analysis_results(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve analysis results with pagination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analysis_results
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))

            rows = cursor.fetchall()
            results = []

            for row in rows:
                result = dict(row)
                # Parse JSON fields
                result['matching_skills'] = json.loads(result['matching_skills'] or '[]')
                result['missing_skills'] = json.loads(result['missing_skills'] or '[]')
                result['improvement_suggestions'] = json.loads(result['improvement_suggestions'] or '[]')
                result['key_strengths'] = json.loads(result['key_strengths'] or '[]')
                results.append(result)

            return results

    def get_batch_analyses(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve batch analysis summaries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM batch_analysis
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_job_descriptions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent job descriptions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM job_descriptions
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def search_results(self, query: str, score_min: float = 0, score_max: float = 100) -> List[Dict[str, Any]]:
        """Search analysis results by filename or content."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analysis_results
                WHERE (filename LIKE ? OR overall_feedback LIKE ?)
                AND relevance_score BETWEEN ? AND ?
                ORDER BY relevance_score DESC
            ''', (f'%{query}%', f'%{query}%', score_min, score_max))

            rows = cursor.fetchall()
            results = []

            for row in rows:
                result = dict(row)
                # Parse JSON fields
                result['matching_skills'] = json.loads(result['matching_skills'] or '[]')
                result['missing_skills'] = json.loads(result['missing_skills'] or '[]')
                result['improvement_suggestions'] = json.loads(result['improvement_suggestions'] or '[]')
                result['key_strengths'] = json.loads(result['key_strengths'] or '[]')
                results.append(result)

            return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total analyses
            cursor.execute('SELECT COUNT(*) as total FROM analysis_results')
            total_analyses = cursor.fetchone()['total']

            # Average score
            cursor.execute('SELECT AVG(relevance_score) as avg_score FROM analysis_results')
            avg_score = cursor.fetchone()['avg_score'] or 0

            # Verdict distribution
            cursor.execute('''
                SELECT verdict, COUNT(*) as count
                FROM analysis_results
                GROUP BY verdict
            ''')
            verdict_dist = {row['verdict']: row['count'] for row in cursor.fetchall()}

            # Recent activity (last 7 days)
            cursor.execute('''
                SELECT COUNT(*) as recent
                FROM analysis_results
                WHERE created_at >= datetime('now', '-7 days')
            ''')
            recent_analyses = cursor.fetchone()['recent']

            return {
                'total_analyses': total_analyses,
                'average_score': round(avg_score, 2),
                'verdict_distribution': verdict_dist,
                'recent_analyses': recent_analyses
            }

    def export_to_csv(self, filename: str = None) -> str:
        """Export all results to CSV format."""
        import pandas as pd

        results = self.get_analysis_results(limit=10000)  # Export up to 10k results

        if not results:
            return ""

        # Flatten JSON fields for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                'filename': result.get('filename', ''),
                'relevance_score': result.get('relevance_score', 0),
                'verdict': result.get('verdict', ''),
                'matching_skills': ', '.join(result.get('matching_skills', [])),
                'missing_skills': ', '.join(result.get('missing_skills', [])),
                'experience_match': result.get('experience_match', ''),
                'education_match': result.get('education_match', ''),
                'overall_feedback': result.get('overall_feedback', ''),
                'analysis_time': result.get('analysis_time', ''),
                'created_at': result.get('created_at', '')
            }
            flattened_results.append(flat_result)

        df = pd.DataFrame(flattened_results)

        if filename:
            df.to_csv(filename, index=False)
            return filename
        else:
            return df.to_csv(index=False)

    def cleanup_old_data(self, days: int = 30):
        """Clean up old analysis results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM analysis_results
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days))

            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count