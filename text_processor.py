import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTextProcessor:
    """Enhanced text processing using spaCy and NLTK for better entity extraction and normalization."""

    def __init__(self):
        """Initialize the enhanced text processor."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Downloading en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Custom stop words for resume/job description context
        self.custom_stop_words = {
            'experience', 'skills', 'education', 'qualification', 'responsibilities',
            'requirements', 'candidate', 'position', 'role', 'job', 'work', 'company',
            'team', 'project', 'development', 'management', 'support', 'system'
        }
        self.stop_words.update(self.custom_stop_words)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better analysis."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not text:
            return {}

        doc = self.nlp(text)

        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities (cities, countries)
            'SKILL': [],  # Custom skill extraction
            'EDUCATION': [],  # Education-related terms
            'EXPERIENCE': []  # Experience-related terms
        }

        # Extract standard NER entities
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Custom skill extraction (nouns that might be technical skills)
        skill_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'hibernate',
            'mysql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws',
            'azure', 'gcp', 'linux', 'git', 'jenkins', 'tensorflow', 'pytorch', 'pandas',
            'numpy', 'scikit-learn', 'machine learning', 'deep learning', 'nlp', 'ai'
        ]

        for token in doc:
            if token.text.lower() in skill_keywords and token.pos_ in ['NOUN', 'PROPN']:
                entities['SKILL'].append(token.text)

        # Education extraction
        education_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'university', 'college', 'school',
            'b.tech', 'm.tech', 'b.e', 'm.e', 'b.sc', 'm.sc', 'mba', 'mca', 'bcom', 'mcom'
        ]

        for token in doc:
            if token.text.lower() in education_keywords:
                entities['EDUCATION'].append(token.text)

        # Experience extraction
        experience_patterns = [
            r'\d+\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
            r'\d+\s*(?:months?|mos?)\s*(?:of\s*)?experience',
            r'(?:senior|junior|lead|principal)\s*(?:developer|engineer|analyst)'
        ]

        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['EXPERIENCE'].extend(matches)

        # Remove duplicates and clean
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def extract_skills(self, text: str) -> List[str]:
        """Extract technical and soft skills from text."""
        if not text:
            return []

        doc = self.nlp(text)

        skills = []

        # Technical skills patterns
        tech_skills = [
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
            'go', 'rust', 'scala', 'kotlin', 'swift', 'r', 'matlab', 'sas',

            # Web technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'spring', 'hibernate', 'jquery', 'bootstrap', 'sass',

            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server',
            'sqlite', 'cassandra', 'elasticsearch', 'dynamodb',

            # Cloud platforms
            'aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'docker', 'kubernetes',
            'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions',

            # Data Science & ML
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'opencv',

            # Tools & Others
            'git', 'linux', 'windows', 'macos', 'agile', 'scrum', 'kanban'
        ]

        # Extract skills from text
        text_lower = text.lower()
        for skill in tech_skills:
            if skill in text_lower:
                skills.append(skill.title())

        # Extract noun phrases that might be skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            # Skip if it's a stop word or too short
            if len(chunk_text) < 3 or chunk_text in self.stop_words:
                continue
            # Check if it looks like a technical term
            if any(keyword in chunk_text for keyword in ['system', 'software', 'data', 'web', 'api', 'database']):
                skills.append(chunk.text.title())

        return list(set(skills))  # Remove duplicates

    def normalize_text(self, text: str) -> str:
        """Normalize text using NLTK for better matching."""
        if not text:
            return ""

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stop words and lemmatize
        normalized_tokens = []
        for token in tokens:
            if token not in self.stop_words and token.isalnum():
                lemma = self.lemmatizer.lemmatize(token)
                normalized_tokens.append(lemma)

        return ' '.join(normalized_tokens)

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using basic NLP techniques."""
        if not text1 or not text2:
            return 0.0

        # Normalize both texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)

        # Simple word overlap similarity
        words1 = set(norm_text1.split())
        words2 = set(norm_text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases using spaCy."""
        if not text:
            return []

        doc = self.nlp(text)

        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                phrases.append(chunk.text.strip())

        # Return top phrases by frequency (simple approach)
        from collections import Counter
        phrase_counts = Counter(phrases)
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(max_phrases)]

        return top_phrases

    def analyze_resume_quality(self, resume_text: str) -> Dict[str, any]:
        """Analyze resume quality and provide insights."""
        if not resume_text:
            return {}

        analysis = {
            'word_count': len(resume_text.split()),
            'sentence_count': len(sent_tokenize(resume_text)),
            'entities': self.extract_entities(resume_text),
            'skills': self.extract_skills(resume_text),
            'key_phrases': self.extract_key_phrases(resume_text),
            'readability_score': self._calculate_readability(resume_text)
        }

        return analysis

    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score."""
        sentences = sent_tokenize(text)
        words = text.split()

        if not sentences or not words:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Simple readability formula (lower is better)
        score = avg_words_per_sentence * 0.4

        return min(score, 100.0)  # Cap at 100