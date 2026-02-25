# app/services/user_profile_service_v2.py
"""
Enhanced User Profile & Memory Service with Multi-Language Support

This service builds and maintains user profiles by:
1. Detecting the language of user queries
2. Using LLM-based extraction to understand queries in any language
3. Normalizing extracted facts to English for consistent storage
4. Deduplicating and merging facts intelligently
5. Providing context for personalized responses
"""

import logging
import re
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

import httpx

from app.config import get_settings
from app.utils.language_utils import detect_language, get_language_name, normalize_language_code

# Try to import LLM client
try:
    from app.clients.vllm_client import generate_once
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

S = get_settings()
logger = logging.getLogger("farm-assistant.profile-v2")

CHAT_BACKEND_URL = (S.CHAT_BACKEND_URL or "").rstrip("/")


@dataclass
class UserProfile:
    """Structured user profile data."""
    user_uuid: str
    expertise_level: str = "beginner"
    farm_type: Optional[str] = None
    region: Optional[str] = None
    preferred_language: str = "en"
    communication_style: str = "detailed"
    crops: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    # Enhanced fields
    query_languages: List[str] = field(default_factory=list)
    total_queries: int = 0
    first_seen: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class UserFact:
    """A specific fact about the user."""
    category: str
    text: str
    confidence: float = 1.0
    source_language: str = "en"
    source_query_hash: Optional[str] = None
    session_uuid: Optional[str] = None
    extracted_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "text": self.text,
            "confidence": self.confidence,
            "source_language": self.source_language,
            "source_query_hash": self.source_query_hash,
            "session_uuid": self.session_uuid,
            "extracted_at": self.extracted_at or datetime.utcnow().isoformat(),
        }


@dataclass
class ExtractedProfile:
    """Result of profile extraction from a message."""
    expertise_level: Optional[str] = None
    farm_type: Optional[str] = None
    region: Optional[str] = None
    crops: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    facts: List[UserFact] = field(default_factory=list)
    communication_style: Optional[str] = None
    preferred_language: Optional[str] = None
    detected_language: str = "en"


class UserProfileServiceV2:
    """
    Enhanced service for managing user profiles with multi-language support.
    
    Key improvements:
    - LLM-based extraction works with any language
    - Language detection for each query
    - Semantic deduplication of facts
    - Confidence-based conflict resolution
    """
    
    # LLM Extraction Prompt (multilingual)
    EXTRACTION_PROMPT_TEMPLATE = """You are an agricultural user profile extraction system.

Analyze the user's message and extract structured information about the user.

User Message: "{message}"
Detected Language: {language} ({language_name})

Extract and respond ONLY with valid JSON in this exact format:
{{
    "expertise_level": "beginner|intermediate|expert|null",
    "farm_type": "organic|conventional|dairy|arable|mixed|horticulture|vineyard|livestock|null",
    "region": "country or region name (in English) or null",
    "crops": ["crop1", "crop2"],
    "topics": ["pest_control", "soil_health", "irrigation", "machinery", "climate", "regulations", "sustainability", "economics", "animal_health", "crop_management"],
    "facts": [
        {{
            "category": "issue|preference|tool|experience|location|farm_type|crop|topic|goal",
            "text": "clear, concise fact in English",
            "confidence": 0.0-1.0
        }}
    ],
    "communication_preferences": {{
        "style": "detailed|concise|technical|null",
        "preferred_language": "language code if user mentions preference, else null"
    }}
}}

Important Rules:
1. The user message may be in ANY of the 24 EU languages. Understand the meaning and extract it.
2. Output ALL text fields in English for consistency, regardless of input language.
3. Set null for fields not mentioned or uncertain.
4. Be SPECIFIC in fact texts. Include: what, where, when, quantities if mentioned.
5. Confidence scoring:
   - 0.9-1.0: User explicitly stated this fact ("I have...", "My farm is...")
   - 0.7-0.8: Strongly implied from context
   - 0.5-0.6: Inferred or guessed
6. Categories for facts:
   - issue: Problems, pests, diseases, challenges user is facing
   - preference: User likes, prefers, wants to avoid
   - tool: Equipment, software, methods user uses
   - experience: Years farming, certifications, expertise areas
   - location: Region, climate zone, country
   - farm_type: Type of farming operation
   - crop: Specific crops/livestock user grows/raises
   - topic: Areas of interest for future learning
   - goal: User's objectives and plans
7. Capture communication preferences if user mentions them ("explain simply", "technical details", etc.)

Examples:
Input (German): "Ich habe ein Problem mit Blattläusen auf meinen Tomaten in Bayern"
Output: {{
    "expertise_level": null,
    "farm_type": null,
    "region": "Bavaria, Germany",
    "crops": ["tomatoes"],
    "topics": ["pest_control"],
    "facts": [
        {{
            "category": "issue",
            "text": "User has an aphid infestation on tomato plants",
            "confidence": 0.95
        }},
        {{
            "category": "location",
            "text": "User farms in Bavaria, Germany",
            "confidence": 0.9
        }}
    ],
    "communication_preferences": {{"style": null, "preferred_language": null}}
}}

Input (French): "Je suis agriculteur bio depuis 5 ans en Provence, je cultive des lavandes"
Output: {{
    "expertise_level": "intermediate",
    "farm_type": "organic",
    "region": "Provence, France",
    "crops": ["lavender"],
    "topics": [],
    "facts": [
        {{
            "category": "experience",
            "text": "User has been farming organically for 5 years",
            "confidence": 0.95
        }},
        {{
            "category": "farm_type",
            "text": "User operates an organic farm",
            "confidence": 0.95
        }}
    ],
    "communication_preferences": {{"style": null, "preferred_language": null}}
}}"""

    def __init__(self):
        self.extraction_stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }
    
    @classmethod
    async def get_or_create_profile(cls, user_uuid: str, auth_token: str = None) -> UserProfile:
        """Get existing profile or create new one."""
        logger.info(f"get_or_create_profile: user_uuid={user_uuid[:8] if user_uuid else None}..., has_auth={bool(auth_token)}")
        
        if not auth_token:
            logger.warning("No auth_token provided, returning empty profile")
            return UserProfile(user_uuid=user_uuid)
        
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        
        profile_data = await cls._fetch_profile(user_uuid, auth_header)
        if profile_data:
            logger.info(f"Loaded profile: expertise={profile_data.get('expertise_level')}, farm={profile_data.get('farm_type')}")
            return UserProfile(
                user_uuid=user_uuid,
                expertise_level=profile_data.get('expertise_level', 'beginner'),
                farm_type=profile_data.get('farm_type'),
                region=profile_data.get('region'),
                preferred_language=profile_data.get('preferred_language', 'en'),
                communication_style=profile_data.get('communication_style', 'detailed'),
                crops=profile_data.get('crops_list', []),
                topics=profile_data.get('common_topics', []),
                query_languages=profile_data.get('query_languages', []),
                total_queries=profile_data.get('total_queries', 0),
                first_seen=profile_data.get('first_seen'),
                last_updated=profile_data.get('last_updated'),
            )
        
        # Create new profile
        await cls._create_profile(user_uuid, auth_header)
        return UserProfile(user_uuid=user_uuid)
    
    @classmethod
    async def _fetch_profile(cls, user_uuid: str, auth_header: str) -> Optional[Dict]:
        """Fetch profile from Django backend."""
        if not CHAT_BACKEND_URL or not auth_header:
            return None
        
        url = f"{CHAT_BACKEND_URL}/chat/user/profile/"
        headers = {"Authorization": auth_header}
        
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=S.VERIFY_SSL) as client:
                r = await client.get(url, headers=headers)
                logger.debug(f"Profile fetch response: {r.status_code}")
                if r.status_code == 200:
                    data = r.json()
                    if data.get('status') == 'success':
                        return data.get('profile')
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch profile: {e}")
        return None
    
    @classmethod
    async def _create_profile(cls, user_uuid: str, auth_header: str) -> bool:
        """Create new profile in Django backend."""
        profile = await cls._fetch_profile(user_uuid, auth_header)
        return profile is not None
    
    @classmethod
    async def update_profile(cls, user_uuid: str, updates: Dict[str, Any], auth_token: str) -> bool:
        """Update profile with new information."""
        if not CHAT_BACKEND_URL or not auth_token:
            return False
        
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        url = f"{CHAT_BACKEND_URL}/chat/user/profile/"
        headers = {"Authorization": auth_header}
        
        # Add timestamp
        updates['last_updated'] = datetime.utcnow().isoformat()
        
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=S.VERIFY_SSL) as client:
                r = await client.patch(url, json=updates, headers=headers)
                logger.debug(f"Profile update response: {r.status_code}")
                return r.status_code == 200
        except httpx.HTTPError as e:
            logger.warning(f"Failed to update profile: {e}")
        return False
    
    @classmethod
    async def add_fact(cls, user_uuid: str, fact: UserFact, auth_token: str) -> bool:
        """Add a fact about the user."""
        if not CHAT_BACKEND_URL or not auth_token:
            return False
        
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        url = f"{CHAT_BACKEND_URL}/chat/user/facts/"
        headers = {"Authorization": auth_header}
        
        payload = {
            "fact_category": fact.category,
            "fact_text": fact.text,
            "confidence_score": fact.confidence,
            "source_session_uuid": fact.session_uuid,
            "source_language": fact.source_language,
            "metadata": {
                "extracted_at": fact.extracted_at,
                "source_query_hash": fact.source_query_hash,
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=S.VERIFY_SSL) as client:
                r = await client.post(url, json=payload, headers=headers)
                logger.debug(f"Add fact response: {r.status_code}")
                return r.status_code in [201, 200]
        except httpx.HTTPError as e:
            logger.warning(f"Failed to add fact: {e}")
        return False
    
    @classmethod
    async def get_facts(cls, user_uuid: str, auth_token: str, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get facts about a user."""
        if not CHAT_BACKEND_URL or not auth_token:
            return []
        
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        url = f"{CHAT_BACKEND_URL}/chat/user/facts/"
        headers = {"Authorization": auth_header}
        params = {"limit": limit}
        if category:
            params["category"] = category
        
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=S.VERIFY_SSL) as client:
                r = await client.get(url, params=params, headers=headers)
                logger.debug(f"Get facts response: {r.status_code}")
                if r.status_code == 200:
                    return r.json().get('results', [])
        except httpx.HTTPError as e:
            logger.warning(f"Failed to get facts: {e}")
        return []
    
    @classmethod
    async def analyze_message_multilingual(cls, message: str) -> ExtractedProfile:
        """
        Analyze a user message using LLM-based extraction.
        Works with any of the 24 EU languages.
        
        Returns extracted attributes with detected language.
        """
        if not message or not message.strip():
            return ExtractedProfile()
        
        # Detect language first
        detected_lang = detect_language(message)
        language_name = get_language_name(detected_lang)
        
        logger.info(f"Analyzing message in {language_name} ({detected_lang}): '{message[:60]}...'")
        
        # Try LLM extraction if available
        if VLLM_AVAILABLE:
            try:
                result = await cls._extract_with_llm(message, detected_lang, language_name)
                if result:
                    result.detected_language = detected_lang
                    return result
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
        
        # Fallback to keyword extraction (English only)
        logger.info("Falling back to keyword extraction")
        result = cls._extract_with_keywords(message)
        result.detected_language = detected_lang
        return result
    
    @classmethod
    async def _extract_with_llm(cls, message: str, lang_code: str, lang_name: str) -> Optional[ExtractedProfile]:
        """Use LLM for intelligent, multilingual extraction."""
        prompt = cls.EXTRACTION_PROMPT_TEMPLATE.format(
            message=message.replace('"', '\\"'),
            language=lang_code,
            language_name=lang_name
        )
        
        response = await generate_once(
            prompt,
            temperature=0.1,  # Low temp for consistent JSON
            max_tokens=800,
        )
        
        # Parse JSON response
        try:
            # Find JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return None
            
            data = json.loads(json_match.group())
            
            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Build facts list
            facts = []
            for f in data.get('facts', []):
                if isinstance(f, dict) and 'text' in f:
                    facts.append(UserFact(
                        category=f.get('category', 'preference'),
                        text=f['text'],
                        confidence=f.get('confidence', 0.7),
                        source_language=lang_code,
                    ))
            
            # Parse communication preferences
            comm_prefs = data.get('communication_preferences', {}) or {}
            
            return ExtractedProfile(
                expertise_level=data.get('expertise_level'),
                farm_type=data.get('farm_type'),
                region=data.get('region'),
                crops=[c.lower() for c in (data.get('crops') or []) if c],
                topics=data.get('topics') or [],
                facts=facts,
                communication_style=comm_prefs.get('style'),
                preferred_language=comm_prefs.get('preferred_language'),
                detected_language=lang_code,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error processing LLM response: {e}")
            return None
    
    @classmethod
    def _extract_with_keywords(cls, message: str) -> ExtractedProfile:
        """Fallback keyword-based extraction (English only)."""
        message_lower = message.lower()
        
        result = ExtractedProfile()
        
        # Expertise keywords
        expertise_keywords = {
            'beginner': ['new to', 'just started', 'beginner', 'learning', 'novice', 'first time', 'getting started'],
            'expert': ['experienced', 'expert', 'advanced', 'specialist', 'professional', 'many years', 'deep knowledge'],
        }
        for level, keywords in expertise_keywords.items():
            if any(kw in message_lower for kw in keywords):
                result.expertise_level = level
                break
        
        # Farm type keywords
        farm_type_keywords = {
            'organic': ['organic', 'bio', 'natural farming', 'no chemicals'],
            'conventional': ['conventional', 'traditional', 'standard'],
            'dairy': ['dairy', 'milk', 'cows', 'cattle', 'livestock'],
            'arable': ['arable', 'crops', 'grain', 'cereals'],
            'mixed': ['mixed farm', 'diverse', 'multiple crops'],
            'horticulture': ['horticulture', 'vegetables', 'fruits', 'greenhouse'],
            'vineyard': ['vineyard', 'wine', 'grapes', 'viticulture'],
        }
        for ftype, keywords in farm_type_keywords.items():
            if any(kw in message_lower for kw in keywords):
                result.farm_type = ftype
                break
        
        # Crop keywords
        crop_keywords = [
            'wheat', 'barley', 'corn', 'maize', 'rice', 'oats', 'rye',
            'potato', 'potatoes', 'tomato', 'tomatoes', 'onion', 'onions',
            'apple', 'apples', 'grape', 'grapes', 'olive', 'olives',
            'cow', 'cows', 'cattle', 'sheep', 'pig', 'pigs', 'chicken', 'chickens'
        ]
        result.crops = [c for c in crop_keywords if c in message_lower]
        
        # Region detection (basic)
        region_patterns = [
            r'\b(?:in|from|located in)\s+([a-z][a-z\s\-]{1,50})\b',
        ]
        for pattern in region_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                potential_region = match.group(1).strip(" ,.")
                if potential_region not in {'the', 'my', 'our', 'this', 'that'}:
                    result.region = potential_region.title()
                    break
        
        return result
    
    @classmethod
    def is_duplicate_fact(cls, new_fact: UserFact, existing_facts: List[Dict], threshold: float = 0.85) -> bool:
        """
        Check if a fact is semantically similar to existing facts.
        Uses simple text similarity. For production, use embeddings.
        """
        new_text = new_fact.text.lower()
        new_words = set(re.findall(r'\b\w+\b', new_text))
        
        for existing in existing_facts:
            existing_text = existing.get('text', '').lower()
            existing_words = set(re.findall(r'\b\w+\b', existing_text))
            
            if not new_words or not existing_words:
                continue
            
            # Jaccard similarity
            intersection = new_words & existing_words
            union = new_words | existing_words
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity >= threshold:
                    logger.debug(f"Duplicate detected (sim={similarity:.2f}): '{new_fact.text}' vs '{existing_text}'")
                    return True
        
        return False
    
    @classmethod
    async def process_conversation_turn(
        cls,
        user_uuid: str,
        session_uuid: str,
        user_message: str,
        assistant_message: str,
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Process a conversation turn to update user profile.
        Enhanced with multi-language support and deduplication.
        """
        result = {
            "profile_updated": False,
            "facts_added": 0,
            "language_detected": "en",
            "errors": []
        }
        
        if not auth_token:
            result["errors"].append("No auth token provided")
            return result
        
        try:
            # Analyze user message (multilingual)
            extracted = await cls.analyze_message_multilingual(user_message)
            result["language_detected"] = extracted.detected_language
            
            logger.info(f"Extracted from {extracted.detected_language}: "
                       f"expertise={extracted.expertise_level}, "
                       f"farm={extracted.farm_type}, "
                       f"crops={extracted.crops}, "
                       f"facts={len(extracted.facts)}")
            
            # Get current profile
            profile = await cls.get_or_create_profile(user_uuid, auth_token)
            
            # Prepare profile updates
            updates = {}
            
            # Update expertise if changed
            if extracted.expertise_level and extracted.expertise_level != profile.expertise_level:
                updates['expertise_level'] = extracted.expertise_level
            
            # Update farm type if specified
            if extracted.farm_type and extracted.farm_type != profile.farm_type:
                updates['farm_type'] = extracted.farm_type
            
            # Update region if specified
            if extracted.region and extracted.region != profile.region:
                updates['region'] = extracted.region
            
            # Merge crops (add new ones)
            new_crops = [c for c in extracted.crops if c not in profile.crops]
            if new_crops:
                updates['crops_list'] = profile.crops + new_crops
            
            # Merge topics
            new_topics = [t for t in extracted.topics if t not in profile.topics]
            if new_topics:
                updates['common_topics'] = profile.topics + new_topics
            
            # Update communication style if specified
            if extracted.communication_style and extracted.communication_style != profile.communication_style:
                updates['communication_style'] = extracted.communication_style
            
            # Update language preferences
            if extracted.preferred_language and extracted.preferred_language != profile.preferred_language:
                updates['preferred_language'] = extracted.preferred_language
            
            # Update query language tracking
            query_langs = set(profile.query_languages)
            query_langs.add(extracted.detected_language)
            if query_langs != set(profile.query_languages):
                updates['query_languages'] = list(query_langs)
            
            # Increment total queries
            updates['total_queries'] = profile.total_queries + 1
            
            # Apply profile updates
            if updates:
                logger.info(f"Updating profile with: {list(updates.keys())}")
                success = await cls.update_profile(user_uuid, updates, auth_token)
                result["profile_updated"] = success
            
            # Get existing facts for deduplication
            existing_facts = await cls.get_facts(user_uuid, auth_token, limit=50)
            
            # Add new facts with deduplication
            facts_added = 0
            for fact in extracted.facts:
                # Add metadata
                fact.session_uuid = session_uuid
                fact.source_query_hash = hashlib.sha256(user_message.encode()).hexdigest()[:16]
                fact.extracted_at = datetime.utcnow().isoformat()
                
                # Check for duplicates
                if not cls.is_duplicate_fact(fact, existing_facts):
                    success = await cls.add_fact(user_uuid, fact, auth_token)
                    if success:
                        facts_added += 1
                        logger.info(f"Added fact: {fact.text[:60]}...")
                else:
                    logger.debug(f"Skipped duplicate fact: {fact.text[:50]}...")
            
            result["facts_added"] = facts_added
            
        except Exception as e:
            logger.error(f"Error processing conversation turn: {e}", exc_info=True)
            result["errors"].append(str(e))
        
        return result
    
    @classmethod
    def build_profile_context(cls, profile: UserProfile, facts: List[Dict] = None) -> str:
        """
        Build a context string from user profile to include in prompts.
        Provides personalization without sending full chat history.
        """
        context_parts = []
        
        # Basic profile info
        if profile.expertise_level:
            context_parts.append(f"User expertise level: {profile.expertise_level}")
        
        if profile.farm_type:
            context_parts.append(f"Farm type: {profile.farm_type}")
        
        if profile.region:
            context_parts.append(f"Region: {profile.region}")
        
        if profile.crops:
            crops_str = ', '.join(profile.crops[:5])  # Limit to 5
            context_parts.append(f"Crops/livestock: {crops_str}")
        
        # Communication preference
        if profile.communication_style == 'concise':
            context_parts.append("User prefers concise answers")
        elif profile.communication_style == 'technical':
            context_parts.append("User prefers technical, detailed information")
        
        # Language preference
        if profile.preferred_language and profile.preferred_language != 'en':
            context_parts.append(f"User prefers responses in {get_language_name(profile.preferred_language)}")
        
        # Query history hint
        if profile.total_queries > 5:
            context_parts.append(f"Returning user ({profile.total_queries} queries)")
        
        # Recent high-confidence facts
        if facts:
            important_facts = [f for f in facts if f.get('confidence', 0) > 0.7][:3]
            for fact in important_facts:
                context_parts.append(f"Note: {fact['text']}")
        
        if context_parts:
            return "User Profile:\n" + "\n".join(f"- {part}" for part in context_parts)
        
        return ""


# Backward compatibility: keep old service name working
UserProfileService = UserProfileServiceV2
