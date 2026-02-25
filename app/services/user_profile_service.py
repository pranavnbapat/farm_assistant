# app/services/user_profile_service.py
"""
User Profile & Memory Service

This service builds and maintains user profiles by:
1. Extracting key attributes from conversations
2. Storing facts and preferences in Django backend
3. Providing context for personalized responses
"""

import logging
import re

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import httpx

from app.config import get_settings


S = get_settings()
logger = logging.getLogger("farm-assistant.profile")

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
    crops: List[str] = None
    topics: List[str] = None
    
    def __post_init__(self):
        if self.crops is None:
            self.crops = []
        if self.topics is None:
            self.topics = []


@dataclass
class UserFact:
    """A specific fact about the user."""
    category: str
    text: str
    confidence: float = 1.0


class UserProfileService:
    """Service for managing user profiles and memory.
    
    NOTE: This uses simple keyword-based extraction for speed and reliability.
    For production scale, consider these improvements:
    1. Use LLM-based extraction for complex patterns (costs more, slower)
    2. Load keywords from database/config for easy updates without code deploy
    3. Use embeddings to match against agricultural taxonomy/ontologies
    4. Use NER (Named Entity Recognition) models trained on agricultural text
    """
    
    # Keywords for expertise detection - could be loaded from DB/config
    EXPERTISE_KEYWORDS = {
        'beginner': [
            'new to', 'just started', 'beginner', 'learning', 'novice',
            'first time', 'getting started', 'help me understand',
            'what is', 'how do i', 'explain', 'don\'t know',
            'never done', 'no experience', 'new farmer'
        ],
        'expert': [
            'experienced', 'expert', 'advanced', 'specialist',
            'professional', 'many years', 'deep knowledge',
            'research', 'optimize', 'efficiency', 'yield improvement',
            'technical', 'implementation', 'integration'
        ]
    }
    
    # Farm type keywords
    FARM_TYPE_KEYWORDS = {
        'organic': ['organic', 'bio', 'natural farming', 'no chemicals', 'pesticide-free'],
        'conventional': ['conventional', 'traditional', 'standard'],
        'dairy': ['dairy', 'milk', 'cows', 'cattle', 'livestock'],
        'arable': ['arable', 'crops', 'grain', 'cereals'],
        'mixed': ['mixed farm', 'diverse', 'multiple crops'],
        'horticulture': ['horticulture', 'vegetables', 'fruits', 'greenhouse'],
        'vineyard': ['vineyard', 'wine', 'grapes', 'viticulture']
    }
    
    # Common EU crops and livestock
    CROP_KEYWORDS = [
        'wheat', 'barley', 'corn', 'maize', 'rice', 'oats', 'rye',
        'potato', 'potatoes', 'tomato', 'tomatoes', 'onion', 'onions',
        'carrot', 'carrots', 'lettuce', 'cabbage', 'cauliflower',
        'apple', 'apples', 'pear', 'pears', 'grape', 'grapes', 'olive', 'olives',
        'sunflower', 'rapeseed', 'canola', 'soybean', 'soybeans',
        'sugar beet', 'sugarbeet', 'cotton',
        'cow', 'cows', 'cattle', 'dairy', 'beef',
        'pig', 'pigs', 'pork', 'swine',
        'sheep', 'lamb', 'goat', 'goats',
        'chicken', 'chickens', 'poultry', 'egg', 'eggs',
        'horse', 'horses'
    ]
    
    # Agricultural topics
    TOPIC_KEYWORDS = {
        'pest_control': ['pest', 'pest control', 'insect', 'aphid', 'disease', 'fungus'],
        'soil_health': ['soil', 'fertilizer', 'compost', 'nutrient', 'pH', 'organic matter'],
        'irrigation': ['irrigation', 'water', 'drainage', 'drought', 'moisture'],
        'machinery': ['tractor', 'machinery', 'equipment', 'harvester', 'tools'],
        'climate': ['weather', 'climate', 'temperature', 'rain', 'frost', 'drought'],
        'regulations': ['regulation', 'policy', 'subsidy', 'EU', 'compliance', 'certification'],
        'sustainability': ['sustainable', 'carbon', 'environment', 'biodiversity', 'eco'],
        'economics': ['price', 'market', 'cost', 'profit', 'economics', 'business']
    }
    
    @classmethod
    async def get_or_create_profile(cls, user_uuid: str, auth_token: str = None) -> UserProfile:
        """Get existing profile or create new one."""
        logger.info(f"get_or_create_profile called: user_uuid={user_uuid}, has_auth={bool(auth_token)}")
        
        if not auth_token:
            logger.warning("No auth_token provided, returning empty profile")
            return UserProfile(user_uuid=user_uuid)
            
        # Ensure Bearer prefix is present
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        
        logger.info(f"Fetching profile for user {user_uuid[:8]}...")
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
                topics=profile_data.get('common_topics', [])
            )
        
        # Try to create profile
        logger.debug("Profile not found, creating new one")
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
                elif r.status_code == 401:
                    logger.warning("Profile fetch failed: 401 Unauthorized - check JWT token")
                elif r.status_code == 403:
                    logger.warning("Profile fetch failed: 403 Forbidden - check permissions")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch profile: {e}")
        return None
    
    @classmethod
    async def _create_profile(cls, user_uuid: str, auth_header: str) -> bool:
        """Create new profile in Django backend."""
        # Profile is created automatically on first GET request to the API
        # Just make a GET call which will create the profile with defaults
        profile = await cls._fetch_profile(user_uuid, auth_header)
        return profile is not None
    
    @classmethod
    async def update_profile(cls, user_uuid: str, updates: Dict[str, Any], auth_token: str) -> bool:
        """Update profile with new information."""
        if not CHAT_BACKEND_URL or not auth_token:
            return False
        
        # Ensure Bearer prefix is present
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        
        url = f"{CHAT_BACKEND_URL}/chat/user/profile/"
        headers = {"Authorization": auth_header}
        
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=S.VERIFY_SSL) as client:
                r = await client.patch(url, json=updates, headers=headers)
                logger.debug(f"Profile update response: {r.status_code}")
                return r.status_code == 200
        except httpx.HTTPError as e:
            logger.warning(f"Failed to update profile: {e}")
        return False
    
    @classmethod
    async def add_fact(cls, user_uuid: str, fact: UserFact, session_uuid: Optional[str] = None, auth_token: str = None) -> bool:
        """Add a fact about the user."""
        if not CHAT_BACKEND_URL or not auth_token:
            return False
        
        # Ensure Bearer prefix is present
        auth_header = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        
        url = f"{CHAT_BACKEND_URL}/chat/user/facts/"
        headers = {"Authorization": auth_header}
        payload = {
            "fact_category": fact.category,
            "fact_text": fact.text,
            "confidence_score": fact.confidence,
            "source_session_uuid": session_uuid
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
        
        # Ensure Bearer prefix is present
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
    def analyze_message(cls, message: str) -> Dict[str, Any]:
        """
        Analyze a user message to extract profile information.
        Returns extracted attributes without saving.
        """
        message_lower = message.lower()
        extracted = {
            'expertise_level': None,
            'farm_type': None,
            'crops': [],
            'topics': [],
            'facts': [],
            'region': None
        }
        
        # Detect expertise level
        for level, keywords in cls.EXPERTISE_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                extracted['expertise_level'] = level
                break
        
        # Detect farm type
        for ftype, keywords in cls.FARM_TYPE_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                extracted['farm_type'] = ftype
                break
        
        # Detect crops/livestock
        for crop in cls.CROP_KEYWORDS:
            if crop in message_lower:
                extracted['crops'].append(crop)
        
        # Detect topics
        for topic, keywords in cls.TOPIC_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                extracted['topics'].append(topic)
        
        # Extract specific facts
        facts = cls._extract_facts(message)
        extracted['facts'] = facts

        # Try to detect region (basic): "in X", "from X", "located in X"
        # NOTE: this is best-effort and should be replaced with NER/geocoding later.
        region_patterns = [
            r'\b(?:in|from|located in)\s+([a-z][a-z\s\-]{1,50})\b',
        ]
        for pattern in region_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                potential_region = match.group(1).strip(" ,.")
                # Avoid obvious non-locations / determiners
                if potential_region not in {'the', 'my', 'our', 'this', 'that'}:
                    # Title-case for nicer display; keep original if you prefer
                    extracted['region'] = potential_region.title()
                    break
        
        return extracted
    
    @classmethod
    def _extract_facts(cls, message: str) -> List[UserFact]:
        """Extract specific facts from message."""
        facts = []
        message_lower = message.lower()

        # Issue patterns (keep simple, but avoid false positives)
        issue_patterns = [
            # "problem/issue/trouble with X"
            r'(?:have|has|having|facing|dealing with)\s+(?:a\s+)?(?:problem|issue|trouble)\s+(?:with\s+)?(.+?)(?:\.|,| and | but |$)',
            # "X is not working" / "X isn't working" / "X stopped working"
            r'(.+?)\s+(?:is|are)\s+(?:not\s+working|n\'t\s+working)',
            r'(.+?)\s+stopped\s+working',
            # "struggling with X"
            r'struggling\s+(?:with\s+)?(.+?)(?:\.|,|$)',
            # "pest/disease on X"
            r'(?:aphid|pest|disease|fungus|weed)\s+(?:on|in)\s+(.+?)(?:\.|,|$)',
        ]

        for pattern in issue_patterns:
            for match in re.finditer(pattern, message_lower, re.IGNORECASE):
                issue_text = match.group(1).strip(" ,.")
                if len(issue_text) > 3:
                    facts.append(UserFact(
                        category='issue',
                        text=f"User mentioned issue with: {issue_text}",
                        confidence=0.8
                    ))
        
        # Preference patterns
        if 'prefer' in message_lower or 'like to' in message_lower:
            pref_match = re.search(r'(?:prefer|like to)\s+(.+?)(?:\.|,|$)', message_lower)
            if pref_match:
                facts.append(UserFact(
                    category='preference',
                    text=f"User prefers: {pref_match.group(1).strip()}",
                    confidence=0.9
                ))
        
        # Tool/equipment mentions
        tool_patterns = [
            r'using\s+(?:a\s+)?(.+?)(?:\s+to\s+|\.|,)',
            r'(?:have|own|bought)\s+(?:a\s+)?(.+?)(?:\.|,|$)',
        ]
        for pattern in tool_patterns:
            match = re.search(pattern, message_lower)
            if match:
                tool = match.group(1).strip()
                if len(tool) > 2 and len(tool) < 50:
                    facts.append(UserFact(
                        category='tool',
                        text=f"User uses/has: {tool}",
                        confidence=0.7
                    ))
        
        # Experience/New farmer patterns
        if 'new to' in message_lower or 'just started' in message_lower or 'beginner' in message_lower:
            exp_match = re.search(r'(?:new to|just started|beginner at)\s+(.+?)(?:\.|,|$)', message_lower)
            if exp_match:
                facts.append(UserFact(
                    category='experience',
                    text=f"User is new to: {exp_match.group(1).strip()}",
                    confidence=0.85
                ))
            else:
                facts.append(UserFact(
                    category='experience',
                    text="User is new to farming",
                    confidence=0.7
                ))
        
        # Location-based facts
        location_patterns = [
            r'\b(?:in|from|located in)\s+([a-z][a-z\s\-]{1,50})\b',
            r'\b(?:farm|land|property)\s+(?:in|at|near)\s+([a-z][a-z\s\-]{1,50})\b',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                location = match.group(1).strip(" ,.")
                if location and location not in {'the', 'my', 'our', 'this'}:
                    facts.append(UserFact(
                        category='location',
                        text=f"User farms in/location: {location.title()}",
                        confidence=0.75
                    ))
        
        # Farm type declaration
        farm_type_patterns = [
            r'(?:have|run|operate|manage)\s+(?:an?\s+)?(organic|conventional|dairy|mixed|arable|horticulture)\s+farm',
            r'(?:organic|conventional|dairy|mixed|arable|horticulture)\s+farm(?:er)?',
        ]
        for pattern in farm_type_patterns:
            match = re.search(pattern, message_lower)
            if match:
                farm_type = match.group(1) if match.groups() else match.group(0).split()[0]
                facts.append(UserFact(
                    category='farm_type',
                    text=f"User has {farm_type} farm",
                    confidence=0.8
                ))
        
        logger.debug(f"Extracted {len(facts)} facts from message")
        return facts

    @classmethod
    async def process_conversation_turn(
        cls,
        user_uuid: str,
        session_uuid: str,
        user_message: str,
        assistant_message: str,
        auth_token: str
    ) -> None:
        """
        Process a conversation turn to update user profile.
        Only processes substantive messages (skips greetings, meta-instructions, etc.)
        """
        if not auth_token:
            logger.warning("No auth token provided for profile update")
            return
        
        logger.info(f"Processing turn for user {user_uuid}: '{user_message[:50]}...'")
        
        try:
            # Analyze user message
            extracted = cls.analyze_message(user_message)
            logger.info(f"Extracted from message: expertise={extracted['expertise_level']}, "
                       f"farm_type={extracted['farm_type']}, crops={extracted['crops']}, "
                       f"facts={len(extracted['facts'])}")
            
            # Get current profile (creates if not exists)
            profile = await cls.get_or_create_profile(user_uuid, auth_token)
            logger.info(f"Loaded profile for {user_uuid}: expertise={profile.expertise_level}")
            
            # Prepare updates
            updates = {}
            
            if extracted['expertise_level'] and extracted['expertise_level'] != profile.expertise_level:
                updates['expertise_level'] = extracted['expertise_level']
            
            if extracted['farm_type'] and extracted['farm_type'] != profile.farm_type:
                updates['farm_type'] = extracted['farm_type']
            
            if extracted['region'] and extracted['region'] != profile.region:
                updates['region'] = extracted['region']
            
            # Merge crops (add new ones)
            new_crops = [c for c in extracted['crops'] if c not in profile.crops]
            if new_crops:
                updates['crops_list'] = profile.crops + new_crops
            
            # Merge topics
            new_topics = [t for t in extracted['topics'] if t not in profile.topics]
            if new_topics:
                updates['common_topics'] = profile.topics + new_topics
            
            # Update profile if we have changes
            if updates:
                logger.info(f"Updating profile with: {updates}")
                success = await cls.update_profile(user_uuid, updates, auth_token)
                if success:
                    logger.info(f"Successfully updated profile for {user_uuid}")
                else:
                    logger.warning(f"Failed to update profile for {user_uuid}")
            else:
                logger.info("No profile updates needed")
            
            # Add extracted facts
            if extracted['facts']:
                logger.info(f"Adding {len(extracted['facts'])} facts")
                for fact in extracted['facts']:
                    success = await cls.add_fact(user_uuid, fact, session_uuid, auth_token)
                    if success:
                        logger.info(f"Added fact: {fact.text[:50]}...")
                    else:
                        logger.warning(f"Failed to add fact: {fact.text[:50]}...")
            else:
                logger.info("No facts to add")
                
        except Exception as e:
            logger.error(f"Error processing conversation turn: {e}", exc_info=True)
    
    @classmethod
    def build_profile_context(cls, profile: UserProfile, facts: List[Dict] = None) -> str:
        """
        Build a context string from user profile to include in prompts.
        This provides personalization without sending full chat history.
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
        
        # Recent facts
        if facts:
            important_facts = [f for f in facts if f.get('confidence', 0) > 0.7][:3]
            for fact in important_facts:
                context_parts.append(f"Note: {fact['text']}")
        
        if context_parts:
            return "User Profile:\n" + "\n".join(f"- {part}" for part in context_parts)
        
        return ""

    # -------------------------------------------------------------------------
    # ADVANCED: LLM-based extraction (optional, for future enhancement)
    # -------------------------------------------------------------------------
    
    EXTRACTION_PROMPT = """Analyze the following user message and extract structured information for a farmer profile.

User message: "{message}"

Extract the following (respond ONLY in JSON format):
{{
    "expertise_level": "beginner|intermediate|expert|null",
    "farm_type": "organic|conventional|dairy|arable|mixed|horticulture|null",
    "region": "country or region name or null",
    "crops": ["list", "of", "crops", "mentioned"],
    "topics": ["pest_control|soil_health|irrigation|machinery|climate|regulations|sustainability|economics"],
    "facts": [
        {{
            "category": "issue|preference|tool|experience|location|farm_type",
            "text": "detailed fact text",
            "confidence": 0.0-1.0
        }}
    ]
}}

Rules:
- Set null for fields not mentioned
- Be specific in fact texts
- Confidence should reflect certainty (0.9=very certain, 0.5=guess)
- Only include agriculture-related facts
"""

    @classmethod
    async def analyze_message_with_llm(cls, message: str) -> Dict[str, Any]:
        """
        Use LLM for smarter extraction. More accurate but slower/expensive.
        Use this when keyword matching is insufficient.
        """
        try:
            from app.clients.vllm_client import generate_once
            
            prompt = cls.EXTRACTION_PROMPT.format(message=message)
            response = await generate_once(
                prompt,
                temperature=0.1,  # Low temp for consistent JSON
                max_tokens=500,
            )
            
            # Parse JSON response
            import json
            # Find JSON block in response
            # Find the first JSON object block in response (non-greedy)
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Basic sanity checks to reduce downstream surprises
                if not isinstance(data, dict):
                    raise ValueError("LLM response JSON is not an object")

                crops = data.get('crops', [])
                topics = data.get('topics', [])
                facts_raw = data.get('facts', [])

                if crops is None:
                    crops = []
                if topics is None:
                    topics = []
                if facts_raw is None:
                    facts_raw = []

                return {
                    'expertise_level': data.get('expertise_level'),
                    'farm_type': data.get('farm_type'),
                    'region': data.get('region'),
                    'crops': crops if isinstance(crops, list) else [],
                    'topics': topics if isinstance(topics, list) else [],
                    'facts': [
                        UserFact(
                            category=f.get('category', 'preference'),
                            text=f.get('text', ''),
                            confidence=f.get('confidence', 0.7)
                        )
                        for f in (facts_raw if isinstance(facts_raw, list) else [])
                        if isinstance(f, dict)
                    ]
                }
        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to keywords: {e}")
        
        # Fallback to keyword extraction
        return cls.analyze_message(message)
