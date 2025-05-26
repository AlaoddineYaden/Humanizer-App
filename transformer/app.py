import ssl
import random
import warnings
import re
from typing import List, Dict, Optional, Tuple
import json

import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


def load_language_models():
    models = {}
    try:
        models['fr'] = spacy.load("fr_core_news_sm")
    except OSError:
        print("Warning: French SpaCy model not found. French processing will be limited.")
        models['fr'] = None
    
    try:
        models['en'] = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: English SpaCy model not found. English processing will be limited.")
        models['en'] = None
    
    if not any(models.values()):
        raise RuntimeError("At least one language model (English or French) is required.")
    
    return models


NLP_MODELS = load_language_models()


class BilingualTextHumanizer:
    """
    Bilingual text humanizer that transforms AI-generated text to appear more human-like.
    Supports both English and French languages.
    
    Features:
    - Automatic language detection
    - Language-specific synonym replacement
    - Sentence structure variation
    - Academic and casual tone options
    - Error injection for natural imperfections
    - Contextual transformations
    """

    def __init__(
        self,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        p_synonym_replacement=0.25,
        p_sentence_restructure=0.2,
        p_transition_add=0.15,
        p_minor_errors=0.05,
        tone="academic",
        language="auto",  # 'auto', 'en', 'fr'
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.nlp_models = NLP_MODELS
        self.language = language

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.model = None

        self.p_synonym_replacement = p_synonym_replacement
        self.p_sentence_restructure = p_sentence_restructure
        self.p_transition_add = p_transition_add
        self.p_minor_errors = p_minor_errors
        self.tone = tone

        self._load_resources()

    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text
        Returns 'en' for English, 'fr' for French
        """
        if self.language != "auto":
            return self.language
            
        # Simple language detection based on common words
        french_indicators = [
            'le', 'la', 'les', 'de', 'des', 'du', 'et', 'est', 'une', 'un',
            'que', 'qui', 'pour', 'avec', 'sur', 'dans', 'par', 'ce', 'cette',
            'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'prendre'
        ]
        
        english_indicators = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'she', 'or', 'an', 'will'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        french_score = sum(1 for word in words if word in french_indicators)
        english_score = sum(1 for word in words if word in english_indicators)
        
        return 'fr' if french_score > english_score else 'en'

    def _load_resources(self):
        """Load linguistic resources and dictionaries for both languages"""

        # Enhanced synonyms dictionary for both languages
        self.synonyms = {
            'fr': {
                # Common French verbs
                "dire": ["affirmer", "déclarer", "exprimer", "mentionner", "soutenir"],
                "faire": ["réaliser", "effectuer", "accomplir", "exécuter"],
                "voir": ["observer", "constater", "remarquer", "percevoir"],
                "donner": ["fournir", "offrir", "accorder", "procurer"],
                "prendre": ["saisir", "adopter", "choisir", "capturer"],
                "venir": ["arriver", "survenir", "apparaître"],
                "aller": ["se rendre", "se diriger", "partir"],
                "savoir": ["connaître", "maîtriser", "comprendre"],
                "pouvoir": ["être capable", "avoir la possibilité"],
                "vouloir": ["désirer", "souhaiter", "ambitionner"],
                # Common French adjectives
                "grand": ["important", "considérable", "majeur", "significatif"],
                "petit": ["réduit", "modeste", "limité", "mineur"],
                "bon": ["excellent", "satisfaisant", "approprié", "convenable"],
                "mauvais": ["défavorable", "négatif", "inadéquat"],
                "important": ["essentiel", "crucial", "fondamental", "primordial"],
                "différent": ["distinct", "varié", "divers", "autre"],
                "possible": ["réalisable", "envisageable", "potentiel"],
                "nouveau": ["récent", "moderne", "inédit", "innovant"],
                "premier": ["initial", "principal", "primaire"],
                "dernier": ["final", "ultime", "récent"],
                # Common French nouns
                "problème": ["difficulté", "enjeu", "défi", "obstacle"],
                "solution": ["résolution", "réponse", "remède"],
                "question": ["interrogation", "problématique", "sujet"],
                "résultat": ["conséquence", "effet", "aboutissement"],
                "exemple": ["illustration", "cas", "modèle"],
                "raison": ["motif", "cause", "justification"],
                "temps": ["période", "durée", "époque"],
                "travail": ["activité", "tâche", "emploi", "œuvre"],
                "vie": ["existence", "vécu", "expérience"],
                "monde": ["univers", "société", "planète"],
                # Common French adverbs
                "très": ["extrêmement", "particulièrement", "fort", "vraiment"],
                "bien": ["correctement", "convenablement", "parfaitement"],
                "mal": ["incorrectement", "défavorablement"],
                "souvent": ["fréquemment", "régulièrement", "habituellement"],
                "toujours": ["constamment", "perpétuellement", "continuellement"],
                "jamais": ["nullement", "en aucun cas"],
                "beaucoup": ["énormément", "considérablement", "grandement"],
                "peu": ["faiblement", "légèrement", "modérément"],
                
                # Maritime context - French
                "bateau": ["navire", "bâtiment"],
                "mer": ["domaine marin", "environnement maritime", "espace maritime"],
                "océan": ["domaine océanique", "espace maritime"],
                "vague": ["état de la mer", "houle", "oscillation marine"],
                "vent": ["facteur météorologique", "condition atmosphérique"],
                "navire": ["bâtiment", "vaisseau", "unité navale"],
                "bouger": ["naviguer", "évoluer", "progresser", "avancer"],
                "rapide": ["véloce", "swift", "accéléré"],
                "lent": ["graduel", "mesuré", "progressif"],
                "sûr": ["sécurisé", "fiable", "protégé"],
                "dangereux": ["risqué", "périlleux", "hasardeux"],
            },
            'en': {
                # Common English verbs
                "say": ["state", "declare", "express", "mention", "assert"],
                "make": ["create", "produce", "construct", "build", "form"],
                "do": ["perform", "execute", "carry out", "accomplish"],
                "get": ["obtain", "acquire", "receive", "gain"],
                "go": ["proceed", "travel", "move", "head"],
                "come": ["arrive", "approach", "reach"],
                "see": ["observe", "notice", "perceive", "view"],
                "know": ["understand", "comprehend", "realize"],
                "take": ["grab", "seize", "capture", "obtain"],
                "give": ["provide", "offer", "supply", "grant"],
                "use": ["utilize", "employ", "apply", "implement"],
                "find": ["discover", "locate", "identify", "uncover"],
                "think": ["believe", "consider", "suppose", "assume"],
                "work": ["function", "operate", "perform", "labor"],
                
                # Common English adjectives
                "big": ["large", "huge", "enormous", "massive", "significant"],
                "small": ["tiny", "little", "minor", "modest"],
                "good": ["excellent", "great", "fine", "satisfactory"],
                "bad": ["poor", "terrible", "awful", "negative"],
                "important": ["crucial", "essential", "vital", "significant"],
                "different": ["distinct", "various", "diverse", "alternative"],
                "possible": ["feasible", "potential", "viable"],
                "new": ["recent", "modern", "fresh", "innovative"],
                "old": ["ancient", "former", "previous", "outdated"],
                "right": ["correct", "proper", "appropriate"],
                "wrong": ["incorrect", "improper", "mistaken"],
                "long": ["extended", "lengthy", "prolonged"],
                "short": ["brief", "concise", "compact"],
                "high": ["elevated", "tall", "superior"],
                "low": ["reduced", "minimal", "inferior"],
                
                # Common English nouns
                "problem": ["issue", "challenge", "difficulty", "obstacle"],
                "solution": ["answer", "resolution", "remedy", "fix"],
                "question": ["inquiry", "query", "issue", "matter"],
                "result": ["outcome", "consequence", "effect", "conclusion"],
                "example": ["instance", "case", "illustration", "sample"],
                "reason": ["cause", "motive", "purpose", "justification"],
                "time": ["period", "duration", "moment", "era"],
                "work": ["job", "task", "activity", "employment"],
                "life": ["existence", "living", "experience"],
                "world": ["globe", "earth", "society", "universe"],
                "way": ["method", "approach", "manner", "technique"],
                "person": ["individual", "human", "being"],
                "place": ["location", "spot", "area", "position"],
                "thing": ["object", "item", "matter", "element"],
                
                # Common English adverbs
                "very": ["extremely", "particularly", "really", "quite"],
                "well": ["properly", "correctly", "effectively"],
                "often": ["frequently", "regularly", "commonly"],
                "always": ["constantly", "continuously", "perpetually"],
                "never": ["not ever", "at no time"],
                "much": ["greatly", "considerably", "significantly"],
                "little": ["slightly", "somewhat", "barely"],
                "just": ["simply", "merely", "only"],
                "only": ["merely", "simply", "just"],
                "also": ["additionally", "furthermore", "moreover"],
                "still": ["yet", "nonetheless", "however"],
                "even": ["including", "also", "as well"],
                
                # Maritime context - English
                "boat": ["vessel", "ship", "craft"],
                "sea": ["ocean", "marine environment", "waters"],
                "ocean": ["sea", "marine domain", "waters"],
                "wave": ["swell", "surge", "breaker"],
                "wind": ["breeze", "gale", "atmospheric condition"],
                "ship": ["vessel", "boat", "craft"],
                "move": ["navigate", "travel", "proceed", "advance"],
                "fast": ["quick", "rapid", "swift", "speedy"],
                "slow": ["gradual", "steady", "measured"],
                "safe": ["secure", "protected", "reliable"],
                "dangerous": ["risky", "hazardous", "perilous"],
            }
        }

        # Transitions for both languages
        self.transitions = {
            'fr': {
                "academic": {
                    "addition": ["De plus,", "En outre,", "Par ailleurs,", "De surcroît,", "Qui plus est,"],
                    "contrast": ["Néanmoins,", "Toutefois,", "Cependant,", "En revanche,", "Malgré cela,"],
                    "consequence": ["Par conséquent,", "Ainsi,", "De ce fait,", "Il en résulte que", "Dès lors,"],
                    "explanation": ["En effet,", "Effectivement,", "À vrai dire,", "Autrement dit,"],
                    "example": ["Par exemple,", "Notamment,", "À titre d'illustration,", "Ainsi,"],
                    "conclusion": ["En conclusion,", "Pour conclure,", "Finalement,", "En définitive,"],
                    "result": ["Par conséquent,", "Ainsi,", "De ce fait,", "En conséquence,"],
                    "emphasis": ["En effet,", "De fait,", "Notamment,", "Il convient de noter que"],
                    "neutral": ["En outre,", "De plus,", "Cependant,", "Par conséquent,", "Néanmoins,", "Ainsi,"],
                },
                "casual": {
                    "addition": ["Aussi,", "Et puis,", "En plus,", "D'ailleurs,"],
                    "contrast": ["Mais,", "Pourtant,", "Quand même,", "Malgré tout,"],
                    "consequence": ["Du coup,", "Alors,", "Donc,", "Résultat,"],
                    "explanation": ["En fait,", "Bon,", "Disons que,", "C'est-à-dire,"],
                    "example": ["Par exemple,", "Tiens,", "Comme,"],
                    "conclusion": ["Bref,", "Bon,", "Voilà,", "Au final,"],
                },
            },
            'en': {
                "academic": {
                    "addition": ["Furthermore,", "Moreover,", "Additionally,", "In addition,", "Besides,"],
                    "contrast": ["However,", "Nevertheless,", "Nonetheless,", "On the contrary,", "Conversely,"],
                    "consequence": ["Therefore,", "Consequently,", "Thus,", "As a result,", "Hence,"],
                    "explanation": ["Indeed,", "In fact,", "Specifically,", "That is to say,"],
                    "example": ["For example,", "For instance,", "Specifically,", "In particular,"],
                    "conclusion": ["In conclusion,", "To conclude,", "Finally,", "In summary,"],
                    "result": ["Consequently,", "Therefore,", "As a result,", "Hence,"],
                    "emphasis": ["Indeed,", "In fact,", "Notably,", "It should be noted that"],
                    "neutral": ["Furthermore,", "However,", "Therefore,", "Nevertheless,", "Thus,"],
                },
                "casual": {
                    "addition": ["Also,", "Plus,", "And,", "Besides,"],
                    "contrast": ["But,", "Still,", "Though,", "Anyway,"],
                    "consequence": ["So,", "Then,", "Well,", "Basically,"],
                    "explanation": ["Actually,", "Well,", "I mean,", "You know,"],
                    "example": ["Like,", "For example,", "Say,"],
                    "conclusion": ["Anyway,", "So,", "Well,", "In the end,"],
                },
            }
        }

        # Filler words for both languages
        self.fillers = {
            'fr': {
                "academic": ["il convient de noter", "il est à souligner", "on peut observer"],
                "casual": ["on peut dire", "il faut dire", "c'est vrai que"],
            },
            'en': {
                "academic": ["it should be noted", "it is worth mentioning", "one can observe"],
                "casual": ["you know", "I mean", "it's true that"],
            }
        }

        # Sentence starters for both languages
        self.sentence_starters = {
            'fr': {
                "academic": [
                    "Il convient de souligner que",
                    "Il est important de noter que",
                    "On peut observer que",
                    "Il apparaît que",
                    "Force est de constater que",
                ],
                "casual": [
                    "On peut dire que",
                    "Il faut reconnaître que",
                    "C'est vrai que",
                    "Bon, il faut dire que",
                ],
            },
            'en': {
                "academic": [
                    "It is important to note that",
                    "One can observe that",
                    "It appears that",
                    "It should be emphasized that",
                    "Research suggests that",
                ],
                "casual": [
                    "You can say that",
                    "It's true that",
                    "Well, you have to admit that",
                    "I think that",
                ],
            }
        }

    def humanize_text(self, text: str, preserve_meaning=True) -> str:
        """
        Main method to humanize text in English or French

        Args:
            text: Input text to humanize
            preserve_meaning: Whether to preserve original meaning strictly

        Returns:
            Humanized text
        """
        if not text.strip():
            return text

        # Detect language
        detected_lang = self.detect_language(text)
        
        # Get appropriate NLP model
        nlp = self.nlp_models.get(detected_lang)
        if not nlp:
            print(f"Warning: No {detected_lang} model available. Using basic processing.")
            return self._basic_humanize(text, detected_lang)

        # Split into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        transformed_sentences = []

        for i, sentence in enumerate(sentences):
            transformed = sentence

            # Apply transformations with controlled randomness
            if random.random() < self.p_synonym_replacement:
                transformed = self._replace_synonyms(transformed, detected_lang, preserve_meaning)

            if random.random() < self.p_sentence_restructure:
                transformed = self._restructure_sentence(transformed, detected_lang)

            if random.random() < self.p_transition_add and i > 0:
                transformed = self._add_transition(transformed, detected_lang)

            # Occasionally add sentence starters for variety
            if random.random() < 0.1 and not any(
                transformed.startswith(t)
                for transitions in self.transitions[detected_lang][self.tone].values()
                for t in transitions
            ):
                transformed = self._add_sentence_starter(transformed, detected_lang)

            # Add minor imperfections occasionally
            if random.random() < self.p_minor_errors:
                transformed = self._add_minor_imperfection(transformed, detected_lang)

            transformed_sentences.append(transformed)

        result = " ".join(transformed_sentences)

        # Final cleanup
        result = self._cleanup_text(result)

        return result

    def _basic_humanize(self, text: str, lang: str) -> str:
        """Basic humanization when NLP model is not available"""
        sentences = text.split('. ')
        transformed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            transformed = sentence
            
            # Add transitions occasionally
            if random.random() < self.p_transition_add and i > 0:
                transformed = self._add_transition(transformed, lang)
            
            transformed_sentences.append(transformed)
        
        return '. '.join(transformed_sentences)

    def _replace_synonyms(self, sentence: str, lang: str, preserve_meaning: bool = True) -> str:
        """Replace words with appropriate synonyms"""
        nlp = self.nlp_models.get(lang)
        if not nlp:
            return sentence
            
        doc = nlp(sentence)
        new_tokens = []
        lang_synonyms = self.synonyms.get(lang, {})

        for token in doc:
            word = token.text.lower()

            # Only replace content words
            if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and len(word) > 3:
                lemma = token.lemma_.lower()

                if lemma in lang_synonyms:
                    synonyms = lang_synonyms[lemma]

                    if preserve_meaning and self.model:
                        # Use semantic similarity for better synonym selection
                        best_synonym = self._select_best_synonym(word, synonyms)
                        if best_synonym:
                            new_tokens.append(best_synonym)
                        else:
                            new_tokens.append(token.text)
                    else:
                        # Random selection
                        if random.random() < 0.4:  # 40% chance to replace
                            new_tokens.append(random.choice(synonyms))
                        else:
                            new_tokens.append(token.text)
                else:
                    new_tokens.append(token.text)
            else:
                new_tokens.append(token.text)

        return self._reconstruct_sentence(doc, new_tokens)

    def _select_best_synonym(self, original_word: str, synonyms: List[str]) -> Optional[str]:
        """Select the most semantically similar synonym"""
        if not self.model or not synonyms:
            return None

        try:
            original_emb = self.model.encode(original_word, convert_to_tensor=True)
            synonym_embs = self.model.encode(synonyms, convert_to_tensor=True)

            cos_scores = util.cos_sim(original_emb, synonym_embs)[0]
            max_score_idx = cos_scores.argmax().item()
            max_score = cos_scores[max_score_idx].item()

            # Only use synonym if similarity is reasonable
            if max_score >= 0.6:
                return synonyms[max_score_idx]
        except Exception:
            pass

        return None

    def _restructure_sentence(self, sentence: str, lang: str) -> str:
        """Apply simple sentence restructuring"""
        # Simple transformations based on language
        if random.random() < 0.5:
            if lang == 'fr':
                # Move French adverbial phrases
                sentence_lower = sentence.lower()
                if sentence_lower.startswith(("cependant", "néanmoins", "toutefois")):
                    parts = sentence.split(",", 1)
                    if len(parts) == 2:
                        return f"{parts[1].strip()}, {parts[0].lower()}"
            elif lang == 'en':
                # Move English adverbial phrases
                sentence_lower = sentence.lower()
                if sentence_lower.startswith(("however", "nevertheless", "nonetheless")):
                    parts = sentence.split(",", 1)
                    if len(parts) == 2:
                        return f"{parts[1].strip()}, {parts[0].lower()}"

        return sentence

    def _add_transition(self, sentence: str, lang: str) -> str:
        """Add appropriate transition words"""
        transitions = self.transitions.get(lang, {}).get(self.tone, {})
        if not transitions:
            return sentence
            
        transition_type = random.choice(list(transitions.keys()))
        transition = random.choice(transitions[transition_type])

        return f"{transition} {sentence}"

    def _add_sentence_starter(self, sentence: str, lang: str) -> str:
        """Add variety with sentence starters"""
        starters = self.sentence_starters.get(lang, {}).get(self.tone, [])
        if not starters:
            return sentence
            
        starter = random.choice(starters)
        return f"{starter} {sentence}"

    def _add_minor_imperfection(self, sentence: str, lang: str) -> str:
        """Add subtle imperfections to make text more human"""
        fillers = self.fillers.get(lang, {}).get(self.tone, [])
        
        if lang == 'fr':
            imperfections = [
                # Add casual filler
                lambda s: f"{random.choice(fillers)}, {s.lower()}" if fillers else s,
                # Minor repetition
                lambda s: s.replace(" et ", " et puis ") if " et " in s else s,
                # Slight informality
                lambda s: s.replace(" très ", " assez " if random.random() < 0.5 else " plutôt "),
            ]
        else:  # English
            imperfections = [
                # Add casual filler
                lambda s: f"{random.choice(fillers)}, {s.lower()}" if fillers else s,
                # Minor repetition
                lambda s: s.replace(" and ", " and also ") if " and " in s else s,
                # Slight informality
                lambda s: s.replace(" very ", " quite " if random.random() < 0.5 else " rather "),
            ]

        if random.random() < 0.3:  # Only apply occasionally
            imperfection = random.choice(imperfections)
            return imperfection(sentence)

        return sentence

    def _reconstruct_sentence(self, doc, new_tokens: List[str]) -> str:
        """Reconstruct sentence preserving punctuation and spacing"""
        result = []
        for i, token in enumerate(doc):
            if i < len(new_tokens):
                result.append(new_tokens[i])
                if token.whitespace_:
                    result.append(token.whitespace_)

        return "".join(result)

    def _cleanup_text(self, text: str) -> str:
        """Final cleanup of the text"""
        # Fix double spaces
        text = re.sub(r"\s+", " ", text)

        # Fix punctuation spacing
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])\s*", r"\1 ", text)

        # Ensure proper capitalization after periods
        sentences = text.split(". ")
        sentences = [s.strip().capitalize() if s else s for s in sentences]
        text = ". ".join(sentences)

        return text.strip()

    def analyze_text_features(self, text: str) -> Dict:
        """Analyze text features to help with detection avoidance"""
        detected_lang = self.detect_language(text)
        nlp = self.nlp_models.get(detected_lang)
        
        if not nlp:
            return {"error": f"No {detected_lang} model available for analysis"}
            
        doc = nlp(text)

        # Calculate various metrics
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct and not token.is_space]

        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(token.lemma_.lower() for token in words))
        lexical_diversity = unique_words / len(words) if words else 0

        # POS distribution
        pos_counts = {}
        for token in words:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        return {
            "detected_language": detected_lang,
            "sentence_count": len(sentences),
            "word_count": len(words),
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "pos_distribution": pos_counts,
            "complexity_score": self._calculate_complexity(doc),
        }

    def _calculate_complexity(self, doc) -> float:
        """Calculate text complexity score"""
        # Simple complexity based on sentence length variation and vocabulary
        sentences = list(doc.sents)
        if not sentences:
            return 0

        lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0

        return min(length_variance / 10, 1.0)  # Normalize to 0-1


# Example usage
if __name__ == "__main__":
    # Initialize the bilingual humanizer
    humanizer = BilingualTextHumanizer(
        tone="academic",
        language="auto",  # Auto-detect language
        p_synonym_replacement=0.3,
        p_transition_add=0.2
    )
    
    # Test with French text
    french_text = """
    Les océans couvrent une grande partie de la surface terrestre. 
    Ils jouent un rôle important dans la régulation du climat mondial. 
    Les vagues et les marées sont des phénomènes naturels fascinants.
    """
    
    # Test with English text
    english_text = """
    The oceans cover a large part of the Earth's surface.
    They play an important role in regulating the global climate.
    Waves and tides are fascinating natural phenomena.
    """
    
    print("Original French:")
    print(french_text.strip())
    print("\nHumanized French:")
    print(humanizer.humanize_text(french_text))
    
    print("\n" + "="*50 + "\n")
    
    print("Original English:")
    print(english_text.strip())
    print("\nHumanized English:")
    print(humanizer.humanize_text(english_text))
    
    # Analysis example
    print("\n" + "="*50 + "\n")
    print("Text Analysis (French):")
    analysis = humanizer.analyze_text_features(french_text)
    for key, value in analysis.items():
        print(f"{key}: {value}")