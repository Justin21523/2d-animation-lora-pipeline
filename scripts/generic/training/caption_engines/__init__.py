"""
Caption generation engines for creating training data descriptions.

Supports multiple caption generation methods:
- VLM-based (Qwen2-VL, InternVL2, BLIP2)
- API-based (LLMProvider, GPT-4V)
- Template-based (schema-guided)
- Hybrid (combines VLM + templates)
"""

# Always available (no external dependencies)
from .template_engine import TemplateCaptionEngine, generate_template_caption

__all__ = [
    'TemplateCaptionEngine',
    'generate_template_caption',
]

# Optional API-based engines
try:
    from .llm_provider_api_engine import LLMProviderAPICaptionEngine, generate_llm_provider_caption
    __all__.extend(['LLMProviderAPICaptionEngine', 'generate_llm_provider_caption'])
except ImportError:
    LLMProviderAPICaptionEngine = None
    generate_llm_provider_caption = None

try:
    from .openai_api_engine import OpenAIAPICaptionEngine, generate_openai_caption
    __all__.extend(['OpenAIAPICaptionEngine', 'generate_openai_caption'])
except ImportError:
    OpenAIAPICaptionEngine = None
    generate_openai_caption = None

# Optional VLM engines (require additional dependencies)
try:
    from .qwen2_vl_engine import Qwen2VLCaptionEngine, generate_qwen2vl_caption
    __all__.extend(['Qwen2VLCaptionEngine', 'generate_qwen2vl_caption'])
except ImportError:
    Qwen2VLCaptionEngine = None
    generate_qwen2vl_caption = None

try:
    from .internvl2_engine import InternVL2CaptionEngine, generate_internvl2_caption
    __all__.extend(['InternVL2CaptionEngine', 'generate_internvl2_caption'])
except ImportError:
    InternVL2CaptionEngine = None
    generate_internvl2_caption = None
