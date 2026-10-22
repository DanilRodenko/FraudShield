import os

# --- LLM (Groq) -----------------------------------------------------------------
# Non-secret settings only. The API key is read at call time in agent.get_client().
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0      # deterministic-ish: the note is analysis, not creative text
MAX_STEPS = 8        # hard limit on tool calls per investigation (cost + runaway loops)