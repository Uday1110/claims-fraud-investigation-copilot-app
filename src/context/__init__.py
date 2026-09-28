"""Context engineering for the copilot.

WRITE      -> ``write.py``       scratchpad notes + tool evidence stored outside the prompt window
SELECT     -> ``select.py``      per-agent state views and memory selection
COMPRESS   -> ``compress.py``    tool-output allow-lists, JSON budgets, message trimming
ISOLATE    -> ``isolate.py``     fresh, role-scoped message lists per worker agent
SUMMARIZE  -> ``summarizer.py``  running conversation summary middleware (RemoveMessage)
QUARANTINE -> ``quarantine.py``  untrusted claimant text: envelope + dual-LLM extraction + injection detection
"""
