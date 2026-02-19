"""
Minimal local fallback for environments without internet access.
Wan pipeline calls `ftfy.fix_text` during prompt cleaning.
"""


def fix_text(text: str) -> str:
    # Fallback behavior: keep text unchanged.
    return text

