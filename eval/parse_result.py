import re

def extract_answer(response_text):
    """Extract answers from the response"""
    ANSWER_REGEX = re.compile(
        r"(?:\bAnswer\s*:\s*([A-Ea-e]))|(?:\\boxed\{\s*([A-Ea-e])\s*\})|(?:([A-Ea-e])\s*\.)"
    )
    if not response_text:
        return ""
    text = response_text.strip()
    text = re.sub(r'[-*]\s*', '', text)  # Remove "- ", "* ", etc.
    last_match = None
    for m in ANSWER_REGEX.finditer(text):
        last_match = m
    if last_match:
        ch = last_match.group(1) or last_match.group(2) or last_match.group(3)
        return ch.upper()
    else:
        return ""





