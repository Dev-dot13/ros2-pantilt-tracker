"""
Intent Parser — pure Python, zero model, instant response.
Parses natural language commands into structured intent dicts.
LLaVA is only invoked when needs_visual is True.
"""

import re

# ---------------------------------------------------------------------------
# Keyword maps
# ---------------------------------------------------------------------------

DIRECTION_WORDS = {
    'right': 'right', 'left': 'left',
    'up': 'up', 'upward': 'up', 'upwards': 'up',
    'down': 'down', 'downward': 'down', 'downwards': 'down',
}

SPEED_WORDS = {
    'slowly': 'slow', 'slow': 'slow', 'gently': 'slow', 'carefully': 'slow',
    'fast': 'fast', 'quickly': 'fast', 'rapid': 'fast', 'rapidly': 'fast',
    'medium': 'medium', 'normal': 'medium',
}

# Objects YOLO can detect — subset of COCO 80 classes most relevant here
YOLO_CLASSES = {
    'person', 'people', 'man', 'woman', 'boy', 'girl', 'human',
    'bottle', 'cup', 'chair', 'couch', 'sofa', 'table', 'desk',
    'laptop', 'phone', 'cell phone', 'book', 'backpack', 'bag',
    'cat', 'dog', 'bird', 'car', 'bicycle', 'bike', 'motorcycle',
    'tv', 'monitor', 'keyboard', 'mouse', 'remote', 'clock',
    'vase', 'bowl', 'banana', 'apple', 'orange', 'sandwich',
}

# Normalise person synonyms to 'person'
PERSON_SYNONYMS = {
    'people', 'man', 'woman', 'boy', 'girl', 'human', 'guy', 'someone',
    'anyone', 'everybody', 'person'
}

# Colour words — presence of these signals needs_visual
COLOUR_WORDS = {
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
    'white', 'black', 'grey', 'gray', 'brown', 'cyan', 'violet',
}

# Clothing / attribute words — presence signals needs_visual
ATTRIBUTE_WORDS = {
    'jacket', 'shirt', 'tshirt', 't-shirt', 'hoodie', 'sweater',
    'coat', 'dress', 'hat', 'cap', 'glasses', 'spectacles',
    'bag', 'backpack', 'scarf', 'suit', 'uniform', 'mask',
    'holding', 'carrying', 'wearing',
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_direction(tokens):
    for t in tokens:
        if t in DIRECTION_WORDS:
            return DIRECTION_WORDS[t]
    return None


def _extract_speed(tokens):
    for t in tokens:
        if t in SPEED_WORDS:
            return SPEED_WORDS[t]
    return 'medium'


def _extract_duration(text):
    """Extract explicit duration like 'for 3 seconds'."""
    match = re.search(r'for\s+(\d+(?:\.\d+)?)\s+second', text)
    if match:
        return float(match.group(1))
    return 2.0


def _extract_target(tokens):
    """Return normalised YOLO class name or None."""
    # Check multi-word first
    text = ' '.join(tokens)
    if 'cell phone' in text:
        return 'cell phone'
    for t in tokens:
        if t in PERSON_SYNONYMS:
            return 'person'
        if t in YOLO_CLASSES:
            return t
    return None


def _extract_attribute(text):
    """
    Extract visual attribute phrase (colour + clothing/object).
    Returns string like 'red jacket' or None.
    """
    words  = text.lower().split()
    result = []
    for i, w in enumerate(words):
        if w in COLOUR_WORDS:
            # Grab the next word if it is an attribute
            if i + 1 < len(words) and (
                    words[i+1] in ATTRIBUTE_WORDS or
                    words[i+1] in YOLO_CLASSES):
                result.append(f'{w} {words[i+1]}')
            else:
                result.append(w)
        elif w in ATTRIBUTE_WORDS and i > 0 and words[i-1] not in COLOUR_WORDS:
            result.append(w)
    return ' '.join(result) if result else None


def _has_visual_attribute(text):
    words = text.lower().split()
    has_colour    = any(w in COLOUR_WORDS    for w in words)
    has_attribute = any(w in ATTRIBUTE_WORDS for w in words)
    return has_colour or has_attribute


def _until_detection(text):
    patterns = [
        r'until\s+you\s+find',
        r'until\s+you\s+see',
        r'until\s+you\s+spot',
        r'until\s+(you\s+)?(detect|find|see|spot)',
        r'till\s+you',
        r'look\s+for',
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse(command: str) -> dict:
    """
    Parse a natural language command string into a structured intent dict.

    Returns:
    {
        "action":         str,
        "direction":      str | None,
        "speed":          str,
        "duration":       float,
        "until_detection":bool,
        "target":         str | None,   # YOLO class
        "attribute":      str | None,   # visual attribute e.g. "red jacket"
        "needs_visual":   bool,         # True = invoke LLaVA
        "region_hint":    str | None,   # "left"/"center"/"right" for RESELECT
        "raw":            str
    }
    """
    text   = command.strip().lower()
    tokens = re.sub(r'[^\w\s]', '', text).split()

    intent = {
        'action':          'UNKNOWN',
        'direction':       None,
        'speed':           _extract_speed(tokens),
        'duration':        _extract_duration(text),
        'until_detection': _until_detection(text),
        'target':          None,
        'attribute':       None,
        'needs_visual':    False,
        'region_hint':     None,
        'raw':             command,
    }

    # ------------------------------------------------------------------
    # STOP / LOCK
    # ------------------------------------------------------------------
    stop_phrases = ['stop', 'halt', 'freeze', 'hold', 'pause']
    lock_phrases = ['stay there', 'stay put', 'hold position',
                    'dont move', "don't move", 'lock']

    if any(p in text for p in lock_phrases):
        intent['action'] = 'LOCK'
        return intent

    if any(p in text for p in stop_phrases):
        intent['action'] = 'STOP'
        return intent

    # ------------------------------------------------------------------
    # TRACK / RESUME
    # ------------------------------------------------------------------
    resume_phrases = ['start tracking', 'resume', 'track again',
                      'follow again', 'continue tracking', 'go back to tracking']
    if any(p in text for p in resume_phrases):
        intent['action'] = 'TRACK'
        return intent

    # ------------------------------------------------------------------
    # RESPOND — questions
    # ------------------------------------------------------------------
    question_words  = ['who', 'what', 'where', 'how many', 'describe',
                       'tell me', 'can you see', 'do you see']
    question_ending = text.endswith('?')
    if question_ending or any(text.startswith(q) for q in question_words):
        intent['action']       = 'RESPOND'
        intent['needs_visual'] = True
        return intent

    # ------------------------------------------------------------------
    # PAN
    # ------------------------------------------------------------------
    pan_phrases = ['look left', 'look right', 'pan left', 'pan right',
                   'turn left', 'turn right', 'move left', 'move right',
                   'go left', 'go right', 'rotate left', 'rotate right']
    if any(p in text for p in pan_phrases):
        intent['action']    = 'PAN'
        intent['direction'] = _extract_direction(tokens)
        return intent

    # ------------------------------------------------------------------
    # TILT
    # ------------------------------------------------------------------
    tilt_phrases = ['look up', 'look down', 'tilt up', 'tilt down',
                    'move up', 'move down', 'go up', 'go down',
                    'pan up', 'pan down', 'face up', 'face down']
    if any(p in text for p in tilt_phrases):
        intent['action']    = 'TILT'
        intent['direction'] = _extract_direction(tokens)
        return intent

    # ------------------------------------------------------------------
    # FIND — search until target found
    # ------------------------------------------------------------------
    find_phrases = ['find me', 'find the', 'find a', 'search for',
                    'look for', 'locate', 'where is', 'spot the']
    if any(p in text for p in find_phrases):
        intent['action']          = 'FIND'
        intent['target']          = _extract_target(tokens) or 'person'
        intent['until_detection'] = True
        attribute = _extract_attribute(text)
        if attribute:
            intent['attribute']    = attribute
            intent['needs_visual'] = True
        return intent

    # ------------------------------------------------------------------
    # RESELECT — switch among currently visible targets
    # ------------------------------------------------------------------
    reselect_phrases = ['focus on', 'switch to', 'track the other',
                        'follow the other', 'the one on the',
                        'focus the', 'change to the']
    if any(p in text for p in reselect_phrases):
        intent['action']     = 'RESELECT'
        intent['direction']  = _extract_direction(tokens)
        intent['target']     = _extract_target(tokens)
        attribute = _extract_attribute(text)
        if attribute:
            intent['attribute']    = attribute
            intent['needs_visual'] = True
        return intent

    # ------------------------------------------------------------------
    # CHANGE_TARGET — switch YOLO class
    # ------------------------------------------------------------------
    follow_phrases = ['follow', 'track', 'follow that', 'track that',
                      'follow the', 'track the', 'start following',
                      'start tracking', 'keep following', 'keep tracking']
    if any(p in text for p in follow_phrases):
        target    = _extract_target(tokens)
        attribute = _extract_attribute(text)
        if target and target != 'person':
            # Non-person target — just switch YOLO class
            intent['action'] = 'CHANGE_TARGET'
            intent['target'] = target
        elif attribute:
            # Person with attribute — need visual grounding
            intent['action']       = 'RESELECT'
            intent['target']       = target or 'person'
            intent['attribute']    = attribute
            intent['needs_visual'] = True
        else:
            # Generic follow — resume tracking
            intent['action'] = 'TRACK'
            intent['target'] = target or 'person'
        return intent

    # ------------------------------------------------------------------
    # SEARCH — general sweep
    # ------------------------------------------------------------------
    search_phrases = ['search', 'sweep', 'scan', 'look around']
    if any(p in text for p in search_phrases):
        intent['action'] = 'SEARCH'
        return intent

    # ------------------------------------------------------------------
    # Fallback — if direction detected, treat as PAN or TILT
    # ------------------------------------------------------------------
    direction = _extract_direction(tokens)
    if direction:
        if direction in ('left', 'right'):
            intent['action']    = 'PAN'
            intent['direction'] = direction
        else:
            intent['action']    = 'TILT'
            intent['direction'] = direction
        return intent

    # ------------------------------------------------------------------
    # Unknown — pass raw to LLaVA as last resort
    # ------------------------------------------------------------------
    intent['action']       = 'RESPOND'
    intent['needs_visual'] = True
    return intent