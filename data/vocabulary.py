"""
Sign2Sound Phase 2 - Vocabulary Definition

This module defines the vocabulary for Phase 2 of the Sign2Sound project.
It includes 15 functional ASL signs covering greetings, basic needs, questions, and responses.
"""

# Sign to ID mapping
SIGN_TO_ID = {
    "HELLO": 0,
    "GOODBYE": 1,
    "THANK_YOU": 2,
    "PLEASE": 3,
    "YES": 4,
    "NO": 5,
    "HELP": 6,
    "WATER": 7,
    "FOOD": 8,
    "BATHROOM": 9,
    "SORRY": 10,
    "UNDERSTAND": 11,
    "WHAT": 12,
    "WHERE": 13,
    "HOW": 14
}

# ID to sign mapping (reverse lookup)
ID_TO_SIGN = {v: k for k, v in SIGN_TO_ID.items()}

# Number of classes
NUM_CLASSES = 15

# Sign categories
CATEGORIES = {
    "Greetings": ["HELLO", "GOODBYE"],
    "Courtesy": ["THANK_YOU", "PLEASE", "SORRY"],
    "Responses": ["YES", "NO"],
    "Needs": ["HELP", "WATER", "FOOD", "BATHROOM"],
    "Questions": ["UNDERSTAND", "WHAT", "WHERE", "HOW"]
}

# Category to signs mapping
CATEGORY_TO_SIGNS = CATEGORIES

# Sign to category mapping
SIGN_TO_CATEGORY = {}
for category, signs in CATEGORIES.items():
    for sign in signs:
        SIGN_TO_CATEGORY[sign] = category


def get_sign_id(sign_name: str) -> int:
    """
    Get the class ID for a given sign name.
    
    Args:
        sign_name: Name of the sign (e.g., "HELLO")
        
    Returns:
        Class ID (0-14)
        
    Raises:
        KeyError: If sign name is not in vocabulary
    """
    return SIGN_TO_ID[sign_name.upper()]


def get_sign_name(sign_id: int) -> str:
    """
    Get the sign name for a given class ID.
    
    Args:
        sign_id: Class ID (0-14)
        
    Returns:
        Sign name (e.g., "HELLO")
        
    Raises:
        KeyError: If sign ID is not in vocabulary
    """
    return ID_TO_SIGN[sign_id]


def get_category(sign_name: str) -> str:
    """
    Get the category for a given sign.
    
    Args:
        sign_name: Name of the sign
        
    Returns:
        Category name (e.g., "Greetings")
    """
    return SIGN_TO_CATEGORY.get(sign_name.upper(), "Unknown")


def get_signs_by_category(category: str) -> list:
    """
    Get all signs in a given category.
    
    Args:
        category: Category name (e.g., "Greetings")
        
    Returns:
        List of sign names in that category
    """
    return CATEGORY_TO_SIGNS.get(category, [])


def is_valid_sign(sign_name: str) -> bool:
    """
    Check if a sign name is in the vocabulary.
    
    Args:
        sign_name: Name of the sign
        
    Returns:
        True if sign is in vocabulary, False otherwise
    """
    return sign_name.upper() in SIGN_TO_ID


def get_all_signs() -> list:
    """
    Get all sign names in the vocabulary.
    
    Returns:
        List of all sign names
    """
    return list(SIGN_TO_ID.keys())


def print_vocabulary():
    """Print the complete vocabulary in a formatted way."""
    print("=" * 60)
    print("Sign2Sound Phase 2 - Vocabulary")
    print("=" * 60)
    print(f"\nTotal Signs: {NUM_CLASSES}\n")
    
    for category, signs in CATEGORIES.items():
        print(f"\n{category}:")
        for sign in signs:
            sign_id = SIGN_TO_ID[sign]
            print(f"  [{sign_id:2d}] {sign}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo usage
    print_vocabulary()
    
    print("\n\nExample Usage:")
    print("-" * 60)
    
    # Get sign ID
    sign = "HELLO"
    sign_id = get_sign_id(sign)
    print(f"Sign '{sign}' has ID: {sign_id}")
    
    # Get sign name from ID
    retrieved_sign = get_sign_name(sign_id)
    print(f"ID {sign_id} corresponds to: {retrieved_sign}")
    
    # Get category
    category = get_category(sign)
    print(f"Sign '{sign}' belongs to category: {category}")
    
    # Get all signs in a category
    greetings = get_signs_by_category("Greetings")
    print(f"\nAll greetings: {greetings}")
    
    # Check if sign is valid
    print(f"\nIs 'HELLO' valid? {is_valid_sign('HELLO')}")
    print(f"Is 'INVALID' valid? {is_valid_sign('INVALID')}")
