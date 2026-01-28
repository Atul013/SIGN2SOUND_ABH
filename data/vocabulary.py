"""
Sign2Sound Phase 2 - ASL Alphabet Vocabulary

This module defines the vocabulary for Phase 2 of the Sign2Sound project.
It includes the 26 letters of the American Sign Language (ASL) fingerspelling alphabet.

This allows users to spell any word using fingerspelling, making the system
much more versatile than a limited word-based vocabulary.
"""

import string

# ASL Alphabet: A-Z (26 letters)
LETTERS = list(string.ascii_uppercase)

# Letter to ID mapping (A=0, B=1, ..., Z=25)
LETTER_TO_ID = {letter: idx for idx, letter in enumerate(LETTERS)}

# ID to letter mapping (reverse lookup)
ID_TO_LETTER = {idx: letter for letter, idx in LETTER_TO_ID.items()}

# Number of classes
NUM_CLASSES = 26

# Letter groups (for organization and learning)
LETTER_GROUPS = {
    "A-E": ["A", "B", "C", "D", "E"],
    "F-J": ["F", "G", "H", "I", "J"],
    "K-O": ["K", "L", "M", "N", "O"],
    "P-T": ["P", "Q", "R", "S", "T"],
    "U-Z": ["U", "V", "W", "X", "Y", "Z"]
}

# Difficulty levels (based on hand shape complexity)
DIFFICULTY_LEVELS = {
    "Easy": ["A", "B", "C", "O", "S"],  # Simple fist/closed hand shapes
    "Medium": ["D", "F", "G", "H", "I", "K", "L", "P", "Q", "U", "V", "W", "X", "Y"],
    "Hard": ["E", "M", "N", "R", "T", "Z"],  # Complex finger positions
    "Very Hard": ["J"]  # Requires motion
}


def get_letter_id(letter: str) -> int:
    """
    Get the class ID for a given letter.
    
    Args:
        letter: Single letter (e.g., "A" or "a")
        
    Returns:
        Class ID (0-25)
        
    Raises:
        KeyError: If letter is not A-Z
    """
    return LETTER_TO_ID[letter.upper()]


def get_letter(letter_id: int) -> str:
    """
    Get the letter for a given class ID.
    
    Args:
        letter_id: Class ID (0-25)
        
    Returns:
        Letter (e.g., "A")
        
    Raises:
        KeyError: If letter ID is not 0-25
    """
    return ID_TO_LETTER[letter_id]


def get_difficulty(letter: str) -> str:
    """
    Get the difficulty level for a given letter.
    
    Args:
        letter: Single letter
        
    Returns:
        Difficulty level ("Easy", "Medium", "Hard", "Very Hard")
    """
    letter = letter.upper()
    for difficulty, letters in DIFFICULTY_LEVELS.items():
        if letter in letters:
            return difficulty
    return "Unknown"


def get_letters_by_difficulty(difficulty: str) -> list:
    """
    Get all letters at a given difficulty level.
    
    Args:
        difficulty: Difficulty level (e.g., "Easy")
        
    Returns:
        List of letters at that difficulty
    """
    return DIFFICULTY_LEVELS.get(difficulty, [])


def is_valid_letter(letter: str) -> bool:
    """
    Check if a letter is in the vocabulary.
    
    Args:
        letter: Single letter
        
    Returns:
        True if letter is A-Z, False otherwise
    """
    return len(letter) == 1 and letter.upper() in LETTER_TO_ID


def get_all_letters() -> list:
    """
    Get all letters in the vocabulary.
    
    Returns:
        List of all letters (A-Z)
    """
    return LETTERS.copy()


def word_to_ids(word: str) -> list:
    """
    Convert a word to a sequence of letter IDs.
    
    Args:
        word: Word to convert (e.g., "HELLO")
        
    Returns:
        List of letter IDs
        
    Example:
        >>> word_to_ids("HELLO")
        [7, 4, 11, 11, 14]  # H=7, E=4, L=11, O=14
    """
    return [get_letter_id(char) for char in word if char.isalpha()]


def ids_to_word(letter_ids: list) -> str:
    """
    Convert a sequence of letter IDs to a word.
    
    Args:
        letter_ids: List of letter IDs
        
    Returns:
        Reconstructed word
        
    Example:
        >>> ids_to_word([7, 4, 11, 11, 14])
        "HELLO"
    """
    return ''.join(get_letter(lid) for lid in letter_ids)


def print_vocabulary():
    """Print the complete vocabulary in a formatted way."""
    print("=" * 70)
    print("Sign2Sound Phase 2 - ASL Fingerspelling Alphabet")
    print("=" * 70)
    print(f"\nTotal Letters: {NUM_CLASSES}\n")
    
    print("Alphabet Groups:")
    print("-" * 70)
    for group_name, letters in LETTER_GROUPS.items():
        letter_ids = [f"{letter}({LETTER_TO_ID[letter]})" for letter in letters]
        print(f"{group_name:8s}: {', '.join(letter_ids)}")
    
    print("\n\nDifficulty Levels:")
    print("-" * 70)
    for difficulty, letters in DIFFICULTY_LEVELS.items():
        print(f"{difficulty:12s}: {', '.join(letters)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Demo usage
    print_vocabulary()
    
    print("\n\nExample Usage:")
    print("-" * 70)
    
    # Get letter ID
    letter = "A"
    letter_id = get_letter_id(letter)
    print(f"Letter '{letter}' has ID: {letter_id}")
    
    # Get letter from ID
    retrieved_letter = get_letter(letter_id)
    print(f"ID {letter_id} corresponds to: {retrieved_letter}")
    
    # Get difficulty
    difficulty = get_difficulty(letter)
    print(f"Letter '{letter}' difficulty: {difficulty}")
    
    # Convert word to IDs
    word = "HELLO"
    ids = word_to_ids(word)
    print(f"\nWord '{word}' as IDs: {ids}")
    
    # Convert IDs back to word
    reconstructed = ids_to_word(ids)
    print(f"IDs {ids} as word: {reconstructed}")
    
    # Check if letter is valid
    print(f"\nIs 'A' valid? {is_valid_letter('A')}")
    print(f"Is '1' valid? {is_valid_letter('1')}")
    
    # Get easy letters
    easy_letters = get_letters_by_difficulty("Easy")
    print(f"\nEasy letters to start with: {easy_letters}")
