// ============================================
// Intelligent Word Prediction & Grammar Correction
// ============================================

// Common English words for prediction
const commonWords = {
    'i': ['i', 'am', 'is', 'it', 'in'],
    'h': ['hello', 'hi', 'have', 'has', 'how', 'he', 'her', 'his', 'help'],
    'w': ['what', 'when', 'where', 'why', 'who', 'will', 'was', 'were', 'we'],
    't': ['the', 'this', 'that', 'they', 'them', 'their', 'to', 'too'],
    'a': ['a', 'an', 'and', 'are', 'am', 'as', 'at', 'all'],
    'y': ['you', 'your', 'yes', 'yet'],
    'c': ['can', 'could', 'come', 'call'],
    'g': ['go', 'good', 'get', 'got'],
    'n': ['no', 'not', 'now', 'need'],
    's': ['see', 'so', 'she', 'said', 'say'],
    'b': ['be', 'but', 'by', 'been'],
    'd': ['do', 'did', 'does', 'done'],
    'f': ['for', 'from', 'feel'],
    'l': ['like', 'love', 'look'],
    'm': ['me', 'my', 'make', 'more'],
    'p': ['please', 'put'],
    'r': ['right', 'really'],
    'o': ['of', 'on', 'or', 'ok', 'one'],
    'k': ['know', 'keep'],
    'j': ['just'],
    'v': ['very'],
};

// Word frequency for better predictions
const wordFrequency = {
    'the': 100, 'i': 95, 'you': 90, 'a': 85, 'to': 80,
    'is': 75, 'it': 70, 'that': 65, 'and': 60, 'of': 55,
    'in': 50, 'have': 45, 'for': 40, 'not': 35, 'on': 30,
    'with': 25, 'he': 20, 'as': 15, 'do': 10, 'at': 5
};

// Grammar rules for sentence correction
const grammarRules = {
    // Article corrections
    articles: {
        'a': ['a', 'an'],
        'an': ['an', 'a']
    },
    // Common contractions
    contractions: {
        'i am': "i'm",
        'you are': "you're",
        'he is': "he's",
        'she is': "she's",
        'it is': "it's",
        'we are': "we're",
        'they are': "they're",
        'i have': "i've",
        'you have': "you've",
        'we have': "we've",
        'they have': "they've",
        'i will': "i'll",
        'you will': "you'll",
        'he will': "he'll",
        'she will': "she'll",
        'we will': "we'll",
        'they will': "they'll",
        'do not': "don't",
        'does not': "doesn't",
        'did not': "didn't",
        'is not': "isn't",
        'are not': "aren't",
        'was not': "wasn't",
        'were not': "weren't",
        'have not': "haven't",
        'has not': "hasn't",
        'had not': "hadn't",
        'will not': "won't",
        'would not': "wouldn't",
        'should not': "shouldn't",
        'could not': "couldn't",
        'cannot': "can't"
    }
};

// Current word being typed
let currentWord = '';
let lastSpokenWord = '';
let lastSpeakTime = 0;

// Word prediction
function predictWord(partialWord) {
    if (!partialWord || partialWord.length === 0) return [];

    const firstLetter = partialWord[0].toLowerCase();
    const candidates = commonWords[firstLetter] || [];

    // Filter words that start with the partial word
    const matches = candidates.filter(word =>
        word.toLowerCase().startsWith(partialWord.toLowerCase())
    );

    // Sort by frequency
    matches.sort((a, b) => {
        const freqA = wordFrequency[a.toLowerCase()] || 0;
        const freqB = wordFrequency[b.toLowerCase()] || 0;
        return freqB - freqA;
    });

    return matches.slice(0, 3); // Top 3 predictions
}

// Complete word when space is detected
function completeWord(partialWord) {
    const predictions = predictWord(partialWord);
    if (predictions.length > 0) {
        return predictions[0]; // Return best prediction
    }
    return partialWord; // Return as-is if no prediction
}

// Grammar correction
function correctGrammar(sentence) {
    if (!sentence) return '';

    let corrected = sentence.toLowerCase().trim();

    // Apply contractions
    for (const [full, contraction] of Object.entries(grammarRules.contractions)) {
        const regex = new RegExp('\\b' + full + '\\b', 'gi');
        corrected = corrected.replace(regex, contraction);
    }

    // Capitalize first letter
    corrected = corrected.charAt(0).toUpperCase() + corrected.slice(1);

    // Capitalize 'I'
    corrected = corrected.replace(/\bi\b/g, 'I');

    // Add period if missing
    if (corrected.length > 0 && !corrected.match(/[.!?]$/)) {
        corrected += '.';
    }

    return corrected;
}

// Speak word naturally (not letter by letter)
function speakWord(word, force = false) {
    if (!word || word.trim().length === 0) return;

    const now = Date.now();

    // Debounce: don't speak same word within 2 seconds
    if (!force && word === lastSpokenWord && now - lastSpeakTime < 2000) {
        return;
    }

    if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(word);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        speechSynthesis.speak(utterance);

        lastSpokenWord = word;
        lastSpeakTime = now;
    }
}

// Speak entire sentence naturally
function speakSentence(sentence, force = false) {
    if (!sentence || sentence.trim().length === 0) return;

    const corrected = correctGrammar(sentence);

    if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(corrected);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        speechSynthesis.speak(utterance);
    }
}

// Process recognized text with word boundaries
function processRecognizedText(text) {
    const words = text.trim().split(/\s+/);

    if (words.length === 0) return text;

    // Get the last word (currently being typed)
    const lastWord = words[words.length - 1];

    // If last word is complete (followed by space in original text)
    if (text.endsWith(' ') && lastWord.length > 0) {
        // Complete and speak the word
        const completed = completeWord(lastWord);
        speakWord(completed);

        // Replace last word with completed version
        words[words.length - 1] = completed;
    }

    return words.join(' ');
}

// Auto-correct and speak on sentence completion
function finalizeSentence(text) {
    const corrected = correctGrammar(text);
    speakSentence(corrected, true);
    return corrected;
}

// Export functions
window.NLP = {
    predictWord,
    completeWord,
    correctGrammar,
    speakWord,
    speakSentence,
    processRecognizedText,
    finalizeSentence
};
