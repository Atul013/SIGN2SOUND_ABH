
// ============================================
// ASL Sign Display & Word Converter
// ============================================

let currentWord = '';
let currentLetterIndex = 0;
let wordLetters = [];

function showSign(letter) {
    const signImage = document.getElementById('signImage');
    const signLabel = document.getElementById('signLabel');
    const imageMap = { 'spc': 'space', 'nil': 'nothing' };
    const imageName = imageMap[letter] || letter;
    signImage.src = `static/images/asl_alphabet/${imageName}.jpg`;
    signLabel.textContent = letter.toUpperCase();
}

function convertWord() {
    const input = document.getElementById('wordInput');
    const word = input.value.toUpperCase().trim();
    if (!word) { alert('Please enter a word'); return; }
    currentWord = word;
    wordLetters = word.split('');
    currentLetterIndex = 0;
    document.getElementById('prevBtn').disabled = false;
    document.getElementById('nextBtn').disabled = false;
    document.getElementById('stopConverterBtn').disabled = false;
    showWordLetter();
}

function showWordLetter() {
    if (wordLetters.length === 0) return;
    const letter = wordLetters[currentLetterIndex];
    showSign(letter);
    document.getElementById('letterIndicator').textContent = `${currentLetterIndex + 1}/${wordLetters.length}: ${letter}`;
    document.getElementById('prevBtn').disabled = currentLetterIndex === 0;
    document.getElementById('nextBtn').disabled = currentLetterIndex === wordLetters.length - 1;
}

function showPrevSign() { if (currentLetterIndex > 0) { currentLetterIndex--; showWordLetter(); } }
function showNextSign() { if (currentLetterIndex < wordLetters.length - 1) { currentLetterIndex++; showWordLetter(); } }

function stopConverter() {
    currentWord = '';
    wordLetters = [];
    currentLetterIndex = 0;
    document.getElementById('wordInput').value = '';
    document.getElementById('letterIndicator').textContent = '-';
    document.getElementById('prevBtn').disabled = true;
    document.getElementById('nextBtn').disabled = true;
    document.getElementById('stopConverterBtn').disabled = true;
    showSign('A');
}

window.convertWord = convertWord;
window.showPrevSign = showPrevSign;
window.showNextSign = showNextSign;
window.stopConverter = stopConverter;

// Update alphabet grid to be clickable
const originalInitGrid = initializeAlphabetGrid;
initializeAlphabetGrid = function () {
    const grid = document.getElementById('alphabetGrid');
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    const specialGestures = ['del', 'spc', 'nil'];
    [...letters, ...specialGestures].forEach(letter => {
        const cell = document.createElement('div');
        cell.className = 'letter-cell';
        cell.textContent = letter;
        cell.dataset.letter = letter;
        cell.addEventListener('click', () => {
            showSign(letter);
            document.querySelectorAll('.letter-cell').forEach(c => c.classList.remove('active'));
            cell.classList.add('active');
        });
        grid.appendChild(cell);
    });
};

// Auto-show sign when detected
const origUpdate = updatePrediction;
updatePrediction = function (letter, confidence) {
    origUpdate(letter, confidence);
    if (confidence > 70) showSign(letter);
};
