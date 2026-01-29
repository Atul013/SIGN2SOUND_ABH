
// ============================================
// Word to Sign Converter (without sign display)
// ============================================

let currentWord = '';
let currentLetterIndex = 0;
let wordLetters = [];

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
}

window.convertWord = convertWord;
window.showPrevSign = showPrevSign;
window.showNextSign = showNextSign;
window.stopConverter = stopConverter;

// Update alphabet grid to be clickable (without showing sign image)
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
            // Just highlight the cell, no sign display
            document.querySelectorAll('.letter-cell').forEach(c => c.classList.remove('active'));
            cell.classList.add('active');
        });
        grid.appendChild(cell);
    });
};
