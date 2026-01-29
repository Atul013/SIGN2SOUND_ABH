// Sign2Sound - Main JavaScript

// ============================================
// Smooth Scrolling
// ============================================

function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

function scrollToTraining() {
    document.getElementById('training').scrollIntoView({ behavior: 'smooth' });
}

// ============================================
// Navigation Active State
// ============================================

const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// ============================================
// Alphabet Grid
// ============================================

function initializeAlphabetGrid() {
    const grid = document.getElementById('alphabetGrid');
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    const specialGestures = ['del', 'spc', 'nil'];

    [...letters, ...specialGestures].forEach(letter => {
        const cell = document.createElement('div');
        cell.className = 'letter-cell';
        cell.textContent = letter;
        cell.dataset.letter = letter;
        grid.appendChild(cell);
    });
}

// ============================================
// Webcam & Demo Functionality
// ============================================

let webcamStream = null;
let isRecognizing = false;
let recognizedText = '';

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');

startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            }
        });
        webcam.srcObject = webcamStream;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        isRecognizing = true;

        // Start recognition loop
        recognitionLoop();
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam. Please check permissions.');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        webcamStream = null;
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    isRecognizing = false;
}

async function recognitionLoop() {
    if (!isRecognizing) return;

    // Capture frame and perform real inference
    await captureAndInfer();

    // Continue loop (10 FPS)
    setTimeout(recognitionLoop, 100);
}

async function captureAndInfer() {
    try {
        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const ctx = canvas.getContext('2d');

        // Draw current video frame
        ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Send to API for inference
        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.success && data.hand_detected) {
            // Update prediction display
            updatePrediction(data.prediction, (data.confidence * 100).toFixed(1));

            // Draw landmarks if available
            if (data.landmarks) {
                drawLandmarks(data.landmarks);
            }
        } else {
            // No hand detected
            document.getElementById('predictedLetter').textContent = '-';
            document.getElementById('confidence').textContent = '0%';

            // Clear landmarks
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, overlay.width, overlay.height);
        }
    } catch (error) {
        console.error('Inference error:', error);
    }
}

function updatePrediction(letter, confidence) {
    document.getElementById('predictedLetter').textContent = letter;
    document.getElementById('confidence').textContent = `${confidence}%`;

    // Highlight in alphabet grid
    document.querySelectorAll('.letter-cell').forEach(cell => {
        cell.classList.remove('active');
        if (cell.dataset.letter === letter) {
            cell.classList.add('active');
        }
    });

    // Add to text output (with debouncing)
    if (confidence > 85) {
        addToText(letter);
    }
}

let lastAddedLetter = '';
let lastAddedTime = 0;

function addToText(letter) {
    const now = Date.now();

    // Debounce: only add if different letter or 1 second has passed
    if (letter !== lastAddedLetter || now - lastAddedTime > 1000) {
        // Handle special gestures
        if (letter.toLowerCase() === 'space') {
            // Add actual space instead of the word "space"
            recognizedText += ' ';
        } else if (letter.toLowerCase() === 'del') {
            // Delete last character instead of adding "del"
            if (recognizedText.length > 0) {
                recognizedText = recognizedText.slice(0, -1);
            }
        } else if (letter.toLowerCase() === 'nothing') {
            // Do nothing for "nothing" gesture
            return;
        } else {
            // Add normal letter
            recognizedText += letter;
        }

        document.querySelector('.output-text').textContent = recognizedText;
        lastAddedLetter = letter;
        lastAddedTime = now;
    }
}

function clearText() {
    recognizedText = '';
    document.querySelector('.output-text').textContent = '';
}

async function speakText() {
    if (!recognizedText || recognizedText.trim().length === 0) {
        alert('No text to speak');
        return;
    }

    try {
        // Call grammar correction API
        const response = await fetch('/api/correct_grammar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: recognizedText })
        });

        const data = await response.json();

        if (data.success) {
            const correctedText = data.corrected;

            // Update display with corrected text
            document.querySelector('.output-text').textContent = correctedText;
            recognizedText = correctedText;

            // Speak corrected text
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(correctedText);
                utterance.rate = 0.9;
                utterance.pitch = 1;
                speechSynthesis.speak(utterance);
            }
        } else {
            // Fallback: speak original text
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(recognizedText);
                utterance.rate = 0.9;
                utterance.pitch = 1;
                speechSynthesis.speak(utterance);
            }
        }
    } catch (error) {
        console.error('Grammar correction error:', error);
        // Fallback: speak original text
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(recognizedText);
            utterance.rate = 0.9;
            utterance.pitch = 1;
            speechSynthesis.speak(utterance);
        } else {
            alert('Text-to-speech not supported in this browser.');
        }
    }
}

// ============================================
// Training Progress Simulation
// ============================================

let trainingProgress = 0;
let currentEpoch = 0;
let trainingStartTime = Date.now();

function updateTrainingProgress() {
    // Simulate training progress
    if (trainingProgress < 100) {
        trainingProgress += Math.random() * 2;
        currentEpoch = Math.floor(trainingProgress / 100 * 100);

        // Update UI
        document.getElementById('progressFill').style.width = `${trainingProgress}%`;
        document.getElementById('progressPercent').textContent = `${Math.floor(trainingProgress)}%`;
        document.getElementById('currentEpoch').textContent = `${currentEpoch}/100`;

        // Simulate metrics
        const trainAcc = (Math.random() * 10 + 85).toFixed(2);
        const valAcc = (Math.random() * 10 + 80).toFixed(2);
        const trainLoss = (Math.random() * 0.5 + 0.1).toFixed(4);
        const valLoss = (Math.random() * 0.5 + 0.2).toFixed(4);

        document.getElementById('trainAcc').textContent = `${trainAcc}%`;
        document.getElementById('valAcc').textContent = `${valAcc}%`;
        document.getElementById('trainLoss').textContent = trainLoss;
        document.getElementById('valLoss').textContent = valLoss;

        // Calculate time
        const elapsed = Date.now() - trainingStartTime;
        const elapsedMinutes = Math.floor(elapsed / 60000);
        const elapsedSeconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('timeElapsed').textContent = `${elapsedMinutes}m ${elapsedSeconds}s`;

        // ETA
        const remaining = (100 - trainingProgress) / trainingProgress * elapsed;
        const etaMinutes = Math.floor(remaining / 60000);
        document.getElementById('eta').textContent = `${etaMinutes}m`;

        // Best accuracy
        const bestAcc = Math.max(parseFloat(trainAcc), parseFloat(valAcc)).toFixed(2);
        document.getElementById('bestAcc').textContent = `${bestAcc}%`;

        // Update status
        if (trainingProgress < 10) {
            document.getElementById('trainingStatus').textContent = 'Preprocessing';
        } else if (trainingProgress < 95) {
            document.getElementById('trainingStatus').textContent = 'Training';
        } else {
            document.getElementById('trainingStatus').textContent = 'Finalizing';
        }
    } else {
        document.getElementById('trainingStatus').textContent = 'Complete';
        document.querySelector('.status-dot').style.background = '#22c55e';
    }
}

// Fetch real metrics on load
async function fetchRealMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();

        if (data.history) {
            const history = data.history;
            const epochs = history.train_accuracy.length;
            const finalTrainAcc = (history.train_accuracy[epochs - 1] * 100).toFixed(2);
            const finalValAcc = (history.val_accuracy[epochs - 1] * 100).toFixed(2);
            const finalTrainLoss = history.train_loss[epochs - 1].toFixed(4);
            const finalValLoss = history.val_loss[epochs - 1].toFixed(4);
            const bestValAcc = (Math.max(...history.val_accuracy) * 100).toFixed(2);

            // Update UI with real data
            document.getElementById('trainingStatus').textContent = 'Complete';
            document.getElementById('progressFill').style.width = '100%';
            document.getElementById('progressPercent').textContent = '100%';
            document.getElementById('currentEpoch').textContent = `${epochs}/${epochs}`;
            document.getElementById('trainAcc').textContent = `${finalTrainAcc}%`;
            document.getElementById('valAcc').textContent = `${finalValAcc}%`;
            document.getElementById('trainLoss').textContent = finalTrainLoss;
            document.getElementById('valLoss').textContent = finalValLoss;
            document.getElementById('bestAcc').textContent = `${bestValAcc}%`;
            document.getElementById('eta').textContent = 'Complete';
            document.getElementById('timeElapsed').textContent = '~6 minutes';

            // Update status indicator
            document.querySelector('.status-dot').style.background = '#22c55e';

            // Update stats in hero section
            document.querySelectorAll('.stat-number')[2].textContent = `${bestValAcc}%`;
        }
    } catch (error) {
        console.log('Metrics API not available');
    }
}

// Fetch metrics on load
fetchRealMetrics();

// ============================================
// Real-time Inference (via API)
// ============================================

async function performInference(imageData) {
    try {
        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.success) {
            updatePrediction(data.prediction, data.confidence * 100);
        }
    } catch (error) {
        console.log('Inference API not available, using simulation');
    }
}

// ============================================
// Canvas Drawing for Landmarks
// ============================================

function drawLandmarks(landmarks) {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Set canvas size to match video
    overlay.width = webcam.videoWidth;
    overlay.height = webcam.videoHeight;

    if (!landmarks || landmarks.length === 0) return;

    // Draw connections
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],  // Index
        [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
        [0, 13], [13, 14], [14, 15], [15, 16],  // Ring
        [0, 17], [17, 18], [18, 19], [19, 20],  // Pinky
        [5, 9], [9, 13], [13, 17], [0, 17]  // Palm
    ];

    connections.forEach(([start, end]) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];

        ctx.beginPath();
        ctx.moveTo(startPoint.x * overlay.width, startPoint.y * overlay.height);
        ctx.lineTo(endPoint.x * overlay.width, endPoint.y * overlay.height);
        ctx.stroke();
    });

    // Draw landmarks
    landmarks.forEach(landmark => {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(
            landmark.x * overlay.width,
            landmark.y * overlay.height,
            5,
            0,
            2 * Math.PI
        );
        ctx.fill();
    });
}

// ============================================
// Keyboard Shortcuts
// ============================================

document.addEventListener('keydown', (e) => {
    // Space to start/stop camera
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        if (!isRecognizing) {
            startWebcam();
        } else {
            stopWebcam();
        }
    }

    // Escape to clear text
    if (e.code === 'Escape') {
        clearText();
    }

    // Enter to speak text
    if (e.code === 'Enter' && e.ctrlKey) {
        speakText();
    }
});

// ============================================
// Initialize on Load
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeAlphabetGrid();

    // Set initial training status
    document.getElementById('trainingStatus').textContent = 'Preprocessing';
    document.getElementById('currentEpoch').textContent = '-';
    document.getElementById('trainAcc').textContent = '-';
    document.getElementById('valAcc').textContent = '-';
    document.getElementById('trainLoss').textContent = '-';
    document.getElementById('valLoss').textContent = '-';
    document.getElementById('timeElapsed').textContent = '-';
    document.getElementById('eta').textContent = '-';
    document.getElementById('bestAcc').textContent = '-';

    console.log('Sign2Sound UI initialized');
});

// ============================================
// Utility Functions
// ============================================

function formatTime(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Export for use in other modules
window.Sign2Sound = {
    scrollToDemo,
    scrollToTraining,
    clearText,
    speakText,
    updatePrediction,
    drawLandmarks
};
