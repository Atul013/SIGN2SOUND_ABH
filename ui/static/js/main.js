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

    // Capture frame from webcam
    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg');

    // Perform real inference
    await performInference(imageData);

    // Continue loop (every 500ms for real-time feel)
    setTimeout(recognitionLoop, 500);
}

// (Simulation removed)

function updatePrediction(letter, confidence) {
    // Determine display text
    let displayLetter = letter;
    if (letter === 'del') displayLetter = '⌫';
    if (letter === 'space' || letter === 'spc') displayLetter = '␣';
    if (letter === 'nothing' || letter === 'nil') displayLetter = '-';

    document.getElementById('predictedLetter').textContent = displayLetter;
    document.getElementById('confidence').textContent = `${confidence}%`;

    // Highlight in alphabet grid
    document.querySelectorAll('.letter-cell').forEach(cell => {
        cell.classList.remove('active');
        if (cell.dataset.letter === letter) {
            cell.classList.add('active');
        }
    });

    // Add to text output (with debouncing)
    // Pass raw 'letter' so addToText handles the logic (del/space) correctly
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
        if (letter === 'del') {
            recognizedText = recognizedText.slice(0, -1);
        } else if (letter === 'space' || letter === 'spc') {
            recognizedText += ' ';
        } else if (letter === 'nothing' || letter === 'nil') {
            // Do nothing
        } else {
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

function speakText() {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(recognizedText);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
    } else {
        alert('Text-to-speech not supported in this browser.');
    }
}

// (Training simulation removed)

// ============================================
// Real Training Status (via API)
// ============================================

async function fetchTrainingStatus() {
    try {
        const response = await fetch('/api/training/status');
        const data = await response.json();

        // Update status
        document.getElementById('trainingStatus').textContent =
            data.status === 'completed' ? 'Complete' :
                data.status === 'running' ? 'Training' : 'Not Started';

        // Update progress
        const progress = data.status === 'completed' ? 100 :
            (data.current_epoch / data.total_epochs) * 100;
        document.getElementById('progressFill').style.width = `${progress}%`;
        document.getElementById('progressPercent').textContent = `${Math.floor(progress)}%`;

        // Update epoch
        document.getElementById('currentEpoch').textContent =
            `${data.current_epoch}/${data.total_epochs}`;

        // Update metrics
        document.getElementById('trainAcc').textContent = `${data.train_accuracy.toFixed(2)}%`;
        document.getElementById('valAcc').textContent = `${data.val_accuracy.toFixed(2)}%`;
        document.getElementById('trainLoss').textContent = data.train_loss.toFixed(4);
        document.getElementById('valLoss').textContent = data.val_loss.toFixed(4);
        document.getElementById('timeElapsed').textContent = data.time_elapsed;
        document.getElementById('eta').textContent = data.eta;
        document.getElementById('bestAcc').textContent = `${data.best_accuracy.toFixed(2)}%`;

        // Update status dot color
        const statusDot = document.querySelector('.status-dot');
        if (data.status === 'completed') {
            statusDot.style.background = '#22c55e';  // Green
        } else if (data.status === 'running') {
            statusDot.style.background = '#3b82f6';  // Blue
        } else {
            statusDot.style.background = '#6b7280';  // Gray
        }

        // Update stats in hero section
        document.querySelector('.stat-card:nth-child(2) .stat-value').textContent =
            formatNumber(data.train_samples + data.val_samples);

    } catch (error) {
        console.log('Training API not available:', error);
    }
}

// Fetch training status immediately and every 5 seconds
fetchTrainingStatus();
setInterval(fetchTrainingStatus, 5000);

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
            updatePrediction(data.prediction, (data.confidence * 100).toFixed(1));

            // Draw landmarks if available
            if (data.landmarks) {
                drawLandmarks(data.landmarks.map(([x, y, z]) => ({ x, y, z })));
            } else {
                clearLandmarks();
            }
        } else {
            // No hand detected - clear output
            clearPrediction();
        }
    } catch (error) {
        // console.log('Inference API error:', error);
        clearPrediction();
    }
}

function clearPrediction() {
    document.getElementById('predictedLetter').textContent = '-';
    document.getElementById('confidence').textContent = '-';
    document.querySelectorAll('.letter-cell').forEach(cell => cell.classList.remove('active'));
    clearLandmarks();
}

function clearLandmarks() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

// ============================================
// Canvas Drawing for Landmarks
// ============================================


function drawLandmarks(landmarks) {
    const ctx = overlay.getContext('2d');

    // Ensure canvas dimensions match video
    if (webcam.videoWidth && webcam.videoHeight) {
        if (overlay.width !== webcam.videoWidth || overlay.height !== webcam.videoHeight) {
            overlay.width = webcam.videoWidth;
            overlay.height = webcam.videoHeight;
        }
    }

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!landmarks || landmarks.length === 0) return;

    // Draw connections
    ctx.strokeStyle = '#00ff00';  // Bright green
    ctx.lineWidth = 3;  // Thicker lines

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],  // Index
        [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
        [0, 13], [13, 14], [14, 15], [15, 16],  // Ring
        [0, 17], [17, 18], [18, 19], [19, 20],  // Pinky
        [5, 9], [9, 13], [13, 17], [0, 17]  // Palm
    ];

    connections.forEach(([start, end]) => {

        ctx.beginPath();
        ctx.moveTo(landmarks[start].x * overlay.width, landmarks[start].y * overlay.height);
        ctx.lineTo(landmarks[end].x * overlay.width, landmarks[end].y * overlay.height);
        ctx.stroke();
    });

    // Draw landmarks
    landmarks.forEach(landmark => {
        ctx.fillStyle = '#ff0000'; // Red dots
        ctx.beginPath();
        ctx.arc(
            landmark.x * overlay.width,
            landmark.y * overlay.height,
            4,
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
