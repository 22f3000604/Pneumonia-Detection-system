const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadCard = document.querySelector('.upload-card');
const resultCard = document.getElementById('result-card');
const previewImg = document.getElementById('preview-img');
const scannerBar = document.querySelector('.scanner-bar');
const predictionLabel = document.getElementById('prediction-label');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const diagnosisText = document.getElementById('diagnosis-text');

// Handle Drop Zone clicks
dropZone.addEventListener('click', () => fileInput.click());

// Handle Drag and Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

async function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
        startAnalysis(file);
    };
    reader.readAsDataURL(file);
}

async function startAnalysis(file) {
    scannerBar.style.display = 'block';
    predictionLabel.innerText = 'Analyzing...';
    predictionLabel.style.color = 'var(--text-main)';
    confidenceBar.style.width = '0%';
    confidenceText.innerText = 'Processing pixels...';
    diagnosisText.innerText = 'The AI is currently processing the image patterns to identify potential signs of pneumonia. Please wait...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Artificial delay for the "WOW" scanner effect
        setTimeout(() => {
            showResult(data);
        }, 2000);

    } catch (error) {
        console.error('Error:', error);
        predictionLabel.innerText = 'Error';
        diagnosisText.innerText = 'Could not connect to the AI server. Please make sure the backend is running.';
        scannerBar.style.display = 'none';
    }
}

function showResult(data) {
    scannerBar.style.display = 'none';

    const isPneumonia = data.prediction === 'PNEUMONIA';
    predictionLabel.innerText = data.prediction;
    predictionLabel.style.color = isPneumonia ? 'var(--danger)' : 'var(--success)';

    const confPercent = (data.confidence * 100).toFixed(1);
    confidenceBar.style.width = `${confPercent}%`;
    confidenceBar.style.backgroundColor = isPneumonia ? 'var(--danger)' : 'var(--success)';
    confidenceText.innerText = `Confidence: ${confPercent}%`;

    if (isPneumonia) {
        diagnosisText.innerText = `ALERT: The AI has detected features consistent with Pneumonia with ${confPercent}% confidence. This case should be prioritized for urgent medical review.`;
        document.querySelector('.diagnosis-report').style.borderColor = 'var(--danger)';
    } else {
        diagnosisText.innerText = `NORMAL: The AI has detected healthy lung patterns with ${confPercent}% confidence. No significant signs of pneumonia were found in this scan.`;
        document.querySelector('.diagnosis-report').style.borderColor = 'var(--success)';
    }
}

function resetApp() {
    uploadCard.classList.remove('hidden');
    resultCard.classList.add('hidden');
    fileInput.value = '';
    previewImg.src = '';
}

// Check Health on load
fetch('/health').then(r => r.json()).then(data => {
    const dot = document.querySelector('.status-dot');
    if (!data.model_loaded) {
        dot.style.backgroundColor = 'var(--danger)';
        document.getElementById('api-status').innerText = 'Model Missing';
    }
}).catch(() => {
    document.querySelector('.status-dot').style.backgroundColor = 'var(--danger)';
    document.getElementById('api-status').innerText = 'Offline';
});
