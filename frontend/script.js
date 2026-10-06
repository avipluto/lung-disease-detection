/**
 * LungScan AI - Frontend JavaScript
 * Handles file upload, API communication, and results rendering
 */

// Auto-detect API URL for local standalone vs reverse proxy (Docker)
const API_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
    ? 'http://localhost:8000/api'
    : '/api';


const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];

// ============ DOM Elements ============
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('results');
const predictionsContainer = document.getElementById('predictionsContainer');
const resultsFilename = document.getElementById('resultsFilename');
const errorContainer = document.getElementById('errorContainer');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const themeToggle = document.getElementById('themeToggle');

let selectedFile = null;

// ============ Theme Toggle ============
function initTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

themeToggle.addEventListener('click', toggleTheme);
initTheme();

// ============ Drag & Drop ============
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// ============ File Handling ============
function handleFile(file) {
    // Validate file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        showError('Invalid file type. Please upload a JPEG, PNG, BMP, TIFF, or WebP image.');
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showError(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB.`);
        return;
    }

    selectedFile = file;
    showPreview(file);
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        dropzone.style.display = 'none';
        previewContainer.style.display = 'block';
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewContainer.style.display = 'none';
    dropzone.style.display = 'block';
    hideResults();
    hideError();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

removeBtn.addEventListener('click', clearPreview);

// ============ Analysis ============
analyzeBtn.addEventListener('click', analyzeImage);

async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading
    previewContainer.style.display = 'none';
    loadingContainer.style.display = 'block';
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        loadingContainer.style.display = 'none';
        previewContainer.style.display = 'block';
        showResults(data);

    } catch (error) {
        loadingContainer.style.display = 'none';
        previewContainer.style.display = 'block';
        console.error('Prediction error:', error);

        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showError('Cannot connect to the server. Make sure the backend is running on port 8000.');
        } else {
            showError(`Analysis failed: ${error.message}`);
        }
    }
}

// ============ Results Rendering ============
function showResults(data) {
    resultsFilename.textContent = `Results for: ${data.filename}`;
    predictionsContainer.innerHTML = '';

    data.predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 0.1}s`;

        const confidenceClass = getConfidenceClass(pred.confidence);
        const barClass = getBarClass(pred.confidence);

        item.innerHTML = `
            <div class="prediction-header">
                <span class="prediction-disease">${pred.disease}</span>
                <span class="prediction-confidence ${confidenceClass}">${pred.confidence.toFixed(1)}%</span>
            </div>
            <div class="prediction-bar-bg">
                <div class="prediction-bar ${barClass}" style="width: 0%" data-width="${pred.confidence}"></div>
            </div>
            <p class="prediction-description">${pred.description}</p>
        `;

        predictionsContainer.appendChild(item);
    });

    resultsSection.style.display = 'block';

    // Animate bars after render
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.prediction-bar').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        }, 100);
    });

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function getConfidenceClass(confidence) {
    if (confidence < 30) return 'confidence-low';
    if (confidence < 60) return 'confidence-medium';
    return 'confidence-high';
}

function getBarClass(confidence) {
    if (confidence < 30) return 'bar-low';
    if (confidence < 60) return 'bar-medium';
    return 'bar-high';
}

// ============ Error Handling ============
function showError(message) {
    errorMessage.textContent = message;
    errorContainer.style.display = 'block';
}

function hideError() {
    errorContainer.style.display = 'none';
}

retryBtn.addEventListener('click', () => {
    hideError();
    if (selectedFile) {
        previewContainer.style.display = 'block';
    } else {
        dropzone.style.display = 'block';
    }
});

// ============ Keyboard Shortcuts ============
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + O to open file
    if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault();
        fileInput.click();
    }
    // Enter to analyze
    if (e.key === 'Enter' && selectedFile && previewContainer.style.display !== 'none') {
        analyzeImage();
    }
    // Escape to clear
    if (e.key === 'Escape') {
        clearPreview();
    }
});
