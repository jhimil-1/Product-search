// Main JavaScript file for AI E-commerce Chatbot

// Global variables
let currentSessionId = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Check for existing session
    currentSessionId = localStorage.getItem('sessionId');
    if (currentSessionId) {
        updateUIForLoggedInUser();
    }
    
    // Initialize event listeners
    initializeEventListeners();
});

// Initialize all event listeners
function initializeEventListeners() {
    // File upload handler
    const fileInput = document.getElementById('imageUpload');
    if (fileInput) {
        fileInput.addEventListener('change', handleImageSelection);
    }
    
    // Drag and drop area
    const dropArea = document.getElementById('drop-area');
    if (dropArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        dropArea.addEventListener('drop', handleDrop, false);
    }
}

// Prevent default drag and drop behavior
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area
function highlight() {
    const dropArea = document.getElementById('drop-area');
    if (dropArea) dropArea.classList.add('highlight');
}

// Remove highlight from drop area
function unhighlight() {
    const dropArea = document.getElementById('drop-area');
    if (dropArea) dropArea.classList.remove('highlight');
}

// Handle dropped files
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// Handle selected files
function handleImageSelection(e) {
    const files = e.target.files;
    handleFiles(files);
}

// Process uploaded files
function handleFiles(files) {
    if (!files || files.length === 0) return;
    
    const file = files[0];
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file.');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('Image file is too large. Please upload an image smaller than 10MB.');
        return;
    }
    
    // Get current session ID
    const sessionId = getCurrentSessionId();
    if (!sessionId) {
        showError('Please create a session first.');
        return;
    }
    
    // Upload the image
    uploadImage(file, sessionId);
}

// Upload image to server
async function uploadImage(file, sessionId) {
    if (!file) return;
    
    // Create FormData and include additional context
    const formData = new FormData();
    formData.append('file', file);
    
    // Extract context from filename for better category detection
    const filename = file.name || '';
    const contextFromFilename = extractCategoryFromFilename(filename);
    
    // Add context to help with category detection
    if (contextFromFilename) {
        formData.append('context', contextFromFilename);
    }
    
    // Add the original filename for server-side processing
    formData.append('original_filename', filename);
    
    try {
        // Show loading state
        showLoadingMessage('Analyzing image and finding similar products...');
        
        const response = await fetch(`/query_image/${sessionId}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data.results || [], data.message || 'Here are some visually similar products:');
            
            // Log detected categories for debugging (remove in production)
            if (data.detected_categories && data.detected_categories.length > 0) {
                console.log('Detected categories:', data.detected_categories);
            }
        } else {
            showError(data.error || 'Failed to process image');
        }
        
    } catch (error) {
        console.error('Error uploading image:', error);
        showError('Failed to upload image. Please try again.');
    }
}

// Extract category from filename
function extractCategoryFromFilename(filename) {
    if (!filename) return '';
    
    const lowerFilename = filename.toLowerCase();
    
    // Common jewelry category patterns in filenames
    const categoryPatterns = {
        'necklace': ['necklace', 'chain', 'pendant', 'choker', 'locket'],
        'ring': ['ring', 'band', 'engagement', 'wedding', 'solitaire'],
        'bracelet': ['bracelet', 'bangle', 'cuff', 'wristband'],
        'earring': ['earring', 'stud', 'hoop', 'dangle', 'drop'],
        'watch': ['watch', 'timepiece']
    };
    
    for (const [category, patterns] of Object.entries(categoryPatterns)) {
        for (const pattern of patterns) {
            if (lowerFilename.includes(pattern)) {
                return `${category} jewelry`; // Add "jewelry" for better context
            }
        }
    }
    
    return 'jewelry'; // Default context
}

// Show loading message
function showLoadingMessage(message) {
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="loading-message">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
    }
}

// Show error message
function showError(message) {
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="error-message">
                <p style="color: red;">Error: ${message}</p>
            </div>
        `;
    }
}

// Get current session ID
function getCurrentSessionId() {
    return currentSessionId || localStorage.getItem('sessionId');
}

// Display search results
function displayResults(results, message = '') {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-results">
                <p>No similar products found. Try uploading a different image or search by text.</p>
            </div>
        `;
        return;
    }
    
    let html = `<div class="results-header"><h3>${message}</h3></div>`;
    html += '<div class="products-grid">';
    
    results.forEach(product => {
        html += `
            <div class="product-card">
                ${product.image_url ? 
                    `<img src="${product.image_url}" alt="${product.name}" class="product-image" loading="lazy">` : 
                    '<div class="no-image-placeholder">No Image</div>'
                }
                <div class="product-info">
                    <h4 class="product-name">${product.name || 'Unnamed Product'}</h4>
                    <p class="product-price"><strong>${product.price || 'Price not available'}</strong></p>
                    ${product.description ? `<p class="product-description">${product.description}</p>` : ''}
                    <div class="product-meta">
                        ${product.category ? `<span class="product-category">Category: ${product.category}</span>` : ''}
                        ${product.score ? `<span class="similarity-score">Similarity: ${product.score}%</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    resultsContainer.innerHTML = html;
}

// Update UI for logged in user
function updateUIForLoggedInUser() {
    // Update UI elements to show user is logged in
    const loginBtn = document.getElementById('loginBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    
    if (loginBtn) loginBtn.style.display = 'none';
    if (logoutBtn) logoutBtn.style.display = 'inline-block';
}

// Logout function
function logout() {
    // Clear session data
    localStorage.removeItem('sessionId');
    currentSessionId = null;
    
    // Update UI
    const loginBtn = document.getElementById('loginBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    
    if (loginBtn) loginBtn.style.display = 'inline-block';
    if (logoutBtn) logoutBtn.style.display = 'none';
    
    // Redirect to login page
    window.location.href = '/login.html';
}

// Export functions that need to be accessible from HTML
window.uploadImage = uploadImage;
window.logout = logout;