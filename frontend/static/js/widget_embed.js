document.addEventListener('DOMContentLoaded', function() {
    // Create widget container if it doesn't exist
    let widgetContainer = document.getElementById('chatbot-widget');
    if (!widgetContainer) {
        widgetContainer = document.createElement('div');
        widgetContainer.id = 'chatbot-widget';
        document.body.appendChild(widgetContainer);
    }

    // Widget styles
    const widgetStyles = `
        #chatbot-widget {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }
        
        .chat-header {
            background-color: ${widgetContainer.dataset.primaryColor || '#4a6fa5'};
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
        }
        
        .message {
            margin-bottom: 10px;
            max-width: 80%;
            padding: 8px 12px;
            border-radius: 15px;
            line-height: 1.4;
        }
        
        .bot-message {
            background-color: #f0f2f5;
            border-bottom-left-radius: 5px;
            margin-right: auto;
        }
        
        .user-message {
            background-color: ${widgetContainer.dataset.primaryColor || '#4a6fa5'};
            color: white;
            border-bottom-right-radius: 5px;
            margin-left: auto;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #e9ecef;
            display: flex;
        }
        
        .chat-input .input-group {
            display: flex;
            flex: 1;
        }
        
        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 20px 0 0 20px;
            border-right: none;
            outline: none;
        }
        
        .chat-input .file-upload {
            position: relative;
            display: inline-block;
        }
        
        .chat-input .file-upload input[type="file"] {
            display: none;
        }
        
        .chat-input .file-upload-label {
            background: #f0f2f5;
            color: #4a6fa5;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-left: none;
            border-radius: 0 20px 20px 0;
            cursor: pointer;
            display: flex;
            align-items: center;
        }
        
        .chat-input .file-upload-label:hover {
            background: #e2e6ea;
        }
        
        .chat-input button {
            background: ${widgetContainer.dataset.primaryColor || '#4a6fa5'};
            color: white;
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            margin-left: 10px;
            cursor: pointer;
        }
        
        .product-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }
        
        .product-card img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
    `;

    // Add styles to head
    const styleElement = document.createElement('style');
    styleElement.textContent = widgetStyles;
    document.head.appendChild(styleElement);

    // Widget HTML
    widgetContainer.innerHTML = `
        <div class="chat-header">
            <h3 style="margin: 0;">Jewelry Assistant</h3>
            <button id="minimize-chat" style="background: none; border: none; color: white; cursor: pointer; font-size: 20px;">−</button>
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="message bot-message">${widgetContainer.dataset.greeting || 'Hello! How can I help you today?'}</div>
        </div>
        <div class="chat-input">
            <div class="input-group">
                <input type="text" id="user-input" placeholder="Type your message or upload an image...">
                <div class="file-upload">
                    <label for="image-upload" class="file-upload-label" title="Upload Image to Search">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                    </label>
                    <input type="file" id="image-upload" accept="image/*" style="display: none;">
                </div>
            </div>
            <button id="send-message" title="Send Message">→</button>
        </div>
    `;

    // Chat functionality
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-message');
    const imageUpload = document.getElementById('image-upload');
    const minimizeButton = document.getElementById('minimize-chat');
    let isMinimized = false;
    
    // Get API URL from configuration or use current origin
    const apiUrl = widgetContainer.dataset.apiUrl || window.location.origin;
    
    // Get API key from widget container
    const apiKey = widgetContainer.dataset.apiKey;
    console.log('Widget API Key:', apiKey);
    
    // Generate a unique session ID for the chat
    let sessionId = localStorage.getItem('chatbot_session_id');
    
    // Function to create a widget session
    async function createWidgetSession() {
        try {
            const response = await fetch(`${apiUrl}/api/v1/widget/create-session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    api_key: apiKey
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to create session: ${response.status}`);
            }
            
            const data = await response.json();
            sessionId = data.session_id;
            localStorage.setItem('chatbot_session_id', sessionId);
            localStorage.setItem('chatbot_session_data', JSON.stringify({
                sessionId: sessionId,
                createdAt: Date.now(),
                collectionName: data.collection_name
            }));
            console.log('Created widget session:', sessionId);
            return sessionId;
        } catch (error) {
            console.error('Error creating widget session:', error);
            // Fallback to generating a local session ID
            sessionId = 'widget-' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('chatbot_session_id', sessionId);
            return sessionId;
        }
    }
    
    // Create session if not exists
    if (!sessionId) {
        createWidgetSession();
    } else {
        // Check if session is old (older than 24 hours) and create new one
        const sessionData = localStorage.getItem('chatbot_session_data');
        if (sessionData) {
            try {
                const data = JSON.parse(sessionData);
                const sessionAge = Date.now() - data.createdAt;
                if (sessionAge > 24 * 60 * 60 * 1000) { // 24 hours
                    console.log('Session is older than 24 hours, creating new session');
                    createWidgetSession();
                }
            } catch (e) {
                console.log('Invalid session data, creating new session');
                createWidgetSession();
            }
        }
    }
    
    // Function to show loading indicator
    function showLoading() {
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'message bot-message';
        loadingMsg.id = 'typing-indicator';
        loadingMsg.innerHTML = 'Searching for similar items...';
        chatMessages.appendChild(loadingMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return loadingMsg;
    }
    
    // Function to remove loading indicator
    function removeLoading(loadingElement) {
        if (loadingElement && loadingElement.parentNode) {
            loadingElement.remove();
        }
    }
    
    // Function to handle image upload and search
    async function handleImageUpload(file) {
        // Get or create a session ID if not provided
        let sessionId = localStorage.getItem('chatbot_session_id');
        if (!sessionId) {
            // Create a new session if one doesn't exist
            sessionId = await createWidgetSession();
        }
        if (!file) {
            console.error('No file provided');
            addMessage('Please select an image file to upload.');
            return;
        }

        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            console.error('Invalid file type:', file.type);
            addMessage('Please upload a valid image file (JPEG, PNG, GIF, or WebP).');
            return;
        }

        // Validate file size (max 5MB)
        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
            console.error('File too large:', file.size, 'bytes');
            addMessage('Image size is too large. Please upload an image smaller than 5MB.');
            return;
        }
        
        // Show the uploaded image in chat
        const reader = new FileReader();
        reader.onload = function(e) {
            const imgMsg = document.createElement('div');
            imgMsg.className = 'message user-message';
            imgMsg.style.textAlign = 'right';
            imgMsg.innerHTML = `
                <div style="display: inline-block; max-width: 200px; margin-bottom: 5px;">
                    <img src="${e.target.result}" style="max-width: 100%; border-radius: 10px;" alt="Uploaded image">
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">
                        ${file.name} (${(file.size / 1024).toFixed(1)} KB)
                    </div>
                </div>
            `;
            chatMessages.appendChild(imgMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };
        reader.onerror = function() {
            console.error('Error reading file:', reader.error);
            addMessage('Error reading the image file. Please try another image.');
            return;
        };
        reader.readAsDataURL(file);
        
        // Show loading indicator
        const loadingElement = showLoading();
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('file', file); // Changed from 'image' to 'file' to match server expectation
            
            // Get API key from widget container
            const apiKey = widgetContainer.dataset.apiKey;
            console.log('Widget API Key:', apiKey);
            if (!apiKey) {
                const errorMsg = 'API key is missing from widget container';
                console.error(errorMsg);
                throw new Error(errorMsg);
            }
            
            // Show a message that we're processing the image
            const processingMsg = document.createElement('div');
            processingMsg.className = 'message bot-message';
            processingMsg.textContent = 'Analyzing your image...';
            chatMessages.appendChild(processingMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Send image to server for processing
            let response;
            let responseData;
            
            try {
                // Show a more specific loading message
                processingMsg.innerHTML = 'Analyzing your image... This may take a moment.';
                
                // Create the URL with session ID
                const fullApiUrl = `${apiUrl}/api/v1/widget/query-image/${sessionId}`;
                console.log('Making request to:', fullApiUrl);
                
                // Create headers with API key
                const headers = new Headers();
                headers.append('X-API-Key', apiKey);
                
                // Log the request details
                console.log('Request method:', 'POST');
                console.log('Request headers:', Object.fromEntries(headers));
                console.log('Form data entries:');
                for (let pair of formData.entries()) {
                    console.log(pair[0] + ': ', pair[1]);
                }
                
                // Make the request
                try {
                    response = await fetch(fullApiUrl, {
                        method: 'POST',
                        headers: headers,
                        body: formData,
                        credentials: 'same-origin',
                        mode: 'cors' // Ensure CORS mode is enabled
                    });
                    
                    console.log('Response status:', response.status, response.statusText);
                    console.log('Response headers:', Object.fromEntries(response.headers.entries()));
                } catch (error) {
                    console.error('Fetch error:', error);
                    throw new Error(`Network error: ${error.message}`);
                }
                
                // Store the response text immediately since we can only read it once
                const responseText = await response.text();
                
                // Log raw response for debugging (remove in production)
                console.log('Raw response:', responseText.substring(0, 200) + (responseText.length > 200 ? '...' : ''));

                // Remove the processing message
                if (processingMsg.parentNode === chatMessages) {
                    chatMessages.removeChild(processingMsg);
                }

                // Try to parse the response as JSON
                try {
                    responseData = responseText ? JSON.parse(responseText) : {};
                    console.log('Parsed response data:', responseData);
                } catch (e) {
                    console.error('Failed to parse response:', responseText);
                    throw new Error('Received an invalid response from the server. Please try again.');
                }

                if (!response.ok) {
                    let errorMessage = 'An error occurred while processing your request.';
                    
                    // Handle specific error codes with user-friendly messages
                    if (response.status === 400) {
                        errorMessage = responseData.error || 'Invalid request. Please check the file and try again.';
                    } else if (response.status === 401) {
                        errorMessage = 'Authentication failed. Please refresh the page and try again.';
                    } else if (response.status === 413) {
                        errorMessage = 'The image is too large. Please upload an image smaller than 10MB.';
                    } else if (response.status >= 500) {
                        errorMessage = 'A server error occurred. Please try again later.';
                    } else {
                        errorMessage = responseData.error || 
                                     responseData.message || 
                                     response.statusText ||
                                     `Server error (${response.status})`;
                    }
                    
                    console.error('Server error:', response.status, errorMessage, responseData);
                    throw new Error(errorMessage);
                }
                
                // If we got here, the request was successful
                const data = responseData;
                removeLoading(loadingElement);
                
                // Display search results
                if (data.results && data.results.length > 0) {
                    const resultsHtml = data.results.slice(0, 3).map(product => {
                        const imageUrl = product.image_url || 'https://via.placeholder.com/200x200?text=No+Image';
                        const productName = product.name || 'Unnamed Product';
                        const productDesc = product.description || 'No description available';
                        const productPrice = product.price ? `$${product.price}` : 'Price not available';
                        const safeProductName = productName.replace(/'/g, "\\'");
                        const s = safeProductName; // Define s variable to fix the error
                        
                        return `
                            <div class="product-card" style="margin: 15px 0; padding: 10px; border: 1px solid #eee; border-radius: 8px;">
                                <img src="${imageUrl}" 
                                     alt="${productName}" 
                                     style="width: 100%; height: 200px; object-fit: contain; border-radius: 4px; margin-bottom: 8px;">
                                <h4 style="margin: 8px 0; font-size: 16px;">${productName}</h4>
                                <p style="margin: 4px 0; font-size: 14px; color: #555;">${productDesc}</p>
                                <p style="margin: 8px 0; font-weight: bold; color: #2c3e50;">${productPrice}</p>
                                <button onclick="alert('Added to cart: ${safeProductName}')" 
                                        style="background: ${widgetContainer.dataset.primaryColor || '#4a6fa5'}; 
                                               color: white; 
                                               border: none; 
                                               padding: 8px 16px; 
                                               border-radius: 4px; 
                                               cursor: pointer;
                                               width: 100%;
                                               font-size: 14px;
                                               transition: background-color 0.2s;
                                               margin-top: 5px;"
                                        onmouseover="this.style.opacity='0.9'"
                                        onmouseout="this.style.opacity='1'">
                                    Add to Cart
                                </button>
                            </div>
                        `;
                    }).join('');
                    
                    const botMessage = document.createElement('div');
                    botMessage.className = 'message bot-message';
                    botMessage.innerHTML = `
                        <div style="margin-bottom: 10px;">
                            ${data.message || 'I found these similar items for you!'}
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px;">
                            ${resultsHtml}
                        </div>
                    `;
                    chatMessages.appendChild(botMessage);
                } else {
                    addMessage(data.message || "I couldn't find any similar items. Try uploading a different image or describe what you're looking for.");
                }
                
                return data;
                
            } catch (error) {
                // Ensure processing message is removed in case of error
                if (processingMsg && processingMsg.parentNode === chatMessages) {
                    chatMessages.removeChild(processingMsg);
                }
                
                // Remove loading indicator if it exists
                if (loadingElement) {
                    removeLoading(loadingElement);
                }
                
                console.error('Image upload failed:', error);
                
                // Show a user-friendly error message
                let errorMessage = 'An error occurred while processing your image. ';
                
                // Handle specific error cases
                if (error.message.includes('NetworkError')) {
                    errorMessage += 'Please check your internet connection and try again.';
                } else if (error.message.includes('API key')) {
                    errorMessage += 'There\'s an issue with the widget configuration. Please contact support.';
                } else if (error.message.includes('timeout') || error.message.includes('timed out')) {
                    errorMessage += 'The request took too long. The server might be busy. Please try again later.';
                } else if (error.message.includes('Failed to fetch')) {
                    errorMessage += 'Could not connect to the server. Please check your internet connection.';
                } else {
                    // Use the error message from the server if available
                    errorMessage = error.message || 'Please try again with a different image.';
                }
                
                // Add the error message to the chat
                addMessage(errorMessage);
                
                // Optionally, log the full error to the console for debugging
                console.error('Full error details:', {
                    error: error,
                    name: error.name,
                    message: error.message,
                    stack: error.stack
                });
            }
            
        } catch (error) {
            // This outer catch is now just a safety net for any unhandled errors
            console.error('Unhandled error in handleImageUpload:', error);
            
            // Only show a generic error message if we haven't already shown one
            if (!chatMessages.querySelector('.error-shown')) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message bot-message error-shown';
                errorDiv.textContent = 'An unexpected error occurred. Please try again or contact support if the problem persists.';
                chatMessages.appendChild(errorDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Log the full error for debugging
            console.error('Full error details:', {
                error: error,
                name: error.name,
                message: error.message,
                stack: error.stack
            });
        } finally {
            // Always ensure loading is removed in case of any errors
            if (loadingElement) {
                removeLoading(loadingElement);
            }
        }
    }

    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function handleUserMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        addMessage(message, true);
        userInput.value = '';

        // Simulate bot response (replace with actual API call)
        setTimeout(() => {
            const products = JSON.parse(widgetContainer.dataset.products || '[]');
            const productMatch = products.find(p => 
                message.toLowerCase().includes(p.name.toLowerCase()) ||
                message.toLowerCase().includes(p.category.toLowerCase())
            );

            if (productMatch) {
                const productHtml = `
                    <div class="product-card">
                        <img src="${productMatch.image_url}" alt="${productMatch.name}">
                        <h4>${productMatch.name}</h4>
                        <p>${productMatch.description}</p>
                        <p><strong>$${productMatch.price}</strong></p>
                        <button onclick="alert('Added to cart: ${productMatch.name}')" 
                                style="background: ${widgetContainer.dataset.primaryColor || '#4a6fa5'}; 
                                       color: white; 
                                       border: none; 
                                       padding: 5px 10px; 
                                       border-radius: 4px; 
                                       cursor: pointer;">
                            Add to Cart
                        </button>
                    </div>
                `;
                const botMessage = document.createElement('div');
                botMessage.className = 'message bot-message';
                botMessage.innerHTML = `I found this ${productMatch.category.toLowerCase()} for you!` + productHtml;
                chatMessages.appendChild(botMessage);
            } else {
                addMessage("I'm here to help you find the perfect jewelry. You can ask me about necklaces, rings, earrings, bangles, or bracelets!");
            }
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 1000);
    }

    // Event listeners
    sendButton.addEventListener('click', handleUserMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleUserMessage();
    });

    // Handle image upload
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // Get or create session ID before calling handleImageUpload
            let sessionId = localStorage.getItem('chatbot_session_id');
            if (!sessionId) {
                sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('chatbot_session_id', sessionId);
            }
            handleImageUpload(file);
            // Reset the input to allow selecting the same file again
            e.target.value = '';
        }
    });

    // Allow dropping images
    widgetContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        widgetContainer.style.border = '2px dashed ' + (widgetContainer.dataset.primaryColor || '#4a6fa5');
    });

    widgetContainer.addEventListener('dragleave', () => {
        widgetContainer.style.border = 'none';
    });

    widgetContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        widgetContainer.style.border = 'none';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            // Get or create session ID before calling handleImageUpload
            let sessionId = localStorage.getItem('chatbot_session_id');
            if (!sessionId) {
                sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('chatbot_session_id', sessionId);
            }
            handleImageUpload(file);
        }
    });

    minimizeButton.addEventListener('click', () => {
        const messages = document.querySelector('.chat-messages');
        const input = document.querySelector('.chat-input');
        
        if (isMinimized) {
            messages.style.display = 'block';
            input.style.display = 'flex';
            widgetContainer.style.height = '500px';
            minimizeButton.textContent = '−';
        } else {
            messages.style.display = 'none';
            input.style.display = 'none';
            widgetContainer.style.height = 'auto';
            minimizeButton.textContent = '+';
        }
        isMinimized = !isMinimized;
    });
});
