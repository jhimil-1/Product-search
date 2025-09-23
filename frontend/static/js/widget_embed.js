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
                    <label for="image-upload" class="file-upload-label" title="Upload Image">
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
    
    // Generate a unique session ID for the chat
    const sessionId = 'widget-' + Math.random().toString(36).substr(2, 9);
    
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
        if (!file) return;
        
        // Show the uploaded image in chat
        const reader = new FileReader();
        reader.onload = function(e) {
            const imgMsg = document.createElement('div');
            imgMsg.className = 'message user-message';
            imgMsg.style.textAlign = 'right';
            imgMsg.innerHTML = `
                <div style="display: inline-block; max-width: 200px; margin-bottom: 5px;">
                    <img src="${e.target.result}" style="max-width: 100%; border-radius: 10px;" alt="Uploaded image">
                </div>
            `;
            chatMessages.appendChild(imgMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };
        reader.readAsDataURL(file);
        
        // Show loading indicator
        const loadingElement = showLoading();
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('image', file);
            
            // Get API key from widget container
            const apiKey = widgetContainer.dataset.apiKey;
            console.log('Widget API Key:', apiKey);
            if (!apiKey) {
                console.error('API key is missing from widget container');
                throw new Error('API key is required');
            }
            
            // Send image to server for processing
            const response = await fetch(`/api/v1/widget/query-image/${sessionId}`, {
                method: 'POST',
                headers: {
                    'X-API-Key': apiKey
                },
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', response.status, errorText);
                throw new Error(`Server responded with ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            removeLoading(loadingElement);
            
            if (data.error) {
                addMessage("Sorry, I couldn't process that image. Please try again.");
                return;
            }
            
            // Display search results
            if (data.results && data.results.length > 0) {
                const resultsHtml = data.results.slice(0, 3).map(product => `
                    <div class="product-card" style="margin-top: 10px;">
                        <img src="${product.image_url}" alt="${product.name}" style="max-width: 100%; border-radius: 5px;">
                        <h4>${product.name}</h4>
                        <p>${product.description}</p>
                        <p><strong>$${product.price}</strong></p>
                        <button onclick="alert('Added to cart: ${product.name}')" 
                                style="background: ${widgetContainer.dataset.primaryColor || '#4a6fa5'}; 
                                       color: white; 
                                       border: none; 
                                       padding: 5px 10px; 
                                       border-radius: 4px; 
                                       cursor: pointer;">
                            Add to Cart
                        </button>
                    </div>
                `).join('');
                
                const botMessage = document.createElement('div');
                botMessage.className = 'message bot-message';
                botMessage.innerHTML = `I found these similar items for you!` + resultsHtml;
                chatMessages.appendChild(botMessage);
            } else {
                addMessage("I couldn't find any similar items. Try uploading a different image or describe what you're looking for.");
            }
            
        } catch (error) {
            console.error('Error processing image:', error);
            removeLoading(loadingElement);
            addMessage("Sorry, there was an error processing your image. Please try again.");
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
            const searchTerm = message.toLowerCase();
            
            // Find all matching products (by name or category)
            const searchTerms = searchTerm.split(' ').filter(term => term.length > 2);
            
            const matchingProducts = products.filter(p => {
                // Check if any search term is in the product name or category
                return searchTerms.some(term => 
                    p.name.toLowerCase().includes(term) ||
                    p.category.toLowerCase().includes(term) ||
                    p.description.toLowerCase().includes(term)
                ) || 
                // Also check if the full search term is contained in any field
                p.name.toLowerCase().includes(searchTerm) ||
                p.category.toLowerCase().includes(searchTerm) ||
                p.description.toLowerCase().includes(searchTerm);
            });

            if (matchingProducts.length > 0) {
                // Group products by category
                const productsByCategory = matchingProducts.reduce((acc, product) => {
                    if (!acc[product.category]) {
                        acc[product.category] = [];
                    }
                    acc[product.category].push(product);
                    return acc;
                }, {});

                // Create HTML for all matching products
                let productsHtml = '';
                Object.entries(productsByCategory).forEach(([category, categoryProducts]) => {
                    productsHtml += `<h4>${category}</h4>`;
                    productsHtml += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px;">';
                    
                    categoryProducts.forEach(product => {
                        productsHtml += `
                        <div class="product-card" style="border: 1px solid #eee; padding: 10px; border-radius: 8px;">
                            <img src="${product.image_url}" alt="${product.name}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 4px;">
                            <h5 style="margin: 8px 0 4px;">${product.name}</h5>
                            <p style="font-size: 0.9em; color: #666; margin: 4px 0;">${product.description}</p>
                            <p style="font-weight: bold; margin: 8px 0;">$${product.price}</p>
                            <button onclick="alert('Added to cart: ${product.name.replace(/'/g, "\'")}')" 
                                    style="background: ${widgetContainer.dataset.primaryColor || '#4a6fa5'}; 
                                           color: white; 
                                           border: none; 
                                           padding: 6px 12px; 
                                           border-radius: 4px; 
                                           cursor: pointer;
                                           width: 100%;
                                           font-size: 0.9em;">
                                Add to Cart
                            </button>
                        </div>`;
                    });
                    
                    productsHtml += '</div>';
                });

                const botMessage = document.createElement('div');
                botMessage.className = 'message bot-message';
                botMessage.innerHTML = `I found ${matchingProducts.length} matching items for you!` + productsHtml;
                chatMessages.appendChild(botMessage);
            } else {
                addMessage("I couldn't find any items matching your search. You can ask me about necklaces, rings, earrings, bangles, or bracelets!");
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
