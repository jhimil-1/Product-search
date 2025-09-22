// Embeddable Chatbot Widget
class ChatbotWidget {
    constructor(config) {
        this.config = {
            apiKey: config.apiKey || '',
            apiUrl: config.apiUrl || window.location.origin,
            primaryColor: config.primaryColor || '#4a6fa5',
            greeting: config.greeting || 'Hello! How can I help you today?',
            position: config.position || 'bottom-right',
            autoInit: config.autoInit !== false,
            container: config.container || 'chatbot-widget'
        };
        
        this.products = [];
        this.sessionId = this.generateSessionId();
        
        if (this.config.autoInit) {
            this.init();
        }
    }
    
    static init(config = {}) {
        // Get config from data attributes if not provided
        const container = document.getElementById(config.container || 'chatbot-widget');
        if (container) {
            config = {
                ...config,
                apiKey: container.dataset.apiKey || config.apiKey,
                primaryColor: container.dataset.primaryColor || config.primaryColor,
                greeting: container.dataset.greeting || config.greeting,
                position: container.dataset.position || config.position,
                autoInit: container.dataset.autoInit !== 'false' && (config.autoInit !== false)
            };
        }
        
        return new ChatbotWidget(config);
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }
    
    async init() {
        this.createWidget();
        await this.loadProducts();
        this.setupEventListeners();
        this.showGreeting();
    }
    
    createWidget() {
        // Create widget container
        this.widget = document.createElement('div');
        this.widget.id = 'chatbot-widget-container';
        this.widget.className = `chatbot-widget ${this.config.position}`;
        this.widget.style.setProperty('--primary-color', this.config.primaryColor);
        
        // Widget HTML
        this.widget.innerHTML = `
            <div class="chatbot-header">
                <h3>Chat with us</h3>
                <button class="chatbot-close">×</button>
            </div>
            <div class="chatbot-messages" id="chatbot-messages"></div>
            <div class="chatbot-input-container">
                <div class="file-upload">
                    <label for="image-upload" class="upload-btn">
                        <i class="fas fa-camera"></i>
                    </label>
                    <input type="file" id="image-upload" accept="image/*" style="display: none;">
                </div>
                <input type="text" id="chatbot-input" placeholder="Type your message...">
                <button id="chatbot-send">Send</button>
            </div>
        `;
        
        // Append to container or body
        const container = document.getElementById(this.config.container);
        if (container) {
            container.appendChild(this.widget);
        } else {
            document.body.appendChild(this.widget);
        }
    }
    
    async loadProducts() {
        try {
            const response = await fetch(`${this.config.apiUrl}/api/widget/products`, {
                headers: {
                    'X-API-Key': this.config.apiKey
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.products = data.products || [];
                this.addMessage('Products loaded successfully!', 'bot');
            } else {
                console.error('Failed to load products');
            }
        } catch (error) {
            console.error('Error loading products:', error);
        }
    }
    
    setupEventListeners() {
        const input = this.widget.querySelector('#chatbot-input');
        const sendButton = this.widget.querySelector('#chatbot-send');
        const closeButton = this.widget.querySelector('.chatbot-close');
        const imageUpload = this.widget.querySelector('#image-upload');
        
        // Send message on button click
        sendButton.addEventListener('click', () => this.handleSendMessage());
        
        // Send message on Enter key
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSendMessage();
            }
        });
        
        // Handle image upload
        imageUpload.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Close widget
        closeButton.addEventListener('click', () => {
            this.widget.style.display = 'none';
        });
    }
    
    async handleSendMessage() {
        const input = this.widget.querySelector('#chatbot-input');
        const message = input.value.trim();
        
        if (message) {
            this.addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch(`${this.config.apiUrl}/api/widget/query`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': this.config.apiKey
                    },
                    body: JSON.stringify({
                        query: message,
                        sessionId: this.sessionId
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    this.handleBotResponse(data);
                } else {
                    throw new Error('Failed to get response');
                }
            } catch (error) {
                console.error('Error sending message:', error);
                this.addMessage('Sorry, something went wrong. Please try again.', 'bot');
            }
        }
    }
    
    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const formData = new FormData();
            formData.append('image', file);
            
            this.addMessage('Processing image...', 'bot');
            
            const response = await fetch(`${this.config.apiUrl}/api/widget/search-by-image`, {
                method: 'POST',
                headers: {
                    'X-API-Key': this.config.apiKey,
                    'X-Session-Id': this.sessionId
                },
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                this.handleSearchResults(data);
            } else {
                throw new Error('Image search failed');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            this.addMessage('Sorry, there was an error processing your image.', 'bot');
        }
    }
    
    handleBotResponse(data) {
        if (data.answer) {
            this.addMessage(data.answer, 'bot');
        }
        
        if (data.products && data.products.length > 0) {
            this.showProducts(data.products);
        }
    }
    
    handleSearchResults(data) {
        if (data.products && data.products.length > 0) {
            this.addMessage('Here are some products that match your image:', 'bot');
            this.showProducts(data.products);
        } else {
            this.addMessage('No matching products found for this image.', 'bot');
        }
    }
    
    showProducts(products) {
        const productsHTML = products.map(product => `
            <div class="product-card">
                ${product.image ? `<img src="${product.image}" alt="${product.name}">` : ''}
                <h4>${product.name}</h4>
                <p>${product.description || ''}</p>
                <div class="product-price">$${product.price || 'N/A'}</div>
                ${product.url ? `<a href="${product.url}" target="_blank" class="product-link">View Product</a>` : ''}
            </div>
        `).join('');
        
        const productsContainer = document.createElement('div');
        productsContainer.className = 'products-container';
        productsContainer.innerHTML = productsHTML;
        
        const messages = this.widget.querySelector('.chatbot-messages');
        messages.appendChild(productsContainer);
        messages.scrollTop = messages.scrollHeight;
    }
    
    addMessage(text, sender) {
        const messages = this.widget.querySelector('.chatbot-messages');
        const message = document.createElement('div');
        message.className = `message ${sender}-message`;
        message.textContent = text;
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }
    
    showGreeting() {
        if (this.config.greeting) {
            this.addMessage(this.config.greeting, 'bot');
        }
    }
}

// Auto-initialize if data attributes are present
document.addEventListener('DOMContentLoaded', () => {
    const widgetElement = document.getElementById('chatbot-widget');
    if (widgetElement) {
        ChatbotWidget.init();
    }
});

// Make ChatbotWidget available globally
window.ChatbotWidget = ChatbotWidget;
