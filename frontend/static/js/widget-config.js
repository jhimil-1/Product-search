document.addEventListener('DOMContentLoaded', function() {
    // Get form elements
    const form = document.getElementById('widget-config-form');
    const widgetName = document.getElementById('widget-name');
    const primaryColor = document.getElementById('primary-color');
    const position = document.getElementById('position');
    const greetingMessage = document.getElementById('greeting-message');
    const embedCodeSection = document.getElementById('embed-code-section');
    const embedCode = document.getElementById('embed-code');
    const copyBtn = document.querySelector('.copy-btn');
    const preview = document.getElementById('widget-preview');

    // Initialize form with default values
    function initForm() {
        if (!form) return;

        // Set initial values
        widgetName.value = 'My Store Assistant';
        primaryColor.value = '#4a6fa5';
        position.value = 'bottom-right';
        greetingMessage.value = 'Hello! How can I help you today?';

        // Update preview on input changes
        [widgetName, primaryColor, position, greetingMessage].forEach(input => {
            input.addEventListener('input', updatePreview);
        });

        // Handle form submission
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            updateEmbedCode();
            embedCodeSection.style.display = 'block';
            
            // Scroll to embed code section
            embedCodeSection.scrollIntoView({ behavior: 'smooth' });
        });

        // Handle copy button click
        if (copyBtn) {
            copyBtn.addEventListener('click', copyToClipboard);
        }

        // Initial preview update
        updatePreview();
    }

    // Update the preview with current settings
    function updatePreview() {
        if (!preview) return;

        const config = getCurrentConfig();
        
        // Update preview styles
        preview.style.backgroundColor = config.primaryColor;
        preview.style.position = 'absolute';
        preview.style.width = '60px';
        preview.style.height = '60px';
        preview.style.borderRadius = '50%';
        preview.style.display = 'flex';
        preview.style.alignItems = 'center';
        preview.style.justifyContent = 'center';
        preview.style.cursor = 'pointer';
        preview.style.color = 'white';
        preview.style.fontSize = '24px';
        preview.style.bottom = config.position.includes('bottom') ? '20px' : 'auto';
        preview.style.top = config.position.includes('top') ? '20px' : 'auto';
        preview.style.left = config.position.includes('left') ? '20px' : 'auto';
        preview.style.right = config.position.includes('right') ? '20px' : 'auto';
        
        // Add chat icon (using a simple dot if no icon font is available)
        preview.innerHTML = '💬'; // Using emoji as fallback
    }

    // Get current configuration from form
    function getCurrentConfig() {
        return {
            widgetName: widgetName.value,
            primaryColor: primaryColor.value,
            position: position.value,
            greetingMessage: greetingMessage.value
        };
    }

    // Generate and update embed code
    function updateEmbedCode() {
        const config = getCurrentConfig();
        const code = `<!-- Add this to your website's HTML -->
<div id="chatbot-widget"
     data-widget-name="${config.widgetName}"
     data-primary-color="${config.primaryColor}"
     data-position="${config.position}"
     data-greeting-message="${config.greetingMessage}">
</div>
<script src="${window.location.origin}/static/js/widget_embed.js" defer></script>`;
        
        if (embedCode) {
            embedCode.textContent = code;
        }
    }

    // Copy embed code to clipboard
    function copyToClipboard() {
        if (!embedCode) return;
        
        navigator.clipboard.writeText(embedCode.textContent).then(() => {
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            setTimeout(() => {
                copyBtn.textContent = originalText;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    }

    // Initialize the form when the DOM is loaded
    initForm();
});