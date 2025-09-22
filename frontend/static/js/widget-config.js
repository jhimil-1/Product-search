// Base configuration for the widget
const WIDGET_CONFIG = {
    apiBaseUrl: window.location.protocol + '//' + window.location.host,
    widgetName: 'My Store Assistant',
    primaryColor: '#4a6fa5',
    position: 'bottom-right',
    greetingMessage: 'Hello! How can I help you today?',
    maxWidth: '400px',
    maxHeight: '600px'
};

// Merge custom configuration with defaults
function getWidgetConfig(customConfig = {}) {
    return {
        ...WIDGET_CONFIG,
        ...customConfig,
        // Ensure colors are properly formatted
        primaryColor: formatColor(customConfig.primaryColor || WIDGET_CONFIG.primaryColor)
    };
}

// Helper function to format color values
function formatColor(color) {
    // If it's already a hex color, return as is
    if (/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(color)) {
        return color;
    }
    
    // If it's an RGB/RGBA color, convert to hex
    if (color.startsWith('rgb')) {
        const rgba = color.match(/\d+/g);
        if (rgba && rgba.length >= 3) {
            const r = parseInt(rgba[0]).toString(16).padStart(2, '0');
            const g = parseInt(rgba[1]).toString(16).padStart(2, '0');
            const b = parseInt(rgba[2]).toString(16).padStart(2, '0');
            return `#${r}${g}${b}`.toLowerCase();
        }
    }
    
    // Default to primary color if invalid
    return WIDGET_CONFIG.primaryColor;
}

// Export for both ES modules and CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WIDGET_CONFIG, getWidgetConfig };
} else {
    window.WidgetConfig = { WIDGET_CONFIG, getWidgetConfig };
}
