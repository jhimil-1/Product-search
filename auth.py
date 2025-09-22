from functools import wraps
from flask import request, jsonify, g, redirect, url_for, session, current_app
import time
from datetime import datetime, timedelta
from config import Config
import logging

def require_login(f):
    """Decorator to require user login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'status': 'error',
                    'error': 'Authentication required',
                    'code': 401
                }), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or not session.get('is_admin', False):
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'status': 'error',
                    'error': 'Admin privileges required',
                    'code': 403
                }), 403
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Rate limiting storage (in production, use Redis or similar)
_rate_limits = {}

def _is_rate_limited(api_key, endpoint):
    """Check if the API key has exceeded the rate limit for the endpoint"""
    now = datetime.utcnow()
    key = f"{api_key}:{endpoint}"
    
    # Clean up old entries
    _rate_limits[key] = [t for t in _rate_limits.get(key, []) 
                        if now - t < timedelta(hours=1)]
    
    # Check rate limit
    rate_limit = Config.WIDGET_API_KEYS.get(api_key, {}).get('rate_limit', 100)
    if len(_rate_limits.get(key, [])) >= rate_limit:
        return True
    
    # Record this request
    if key not in _rate_limits:
        _rate_limits[key] = []
    _rate_limits[key].append(now)
    return False

def require_api_key(f):
    """Decorator to require a valid API key for the endpoint"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from headers or query params
        api_key = request.headers.get('X-API-KEY') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({
                'error': 'API key is missing',
                'status': 'error',
                'code': 401
            }), 401
        
        # Check if API key exists and is enabled
        key_info = Config.WIDGET_API_KEYS.get(api_key)
        if not key_info or not key_info.get('enabled', True):
            return jsonify({
                'error': 'Invalid or disabled API key',
                'status': 'error',
                'code': 403
            }), 403
        
        # Check rate limiting
        endpoint = f"{request.endpoint}:{request.method}"
        if _is_rate_limited(api_key, endpoint):
            return jsonify({
                'error': 'Rate limit exceeded',
                'status': 'error',
                'code': 429
            }), 429
        
        # Store key info in g for later use in the route
        g.api_key = api_key
        g.key_info = key_info
        
        return f(*args, **kwargs)
    
    return decorated_function

def has_permission(permission):
    """Check if the current API key has the required permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'key_info' not in g or permission not in g.key_info.get('permissions', []):
                return jsonify({
                    'error': f'Missing required permission: {permission}',
                    'status': 'error',
                    'code': 403
                }), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator
