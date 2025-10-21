// Security Configuration
const AUTH_CONFIG = {
    API_BASE_URL: window.location.hostname === 'localhost' 
        ? 'http://localhost:8000'
        : 'https://titweng-capstone-project.onrender.com',
    TOKEN_KEY: 'titweng_admin_token',
    SESSION_TIMEOUT: 30 * 60 * 1000 // 30 minutes
};

// Check authentication on page load
document.addEventListener('DOMContentLoaded', function() {
    checkAuthStatus();
    
    const loginForm = document.getElementById('loginForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
});

// Check if user is authenticated
function checkAuthStatus() {
    const token = localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
    const currentPage = window.location.pathname;
    
    if (currentPage.includes('dashboard.html') && !token) {
        // Redirect to login if accessing dashboard without token
        window.location.href = 'index.html';
        return;
    }
    
    if (token && currentPage.includes('index.html')) {
        // Redirect to dashboard if already logged in
        window.location.href = 'dashboard.html';
        return;
    }
    
    if (token) {
        // Verify token validity
        verifyToken(token);
    }
}

// Handle login form submission
async function handleLogin(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        showLoginMessage('Please fill in all fields', 'error');
        return;
    }
    
    try {
        showLoginMessage('Authenticating...', 'info');
        
        const response = await fetch(`${AUTH_CONFIG.API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.token) {
            // Store token securely
            localStorage.setItem(AUTH_CONFIG.TOKEN_KEY, data.token);
            localStorage.setItem('admin_user', JSON.stringify(data.user));
            
            showLoginMessage('Login successful! Redirecting...', 'success');
            
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        } else {
            showLoginMessage(data.message || 'Invalid credentials', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showLoginMessage('Connection error. Please try again.', 'error');
    }
}

// Verify token validity
async function verifyToken(token) {
    try {
        const response = await fetch(`${AUTH_CONFIG.API_BASE_URL}/auth/verify`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            // Token invalid, logout
            logoutAdmin();
        }
    } catch (error) {
        console.error('Token verification error:', error);
        logoutAdmin();
    }
}

// Logout function
function logoutAdmin() {
    localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
    localStorage.removeItem('admin_user');
    window.location.href = 'index.html';
}

// Auto-logout on session timeout
let sessionTimer;
function resetSessionTimer() {
    clearTimeout(sessionTimer);
    sessionTimer = setTimeout(() => {
        showLoginMessage('Session expired. Please login again.', 'error');
        logoutAdmin();
    }, AUTH_CONFIG.SESSION_TIMEOUT);
}

// Reset timer on user activity
document.addEventListener('click', resetSessionTimer);
document.addEventListener('keypress', resetSessionTimer);

// Initialize session timer
if (localStorage.getItem(AUTH_CONFIG.TOKEN_KEY)) {
    resetSessionTimer();
}

// Custom login message function
function showLoginMessage(message, type) {
    const existingMessage = document.querySelector('.login-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `login-message ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                type === 'info' ? 'fa-info-circle' : 'fa-exclamation-circle';
    
    messageDiv.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    const form = document.querySelector('.auth-form');
    if (form && form.parentNode) {
        form.parentNode.insertBefore(messageDiv, form);
    }
    
    if (type === 'error' || type === 'info') {
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 5000);
    }
}

// API request helper with authentication
async function authenticatedFetch(url, options = {}) {
    const token = localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
    
    if (!token) {
        logoutAdmin();
        return;
    }
    
    const defaultOptions = {
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            ...options.headers
        }
    };
    
    try {
        const response = await fetch(url, { ...options, ...defaultOptions });
        
        if (response.status === 401) {
            logoutAdmin();
            return;
        }
        
        return response;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

// Password visibility toggle
function togglePassword(inputId) {
    const passwordInput = document.getElementById(inputId);
    const eyeIcon = document.getElementById(inputId + '-eye');
    
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        eyeIcon.className = 'fas fa-eye-slash';
    } else {
        passwordInput.type = 'password';
        eyeIcon.className = 'fas fa-eye';
    }
}

// Password strength indicator (optional enhancement)
function checkPasswordStrength(password) {
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^A-Za-z0-9]/.test(password)) strength++;
    
    return strength;
}