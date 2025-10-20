// Security Configuration and Measures
const SECURITY_CONFIG = {
    // Rate limiting for login attempts
    MAX_LOGIN_ATTEMPTS: 5,
    LOCKOUT_DURATION: 15 * 60 * 1000, // 15 minutes
    
    // Session security
    IDLE_TIMEOUT: 30 * 60 * 1000, // 30 minutes
    
    // Security headers
    SECURITY_HEADERS: {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block'
    }
};

// Rate limiting for login attempts
class LoginRateLimit {
    constructor() {
        this.attempts = JSON.parse(localStorage.getItem('login_attempts') || '{}');
    }
    
    isBlocked(username) {
        const userAttempts = this.attempts[username];
        if (!userAttempts) return false;
        
        const now = Date.now();
        if (userAttempts.count >= SECURITY_CONFIG.MAX_LOGIN_ATTEMPTS) {
            if (now - userAttempts.lastAttempt < SECURITY_CONFIG.LOCKOUT_DURATION) {
                return true;
            } else {
                // Reset after lockout period
                delete this.attempts[username];
                this.save();
                return false;
            }
        }
        return false;
    }
    
    recordAttempt(username, success = false) {
        const now = Date.now();
        
        if (success) {
            // Clear attempts on successful login
            delete this.attempts[username];
        } else {
            // Record failed attempt
            if (!this.attempts[username]) {
                this.attempts[username] = { count: 0, lastAttempt: now };
            }
            this.attempts[username].count++;
            this.attempts[username].lastAttempt = now;
        }
        
        this.save();
    }
    
    getRemainingLockoutTime(username) {
        const userAttempts = this.attempts[username];
        if (!userAttempts) return 0;
        
        const elapsed = Date.now() - userAttempts.lastAttempt;
        const remaining = SECURITY_CONFIG.LOCKOUT_DURATION - elapsed;
        return Math.max(0, Math.ceil(remaining / 1000 / 60)); // minutes
    }
    
    save() {
        localStorage.setItem('login_attempts', JSON.stringify(this.attempts));
    }
}

// Initialize rate limiter
const loginRateLimit = new LoginRateLimit();

// Security warning messages
const SECURITY_MESSAGES = {
    UNAUTHORIZED_ACCESS: "⚠️ UNAUTHORIZED ACCESS DETECTED ⚠️\nThis is a restricted admin dashboard. All access attempts are logged and monitored.",
    SESSION_EXPIRED: "🔒 Your session has expired for security reasons. Please login again.",
    RATE_LIMITED: "🚫 Too many failed login attempts. Account temporarily locked.",
    SUSPICIOUS_ACTIVITY: "⚠️ Suspicious activity detected. Please contact system administrator."
};

// Display security warning
function showSecurityWarning() {
    const warningDiv = document.createElement('div');
    warningDiv.className = 'security-warning';
    warningDiv.innerHTML = `
        <div class="security-warning-content">
            <i class="fas fa-shield-alt"></i>
            <h3>🔒 RESTRICTED ACCESS</h3>
            <p>This is an authorized personnel only area.</p>
            <p>All activities are monitored and logged.</p>
            <small>Unauthorized access is prohibited and may result in legal action.</small>
        </div>
    `;
    
    document.body.appendChild(warningDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (warningDiv.parentNode) {
            warningDiv.remove();
        }
    }, 5000);
}

// Enhanced login validation with rate limiting
async function secureLogin(username, password) {
    // Check if user is rate limited
    if (loginRateLimit.isBlocked(username)) {
        const remainingTime = loginRateLimit.getRemainingLockoutTime(username);
        showLoginMessage(`Account locked. Try again in ${remainingTime} minutes.`, 'error');
        return false;
    }
    
    try {
        const response = await fetch(`${AUTH_CONFIG.API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...SECURITY_CONFIG.SECURITY_HEADERS
            },
            body: JSON.stringify({ 
                username, 
                password,
                timestamp: Date.now(),
                userAgent: navigator.userAgent
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.token) {
            // Record successful login
            loginRateLimit.recordAttempt(username, true);
            
            // Store token with expiration
            const tokenData = {
                token: data.token,
                expires: Date.now() + (24 * 60 * 60 * 1000), // 24 hours
                user: data.user
            };
            
            localStorage.setItem(AUTH_CONFIG.TOKEN_KEY, JSON.stringify(tokenData));
            return true;
        } else {
            // Record failed attempt
            loginRateLimit.recordAttempt(username, false);
            showLoginMessage(data.message || 'Invalid credentials', 'error');
            return false;
        }
    } catch (error) {
        console.error('Login error:', error);
        showLoginMessage('Connection error. Please try again.', 'error');
        return false;
    }
}

// Enhanced token validation
function validateToken() {
    const tokenData = localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
    
    if (!tokenData) return false;
    
    try {
        const parsed = JSON.parse(tokenData);
        
        // Check if token is expired
        if (Date.now() > parsed.expires) {
            localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
            showLoginMessage(SECURITY_MESSAGES.SESSION_EXPIRED, 'error');
            return false;
        }
        
        return parsed.token;
    } catch (error) {
        localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
        return false;
    }
}

// Secure logout with cleanup
function secureLogout() {
    // Clear all stored data
    localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
    localStorage.removeItem('admin_user');
    
    // Clear session storage
    sessionStorage.clear();
    
    // Redirect to login
    window.location.href = 'index.html';
}

// Monitor for suspicious activity
function initSecurityMonitoring() {
    // Detect developer tools
    let devtools = {open: false, orientation: null};
    
    setInterval(() => {
        if (window.outerHeight - window.innerHeight > 200 || 
            window.outerWidth - window.innerWidth > 200) {
            if (!devtools.open) {
                devtools.open = true;
                console.warn(SECURITY_MESSAGES.SUSPICIOUS_ACTIVITY);
                // Optional: Log this event to your backend
            }
        } else {
            devtools.open = false;
        }
    }, 500);
    
    // Disable right-click context menu
    document.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        return false;
    });
    
    // Disable common keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Disable F12, Ctrl+Shift+I, Ctrl+U, etc.
        if (e.key === 'F12' || 
            (e.ctrlKey && e.shiftKey && e.key === 'I') ||
            (e.ctrlKey && e.key === 'u')) {
            e.preventDefault();
            return false;
        }
    });
}

// Initialize security measures
document.addEventListener('DOMContentLoaded', () => {
    showSecurityWarning();
    initSecurityMonitoring();
});

// Export for use in other files
window.SECURITY = {
    secureLogin,
    validateToken,
    secureLogout,
    loginRateLimit,
    SECURITY_MESSAGES
};