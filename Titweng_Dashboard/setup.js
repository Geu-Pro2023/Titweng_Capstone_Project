// System Setup Configuration
const SETUP_CONFIG = {
    API_BASE_URL: window.location.hostname === 'localhost' 
        ? 'http://localhost:8000/api'
        : 'https://your-backend-api.com/api'
};

// Setup state management
let setupState = {
    currentStep: 'systemCheck',
    systemReady: false,
    rootAdminCreated: false
};

// Initialize setup process
document.addEventListener('DOMContentLoaded', function() {
    checkIfSetupNeeded();
    initializeSetup();
});

// Check if system setup is needed
async function checkIfSetupNeeded() {
    try {
        const response = await fetch(`${SETUP_CONFIG.API_BASE_URL}/setup/status`);
        const data = await response.json();
        
        if (data.setupComplete) {
            // Redirect to login if setup is already complete
            window.location.href = 'index.html';
        }
    } catch (error) {
        console.log('Setup needed - proceeding with initialization');
    }
}

// Initialize setup interface
function initializeSetup() {
    const rootAdminForm = document.getElementById('rootAdminForm');
    if (rootAdminForm) {
        rootAdminForm.addEventListener('submit', handleRootAdminCreation);
    }

    // Password strength checker
    const passwordInput = document.getElementById('rootPassword');
    if (passwordInput) {
        passwordInput.addEventListener('input', checkPasswordStrength);
    }
}

// Run system check
async function runSystemCheck() {
    const checkBtn = document.getElementById('checkBtn');
    checkBtn.disabled = true;
    checkBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking...';

    // Simulate system checks
    await checkDatabase();
    await checkAPI();
    await checkSecurity();

    // If all checks pass, proceed to next step
    if (setupState.systemReady) {
        setTimeout(() => {
            showStep('rootAdminSetup');
        }, 1000);
    }
}

// Check database connection
async function checkDatabase() {
    const statusItem = document.getElementById('dbStatus');
    const indicator = statusItem.querySelector('.status-indicator');
    
    try {
        const response = await fetch(`${SETUP_CONFIG.API_BASE_URL}/setup/check-database`);
        
        if (response.ok) {
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            statusItem.classList.add('success');
        } else {
            throw new Error('Database connection failed');
        }
    } catch (error) {
        indicator.innerHTML = '<i class="fas fa-times-circle text-error"></i>';
        statusItem.classList.add('error');
        setupState.systemReady = false;
        return;
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
}

// Check API service
async function checkAPI() {
    const statusItem = document.getElementById('apiStatus');
    const indicator = statusItem.querySelector('.status-indicator');
    
    try {
        const response = await fetch(`${SETUP_CONFIG.API_BASE_URL}/setup/check-api`);
        
        if (response.ok) {
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            statusItem.classList.add('success');
        } else {
            throw new Error('API service not available');
        }
    } catch (error) {
        indicator.innerHTML = '<i class="fas fa-times-circle text-error"></i>';
        statusItem.classList.add('error');
        setupState.systemReady = false;
        return;
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
}

// Check security configuration
async function checkSecurity() {
    const statusItem = document.getElementById('securityStatus');
    const indicator = statusItem.querySelector('.status-indicator');
    
    try {
        // Check if HTTPS is enabled in production
        const isSecure = window.location.protocol === 'https:' || 
                        window.location.hostname === 'localhost';
        
        if (isSecure) {
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            statusItem.classList.add('success');
            setupState.systemReady = true;
        } else {
            throw new Error('Security requirements not met');
        }
    } catch (error) {
        indicator.innerHTML = '<i class="fas fa-times-circle text-error"></i>';
        statusItem.classList.add('error');
        setupState.systemReady = false;
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
}

// Handle root admin creation
async function handleRootAdminCreation(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const password = formData.get('password');
    const confirmPassword = formData.get('confirmPassword');
    
    // Validate passwords match
    if (password !== confirmPassword) {
        showSetupMessage('Passwords do not match', 'error');
        return;
    }
    
    // Validate password strength
    if (getPasswordStrength(password) < 3) {
        showSetupMessage('Password is too weak. Please use a stronger password.', 'error');
        return;
    }
    
    const rootAdminData = {
        fullName: formData.get('fullName'),
        email: formData.get('email'),
        phone: formData.get('phone'),
        organization: formData.get('organization'),
        password: password,
        role: 'root_admin'
    };
    
    try {
        showSetupMessage('Creating root administrator...', 'info');
        
        const response = await fetch(`${SETUP_CONFIG.API_BASE_URL}/setup/create-root-admin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(rootAdminData)
        });
        
        if (response.ok) {
            const result = await response.json();
            setupState.rootAdminCreated = true;
            showSetupMessage('Root administrator created successfully!', 'success');
            
            setTimeout(() => {
                showStep('setupComplete');
            }, 2000);
        } else {
            const error = await response.json();
            showSetupMessage(error.message || 'Failed to create root administrator', 'error');
        }
    } catch (error) {
        console.error('Setup error:', error);
        showSetupMessage('Connection error. Please try again.', 'error');
    }
}

// Password strength checker
function checkPasswordStrength() {
    const password = document.getElementById('rootPassword').value;
    const strengthDiv = document.getElementById('passwordStrength');
    const strength = getPasswordStrength(password);
    
    const strengthLevels = [
        { level: 0, text: 'Very Weak', class: 'very-weak' },
        { level: 1, text: 'Weak', class: 'weak' },
        { level: 2, text: 'Fair', class: 'fair' },
        { level: 3, text: 'Good', class: 'good' },
        { level: 4, text: 'Strong', class: 'strong' },
        { level: 5, text: 'Very Strong', class: 'very-strong' }
    ];
    
    const currentLevel = strengthLevels[strength];
    strengthDiv.innerHTML = `
        <div class="strength-bar ${currentLevel.class}">
            <div class="strength-fill" style="width: ${(strength / 5) * 100}%"></div>
        </div>
        <span class="strength-text ${currentLevel.class}">${currentLevel.text}</span>
    `;
}

// Calculate password strength
function getPasswordStrength(password) {
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^A-Za-z0-9]/.test(password)) strength++;
    return strength;
}

// Show setup step
function showStep(stepId) {
    // Hide all steps
    document.querySelectorAll('.setup-step').forEach(step => {
        step.classList.remove('active');
    });
    
    // Show target step
    document.getElementById(stepId).classList.add('active');
    setupState.currentStep = stepId;
}

// Show setup message
function showSetupMessage(message, type) {
    // Remove existing message
    const existingMessage = document.querySelector('.setup-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `setup-message ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                type === 'info' ? 'fa-info-circle' : 'fa-exclamation-circle';
    
    messageDiv.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    // Insert message
    const activeStep = document.querySelector('.setup-step.active');
    if (activeStep) {
        activeStep.insertBefore(messageDiv, activeStep.firstChild);
    }
    
    // Auto remove after delay
    if (type === 'error' || type === 'info') {
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 5000);
    }
}

// Go to dashboard after setup
function goToDashboard() {
    window.location.href = 'index.html';
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