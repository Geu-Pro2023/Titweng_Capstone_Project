// Admin Management System
class AdminManager {
    constructor() {
        this.currentUser = this.getCurrentUser();
        this.initializeAdminManagement();
    }

    getCurrentUser() {
        const userData = localStorage.getItem('admin_user');
        return userData ? JSON.parse(userData) : null;
    }

    initializeAdminManagement() {
        // Load admin list on page load
        if (document.getElementById('adminTableBody')) {
            this.loadAdminList();
        }

        // Setup form handlers
        const addAdminForm = document.getElementById('addAdminForm');
        if (addAdminForm) {
            addAdminForm.addEventListener('submit', this.handleAddAdmin.bind(this));
        }

        // Check permissions for admin management
        this.checkAdminPermissions();
    }

    checkAdminPermissions() {
        if (!this.currentUser) return;

        const adminManagementSection = document.getElementById('admins');
        const adminNavLink = document.querySelector('a[href="#admins"]');

        // Only Super Admins can manage other admins
        if (this.currentUser.role !== 'super_admin') {
            if (adminNavLink) {
                adminNavLink.style.display = 'none';
            }
        }
    }

    async loadAdminList() {
        try {
            const response = await authenticatedFetch(`${AUTH_CONFIG.API_BASE_URL}/admins`);
            
            if (!response.ok) {
                throw new Error('Failed to load admin list');
            }

            const admins = await response.json();
            this.renderAdminTable(admins);
        } catch (error) {
            console.error('Error loading admin list:', error);
            this.renderAdminTable(this.getMockAdminData());
        }
    }

    renderAdminTable(admins) {
        const tbody = document.getElementById('adminTableBody');
        if (!tbody) return;

        tbody.innerHTML = admins.map(admin => `
            <tr>
                <td>${admin.id}</td>
                <td>${admin.fullName}</td>
                <td>${admin.email}</td>
                <td>
                    <span class="location-badge">${admin.location}</span>
                </td>
                <td>
                    <span class="status ${admin.role.replace('_', '-')}">${this.formatRole(admin.role)}</span>
                </td>
                <td>
                    <span class="status ${admin.status}">${admin.status}</span>
                </td>
                <td>${this.formatDate(admin.lastLogin)}</td>
                <td>
                    <button class="btn tertiary" onclick="adminManager.viewAdmin('${admin.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${admin.id !== this.currentUser?.id ? `
                        <button class="btn secondary" onclick="adminManager.toggleAdminStatus('${admin.id}')">
                            <i class="fas fa-${admin.status === 'active' ? 'pause' : 'play'}"></i>
                        </button>
                        <button class="btn tertiary" onclick="adminManager.resetPassword('${admin.id}')">
                            <i class="fas fa-key"></i>
                        </button>
                    ` : ''}
                </td>
            </tr>
        `).join('');
    }

    async handleAddAdmin(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const adminData = {
            fullName: formData.get('fullName'),
            email: formData.get('email'),
            phone: formData.get('phone'),
            location: formData.get('location'),
            role: formData.get('role'),
            department: formData.get('department'),
            employeeId: formData.get('employeeId'),
            tempPassword: formData.get('tempPassword'),
            createdBy: this.currentUser.id
        };

        try {
            const response = await authenticatedFetch(`${AUTH_CONFIG.API_BASE_URL}/admins`, {
                method: 'POST',
                body: JSON.stringify(adminData)
            });

            if (response.ok) {
                const result = await response.json();
                showLoginMessage('Admin account created successfully!', 'success');
                closeModal('addAdminModal');
                this.loadAdminList();
                
                // Show credentials to current admin
                this.showNewAdminCredentials(result);
            } else {
                const error = await response.json();
                showLoginMessage(error.message || 'Failed to create admin account', 'error');
            }
        } catch (error) {
            console.error('Error creating admin:', error);
            showLoginMessage('Connection error. Please try again.', 'error');
        }
    }

    showNewAdminCredentials(adminData) {
        const modal = document.getElementById('customModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');
        
        modalTitle.textContent = 'New Admin Account Created';
        modalBody.innerHTML = `
            <div class="admin-credentials">
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i>
                    <strong>Admin account created successfully!</strong>
                </div>
                
                <h4>Login Credentials:</h4>
                <div class="credentials-info">
                    <div class="credential-item">
                        <label>Email:</label>
                        <span class="credential-value">${adminData.email}</span>
                        <button onclick="copyToClipboard('${adminData.email}')" class="btn tertiary">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                    <div class="credential-item">
                        <label>Temporary Password:</label>
                        <span class="credential-value">${adminData.tempPassword}</span>
                        <button onclick="copyToClipboard('${adminData.tempPassword}')" class="btn tertiary">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                </div>
                
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Important:</strong> The admin must change this password on first login.
                    Share these credentials securely with the new administrator.
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    async viewAdmin(adminId) {
        try {
            const response = await authenticatedFetch(`${AUTH_CONFIG.API_BASE_URL}/admins/${adminId}`);
            const admin = await response.json();
            
            const modalBody = document.getElementById('adminDetailsBody');
            modalBody.innerHTML = `
                <div class="admin-details">
                    <div class="detail-section">
                        <h4>Personal Information</h4>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Full Name:</label>
                                <span>${admin.fullName}</span>
                            </div>
                            <div class="detail-item">
                                <label>Email:</label>
                                <span>${admin.email}</span>
                            </div>
                            <div class="detail-item">
                                <label>Phone:</label>
                                <span>${admin.phone}</span>
                            </div>
                            <div class="detail-item">
                                <label>Employee ID:</label>
                                <span>${admin.employeeId || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h4>Role & Permissions</h4>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Role:</label>
                                <span class="status ${admin.role.replace('_', '-')}">${this.formatRole(admin.role)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Department:</label>
                                <span>${admin.department || 'N/A'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Status:</label>
                                <span class="status ${admin.status}">${admin.status}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h4>Activity Information</h4>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Created:</label>
                                <span>${this.formatDate(admin.createdAt)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Last Login:</label>
                                <span>${this.formatDate(admin.lastLogin)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Total Logins:</label>
                                <span>${admin.loginCount || 0}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            openModal('adminDetailsModal');
        } catch (error) {
            console.error('Error loading admin details:', error);
            showLoginMessage('Failed to load admin details', 'error');
        }
    }

    async toggleAdminStatus(adminId) {
        if (!confirm('Are you sure you want to change this admin\'s status?')) {
            return;
        }

        try {
            const response = await authenticatedFetch(`${AUTH_CONFIG.API_BASE_URL}/admins/${adminId}/toggle-status`, {
                method: 'PATCH'
            });

            if (response.ok) {
                showLoginMessage('Admin status updated successfully', 'success');
                this.loadAdminList();
            } else {
                showLoginMessage('Failed to update admin status', 'error');
            }
        } catch (error) {
            console.error('Error toggling admin status:', error);
            showLoginMessage('Connection error. Please try again.', 'error');
        }
    }

    async resetPassword(adminId) {
        if (!confirm('Are you sure you want to reset this admin\'s password?')) {
            return;
        }

        try {
            const response = await authenticatedFetch(`${AUTH_CONFIG.API_BASE_URL}/admins/${adminId}/reset-password`, {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                showLoginMessage('Password reset successfully', 'success');
                this.showPasswordResetInfo(result);
            } else {
                showLoginMessage('Failed to reset password', 'error');
            }
        } catch (error) {
            console.error('Error resetting password:', error);
            showLoginMessage('Connection error. Please try again.', 'error');
        }
    }

    showPasswordResetInfo(resetData) {
        alert(`New temporary password: ${resetData.tempPassword}\n\nPlease share this securely with the admin.`);
    }

    formatRole(role) {
        const roleMap = {
            'super_admin': 'Super Admin',
            'location_admin': 'Location Admin',
            'admin': 'Admin',
            'operator': 'Operator'
        };
        return roleMap[role] || role;
    }

    formatDate(dateString) {
        if (!dateString) return 'Never';
        return new Date(dateString).toLocaleDateString();
    }

    getMockAdminData() {
        return [
            {
                id: '1',
                fullName: 'John Doe',
                email: 'john.juba@titweng.gov.ss',
                location: 'Juba',
                role: 'super_admin',
                status: 'active',
                lastLogin: '2024-01-15T10:30:00Z',
                department: 'System Administration'
            },
            {
                id: '2',
                fullName: 'Jane Smith',
                email: 'jane.wau@titweng.gov.ss',
                location: 'Wau',
                role: 'location_admin',
                status: 'active',
                lastLogin: '2024-01-14T15:45:00Z',
                department: 'Livestock Registration'
            },
            {
                id: '3',
                fullName: 'Peter Malakal',
                email: 'peter.malakal@titweng.gov.ss',
                location: 'Malakal',
                role: 'location_admin',
                status: 'active',
                lastLogin: '2024-01-13T09:20:00Z',
                department: 'Livestock Registration'
            }
        ];
    }
}

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showLoginMessage('Copied to clipboard!', 'success');
    });
}

// Initialize admin manager
let adminManager;
document.addEventListener('DOMContentLoaded', () => {
    adminManager = new AdminManager();
});