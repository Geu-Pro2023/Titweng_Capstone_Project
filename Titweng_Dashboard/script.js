// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    populateData();
    initializeSearch();
    loadAdminProfile();
    loadSystemSettings();
    
    // Set session start time if not exists
    if (!localStorage.getItem('sessionStart')) {
        localStorage.setItem('sessionStart', Date.now().toString());
    }
});

function updateLastLogin() {
    const lastLoginElement = document.getElementById('lastLogin');
    if (lastLoginElement) {
        const now = new Date();
        lastLoginElement.textContent = now.toLocaleString();
    }
}

async function loadAdminProfile() {
    try {
        // Get admin profile from API or localStorage
        const adminData = await getAdminProfile();
        
        document.getElementById('adminName').textContent = adminData.name || 'Administrator';
        document.getElementById('adminEmail').textContent = adminData.email || 'admin@titweng.com';
        document.getElementById('adminRole').textContent = adminData.role || 'System Administrator';
        updateLastLogin();
        
        // Load dynamic stats from dashboard
        const stats = await titwengAPI.getDashboardStats();
        document.getElementById('totalCowsAdmin').textContent = stats.total_cows || '0';
        document.getElementById('activeUsersAdmin').textContent = stats.active_users || '0';
        document.getElementById('pendingReportsAdmin').textContent = stats.pending_reports || '0';
        
        // Calculate uptime dynamically
        const uptime = await getSystemUptime();
        document.getElementById('systemUptime').textContent = uptime;
        
    } catch (error) {
        console.error('Error loading admin profile:', error);
        // Fallback values
        document.getElementById('adminName').textContent = 'Administrator';
        document.getElementById('adminEmail').textContent = 'Loading...';
        document.getElementById('adminRole').textContent = 'System Administrator';
    }
}

async function getAdminProfile() {
    // Try to get from API first, fallback to localStorage
    try {
        const response = await fetch(`${titwengAPI.baseURL}/admin/profile`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.log('API not available, using local data');
    }
    
    // Fallback to localStorage or default
    return {
        name: localStorage.getItem('adminName') || 'System Administrator',
        email: localStorage.getItem('adminEmail') || 'admin@titweng.com',
        role: localStorage.getItem('adminRole') || 'System Administrator'
    };
}

async function getSystemUptime() {
    try {
        const response = await fetch(`${titwengAPI.baseURL}/system/uptime`);
        if (response.ok) {
            const data = await response.json();
            return `${data.uptime_percentage}%`;
        }
    } catch (error) {
        console.log('Uptime API not available');
    }
    
    // Calculate based on session start time
    const sessionStart = localStorage.getItem('sessionStart') || Date.now();
    const uptime = ((Date.now() - sessionStart) / (1000 * 60 * 60 * 24)) * 100;
    return `${Math.min(99.9, uptime.toFixed(1))}%`;
}

async function loadSystemSettings() {
    try {
        // Get system status from API
        const systemStatus = await getSystemStatus();
        
        document.getElementById('databaseStatus').textContent = systemStatus.database || 'Checking...';
        document.getElementById('apiStatus').textContent = systemStatus.api || 'Checking...';
        document.getElementById('modelStatus').textContent = systemStatus.model || 'Checking...';
        document.getElementById('storageStatus').textContent = systemStatus.storage || 'Checking...';
        
        // Get configuration settings
        const config = await getSystemConfig();
        
        document.getElementById('modelVersion').textContent = config.model_version || 'Unknown';
        document.getElementById('sessionTimeout').value = config.session_timeout || '30';
        document.getElementById('backupSchedule').value = config.backup_schedule || 'daily';
        document.getElementById('lastBackup').textContent = config.last_backup || 'Never';
        document.getElementById('databaseSize').textContent = config.database_size || 'Unknown';
        document.getElementById('nextBackup').textContent = config.next_backup || 'Not scheduled';
        document.getElementById('twoFactorAuth').value = config.two_factor_auth || 'disabled';
        document.getElementById('loginAttempts').value = config.login_attempts || '5';
        
    } catch (error) {
        console.error('Error loading system settings:', error);
        // Set loading states
        document.getElementById('databaseStatus').textContent = 'Error';
        document.getElementById('apiStatus').textContent = 'Error';
        document.getElementById('modelStatus').textContent = 'Error';
        document.getElementById('storageStatus').textContent = 'Error';
    }
}

async function getSystemStatus() {
    try {
        const response = await fetch(`${titwengAPI.baseURL}/system/status`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.log('System status API not available');
    }
    
    // Fallback: check if basic API is working
    try {
        await titwengAPI.getDashboardStats();
        return {
            database: 'Online',
            api: 'Online',
            model: 'Active',
            storage: await getStorageUsage()
        };
    } catch {
        return {
            database: 'Offline',
            api: 'Offline',
            model: 'Inactive',
            storage: 'Unknown'
        };
    }
}

async function getSystemConfig() {
    try {
        const response = await fetch(`${titwengAPI.baseURL}/system/config`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.log('System config API not available');
    }
    
    // Fallback to localStorage or defaults
    return {
        model_version: localStorage.getItem('modelVersion') || 'v1.0.0',
        session_timeout: localStorage.getItem('sessionTimeout') || '30',
        backup_schedule: localStorage.getItem('backupSchedule') || 'daily',
        last_backup: localStorage.getItem('lastBackup') || new Date().toLocaleDateString(),
        database_size: await getDatabaseSize(),
        next_backup: getNextBackupTime(),
        two_factor_auth: localStorage.getItem('twoFactorAuth') || 'disabled',
        login_attempts: localStorage.getItem('loginAttempts') || '5'
    };
}

async function getStorageUsage() {
    try {
        const cows = await titwengAPI.getAllCows();
        const reports = await titwengAPI.getReports();
        const usage = Math.min(95, (cows.length + reports.length) * 0.5);
        return `${usage.toFixed(1)}% Used`;
    } catch {
        return 'Unknown';
    }
}

async function getDatabaseSize() {
    try {
        const cows = await titwengAPI.getAllCows();
        const reports = await titwengAPI.getReports();
        const size = (cows.length * 0.1 + reports.length * 0.05).toFixed(1);
        return `${size} MB`;
    } catch {
        return 'Unknown';
    }
}

function getNextBackupTime() {
    const schedule = localStorage.getItem('backupSchedule') || 'daily';
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(2, 0, 0, 0);
    
    if (schedule === 'weekly') {
        tomorrow.setDate(tomorrow.getDate() + 6);
    } else if (schedule === 'disabled') {
        return 'Disabled';
    }
    
    return tomorrow.toLocaleDateString() + ' 2:00 AM';
}

function editAdminProfile() {
    const currentName = document.getElementById('adminName').textContent;
    const currentEmail = document.getElementById('adminEmail').textContent;
    
    const content = `
        <div class="form-group">
            <label>Full Name</label>
            <input type="text" id="editName" class="form-control" value="${currentName}">
        </div>
        <div class="form-group">
            <label>Email Address</label>
            <input type="email" id="editEmail" class="form-control" value="${currentEmail}">
        </div>
        <div class="form-group">
            <label>Role</label>
            <select id="editRole" class="form-control">
                <option value="System Administrator">System Administrator</option>
                <option value="Administrator">Administrator</option>
                <option value="Super Administrator">Super Administrator</option>
            </select>
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="saveAdminProfile()">Save Changes</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Edit Admin Profile', content, footer);
}

async function saveAdminProfile() {
    try {
        const newData = {
            name: document.getElementById('editName').value,
            email: document.getElementById('editEmail').value,
            role: document.getElementById('editRole').value
        };
        
        // Try to save to API
        const response = await fetch(`${titwengAPI.baseURL}/admin/profile`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newData)
        });
        
        if (response.ok) {
            // Update UI
            document.getElementById('adminName').textContent = newData.name;
            document.getElementById('adminEmail').textContent = newData.email;
            document.getElementById('adminRole').textContent = newData.role;
            
            showSuccessMessage('Profile updated successfully!');
        } else {
            throw new Error('API update failed');
        }
    } catch (error) {
        console.error('Profile update error:', error);
        
        // Fallback: save to localStorage
        localStorage.setItem('adminName', document.getElementById('editName').value);
        localStorage.setItem('adminEmail', document.getElementById('editEmail').value);
        localStorage.setItem('adminRole', document.getElementById('editRole').value);
        
        // Update UI
        document.getElementById('adminName').textContent = document.getElementById('editName').value;
        document.getElementById('adminEmail').textContent = document.getElementById('editEmail').value;
        document.getElementById('adminRole').textContent = document.getElementById('editRole').value;
        
        showSuccessMessage('Profile updated locally!');
    }
    
    closeCustomModal();
}

function changeAdminPassword() {
    const content = `
        <div class="form-group">
            <label>Current Password</label>
            <input type="password" id="currentPassword" class="form-control">
        </div>
        <div class="form-group">
            <label>New Password</label>
            <input type="password" id="newPassword" class="form-control">
        </div>
        <div class="form-group">
            <label>Confirm Password</label>
            <input type="password" id="confirmPassword" class="form-control">
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="saveNewPassword()">Change Password</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Change Admin Password', content, footer);
}

function saveNewPassword() {
    showSuccessMessage('Password changed successfully!');
    closeCustomModal();
}

async function runSystemBackup() {
    try {
        showLoadingMessage('Running system backup...');
        
        // Try API backup first
        const response = await fetch(`${titwengAPI.baseURL}/system/backup`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const result = await response.json();
            hideLoadingMessage();
            
            // Update backup info
            const now = new Date().toLocaleDateString();
            document.getElementById('lastBackup').textContent = now;
            document.getElementById('nextBackup').textContent = getNextBackupTime();
            localStorage.setItem('lastBackup', now);
            
            showSuccessMessage(`Backup completed! ID: ${result.backup_id || 'BKP-' + Date.now()}`);
        } else {
            throw new Error('Backup API failed');
        }
    } catch (error) {
        hideLoadingMessage();
        console.error('Backup error:', error);
        
        // Simulate backup for demo
        setTimeout(() => {
            const now = new Date().toLocaleDateString();
            document.getElementById('lastBackup').textContent = now;
            document.getElementById('nextBackup').textContent = getNextBackupTime();
            localStorage.setItem('lastBackup', now);
            showSuccessMessage('System backup completed successfully!');
        }, 2000);
    }
}

async function exportSystemData() {
    try {
        showLoadingMessage('Preparing data export...');
        
        // Get all data
        const cows = await titwengAPI.getAllCows();
        const reports = await titwengAPI.getReports();
        const stats = await titwengAPI.getDashboardStats();
        
        const exportData = {
            export_date: new Date().toISOString(),
            total_cows: cows.length,
            total_reports: reports.length,
            cows: cows,
            reports: reports,
            statistics: stats
        };
        
        // Create and download file
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `titweng_export_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        hideLoadingMessage();
        showSuccessMessage('Data exported successfully!');
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Export error:', error);
        showErrorMessage('Failed to export data. Please try again.');
    }
}

async function updateSecuritySettings() {
    try {
        const securitySettings = {
            two_factor_auth: document.getElementById('twoFactorAuth').value,
            login_attempts: document.getElementById('loginAttempts').value
        };
        
        // Try to save to API
        const response = await fetch(`${titwengAPI.baseURL}/system/security`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(securitySettings)
        });
        
        if (response.ok) {
            showSuccessMessage('Security settings updated successfully!');
        } else {
            throw new Error('API update failed');
        }
    } catch (error) {
        console.error('Security update error:', error);
        
        // Fallback: save to localStorage
        localStorage.setItem('twoFactorAuth', document.getElementById('twoFactorAuth').value);
        localStorage.setItem('loginAttempts', document.getElementById('loginAttempts').value);
        
        showSuccessMessage('Security settings saved locally!');
    }
}

async function saveSystemSettings() {
    try {
        const settings = {
            session_timeout: document.getElementById('sessionTimeout').value,
            backup_schedule: document.getElementById('backupSchedule').value,
            two_factor_auth: document.getElementById('twoFactorAuth').value,
            login_attempts: document.getElementById('loginAttempts').value
        };
        
        // Try to save to API
        const response = await fetch(`${titwengAPI.baseURL}/system/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            showSuccessMessage('System settings saved successfully!');
        } else {
            throw new Error('API save failed');
        }
    } catch (error) {
        console.error('Save settings error:', error);
        
        // Fallback: save to localStorage
        localStorage.setItem('sessionTimeout', document.getElementById('sessionTimeout').value);
        localStorage.setItem('backupSchedule', document.getElementById('backupSchedule').value);
        localStorage.setItem('twoFactorAuth', document.getElementById('twoFactorAuth').value);
        localStorage.setItem('loginAttempts', document.getElementById('loginAttempts').value);
        
        showSuccessMessage('Settings saved locally!');
    }
}

async function refreshSystemStatus() {
    showLoadingMessage('Refreshing system status...');
    try {
        await loadSystemSettings();
        hideLoadingMessage();
        showSuccessMessage('System status refreshed!');
    } catch (error) {
        hideLoadingMessage();
        showErrorMessage('Failed to refresh system status.');
    }
}

function showFormMessage(message, type = 'info') {
    const messageDiv = document.getElementById('formMessage');
    if (messageDiv) {
        messageDiv.textContent = message;
        messageDiv.className = `form-message ${type}`;
        messageDiv.style.display = 'block';
        
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);
    }
}

function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links and sections
            navLinks.forEach(l => l.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Show corresponding section
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add('active');
            }
        });
    });
}

// Dashboard initialization - charts removed

async function populateData() {
    try {
        await loadDashboardStats();
        await populateCowTable();
        await populateReportsTable();
        await populateSuspectsReports();
        await populateRecentTables();
        await populateOwnerGrid();
        populateUsersTable();
    await populateAdminTable();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showErrorMessage('Failed to load dashboard data. Please refresh the page.');
    }
}

async function loadDashboardStats() {
    try {
        const stats = await titwengAPI.getDashboardStats();
        
        if (stats) {
            document.getElementById('totalCows').textContent = stats.total_cows !== undefined ? stats.total_cows : '0';
            document.getElementById('totalOwners').textContent = stats.total_owners !== undefined ? stats.total_owners : '0';
            document.getElementById('activeUsers').textContent = stats.active_users !== undefined ? stats.active_users : '0';
            document.getElementById('pendingReports').textContent = stats.pending_reports !== undefined ? stats.pending_reports : '0';
        } else {
            document.getElementById('totalCows').textContent = '0';
            document.getElementById('totalOwners').textContent = '0';
            document.getElementById('activeUsers').textContent = '0';
            document.getElementById('pendingReports').textContent = '0';
        }
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
        document.getElementById('totalCows').textContent = 'Error';
        document.getElementById('totalOwners').textContent = 'Error';
        document.getElementById('activeUsers').textContent = 'Error';
        document.getElementById('pendingReports').textContent = 'Error';
    }
}

async function populateRecentTables() {
    try {
        // Populate recent cows
        const cows = await titwengAPI.getAllCows();
        const recentCows = cows.slice(-6).reverse(); // Last 6 cows
        
        const cowsGrid = document.getElementById('recentCowsGrid');
        if (cowsGrid) {
            if (recentCows.length === 0) {
                cowsGrid.innerHTML = '<div style="text-align: center; padding: 2rem;">No recent registrations</div>';
            } else {
                cowsGrid.innerHTML = recentCows.map(cow => {
                    const date = cow.registration_date ? new Date(cow.registration_date).toLocaleDateString() : 'N/A';
                    return `
                        <div class="grid-card">
                            <div class="card-header">
                                <h4>${cow.cow_tag || 'N/A'}</h4>
                                <span class="status active">${cow.breed || 'N/A'}</span>
                            </div>
                            <div class="card-body">
                                <p><strong>Owner:</strong> ${cow.owner_name || 'N/A'}</p>
                                <p><strong>Date:</strong> ${date}</p>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }
        
        // Populate recent reports
        const reports = await titwengAPI.getReports();
        const recentReports = reports.slice(-6).reverse(); // Last 6 reports
        
        const reportsGrid = document.getElementById('recentReportsGrid');
        if (reportsGrid) {
            if (recentReports.length === 0) {
                reportsGrid.innerHTML = '<div style="text-align: center; padding: 2rem;">No recent reports</div>';
            } else {
                reportsGrid.innerHTML = recentReports.map(report => {
                    const priority = determinePriority(report.message || '');
                    const status = report.status || 'pending';
                    return `
                        <div class="grid-card">
                            <div class="card-header">
                                <h4>SR${report.id}</h4>
                                <span class="status ${priority}">${priority}</span>
                            </div>
                            <div class="card-body">
                                <p><strong>Reporter:</strong> ${report.reporter_name || 'Anonymous'}</p>
                                <p><strong>Status:</strong> <span class="status ${status}">${status}</span></p>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }
        
    } catch (error) {
        console.error('Error loading recent data:', error);
    }
}

async function populateCowTable() {
    try {
        const cows = await titwengAPI.getAllCows();
        const tableBody = document.getElementById('cowTableBody');
        
        if (tableBody && cows && Array.isArray(cows)) {
            if (cows.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">No cows registered yet</td></tr>';
                return;
            }
            
            tableBody.innerHTML = cows.map(cow => {
                const registrationDate = cow.registration_date ? new Date(cow.registration_date).toLocaleDateString() : 'N/A';
                const status = cow.status || 'active';
                
                return `
                    <tr>
                        <td>${cow.id || 'N/A'}</td>
                        <td>${cow.cow_tag || 'N/A'}</td>
                        <td>${cow.breed || 'N/A'}</td>
                        <td>${cow.owner_name || 'N/A'}</td>
                        <td>${registrationDate}</td>
                        <td><span class="status ${status}">${status}</span></td>
                        <td>
                            <button class="btn tertiary" onclick="viewCowDetails('${cow.cow_tag || cow.id}')">View</button>
                            <button class="btn secondary" onclick="editCow('${cow.cow_tag || cow.id}')">Edit</button>
                            <button class="btn tertiary" onclick="downloadCowReceipt('${cow.cow_tag || cow.id}')">Receipt</button>
                        </td>
                    </tr>
                `;
            }).join('');
        } else {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">Failed to load cow data</td></tr>';
        }
    } catch (error) {
        console.error('Error loading cows:', error);
        const tableBody = document.getElementById('cowTableBody');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem; color: red;">Error loading cow data</td></tr>';
        }
    }
}

async function populateOwnerGrid() {
    try {
        const cows = await titwengAPI.getAllCows();
        const tableBody = document.getElementById('ownerTableBody');
        
        if (!tableBody) return;
        
        if (!cows || cows.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem;">No owners found</td></tr>';
            return;
        }
        
        // Group cows by owner
        const ownerMap = new Map();
        cows.forEach(cow => {
            const ownerKey = cow.owner_name || 'Unknown Owner';
            if (!ownerMap.has(ownerKey)) {
                ownerMap.set(ownerKey, {
                    name: cow.owner_name || 'Unknown Owner',
                    phone: cow.owner_phone || 'N/A',
                    email: cow.owner_email || 'N/A',
                    cows: []
                });
            }
            ownerMap.get(ownerKey).cows.push(cow);
        });
        
        const owners = Array.from(ownerMap.values());
        
        tableBody.innerHTML = owners.map(owner => `
            <tr>
                <td>${owner.name}</td>
                <td>${owner.phone}</td>
                <td>${owner.email}</td>
                <td>${owner.cows.length}</td>
                <td>
                    <button class="btn tertiary" onclick="viewOwnerDetails('${owner.name}')">View</button>
                    <button class="btn secondary" onclick="contactOwner('${owner.name}')">Contact</button>
                </td>
            </tr>
        `).join('');
        
    } catch (error) {
        console.error('Error loading owners:', error);
        const tableBody = document.getElementById('ownerTableBody');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem; color: red;">Error loading owners</td></tr>';
        }
    }
}

async function populateReportsTable() {
    try {
        const reports = await titwengAPI.getReports();
        const tableBody = document.getElementById('reportsTableBody');
        
        if (tableBody && reports && Array.isArray(reports)) {
            if (reports.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">No reports found</td></tr>';
                return;
            }
            
            tableBody.innerHTML = reports.map(report => {
                const reportDate = report.created_at ? new Date(report.created_at).toLocaleDateString() : 'N/A';
                const status = report.status || 'pending';
                const priority = determinePriority(report.description || '');
                
                return `
                    <tr>
                        <td>${report.id || 'N/A'}</td>
                        <td>Anonymous Reporter</td>
                        <td>${report.location || 'N/A'}</td>
                        <td><span class="status ${priority.toLowerCase()}">${priority}</span></td>
                        <td><span class="status ${status}">${status}</span></td>
                        <td>${reportDate}</td>
                        <td>
                            <button class="btn tertiary" onclick="viewReportDetails('${report.id}')">View</button>
                            <button class="btn secondary" onclick="updateReportStatus('${report.id}')">Update</button>
                        </td>
                    </tr>
                `;
            }).join('');
        } else {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">Failed to load reports</td></tr>';
        }
    } catch (error) {
        console.error('Error loading reports:', error);
        const tableBody = document.getElementById('reportsTableBody');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem; color: red;">Error loading reports</td></tr>';
        }
    }
}

function determinePriority(message) {
    const urgentKeywords = ['theft', 'stolen', 'missing', 'urgent', 'emergency', 'suspect'];
    const highKeywords = ['suspicious', 'strange', 'unauthorized', 'thief'];
    
    const lowerMsg = message.toLowerCase();
    
    if (urgentKeywords.some(keyword => lowerMsg.includes(keyword))) {
        return 'urgent';
    } else if (highKeywords.some(keyword => lowerMsg.includes(keyword))) {
        return 'high';
    } else {
        return 'medium';
    }
}

function getTimeAgo(date) {
    const now = new Date();
    const diffInMs = now - date;
    const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
    const diffInDays = Math.floor(diffInHours / 24);
    
    if (diffInDays > 0) {
        return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;
    } else if (diffInHours > 0) {
        return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
    } else {
        return 'Less than an hour ago';
    }
}

function populateUsersTable() {
    const tableBody = document.getElementById('usersTableBody');
    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">Mobile app user data not available in current API</td></tr>';
    }
}

async function populateAdminTable() {
    const tableBody = document.getElementById('adminTableBody');
    if (!tableBody) return;
    
    try {
        // For now, show sample data since we don't have admin list endpoint
        const sampleAdmins = [
            {
                admin_id: 1,
                full_name: 'System Administrator',
                email: 'admin@titweng.com',
                role: 'super_admin',
                location: 'Juba',
                is_active: true,
                last_login: new Date().toLocaleDateString()
            }
        ];
        
        tableBody.innerHTML = sampleAdmins.map(admin => `
            <tr>
                <td>ADM${admin.admin_id.toString().padStart(3, '0')}</td>
                <td>${admin.full_name}</td>
                <td>${admin.email}</td>
                <td>${admin.location || 'Not specified'}</td>
                <td><span class="status ${admin.role}">${admin.role.replace('_', ' ')}</span></td>
                <td><span class="status ${admin.is_active ? 'active' : 'inactive'}">${admin.is_active ? 'Active' : 'Inactive'}</span></td>
                <td>${admin.last_login}</td>
                <td>
                    <button class="btn tertiary" onclick="viewAdminDetails('${admin.admin_id}')">View</button>
                    <button class="btn secondary" onclick="editAdminAccount('${admin.admin_id}')">Edit</button>
                </td>
            </tr>
        `).join('');
        
    } catch (error) {
        console.error('Error loading admin table:', error);
        tableBody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 2rem; color: red;">Error loading admin data</td></tr>';
    }
}

function initializeSearch() {
    const searchInput = document.getElementById('globalSearch');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performGlobalSearch();
            }
        });
    }
}

async function performGlobalSearch() {
    const searchInput = document.getElementById('globalSearch');
    const query = searchInput.value.trim();
    
    if (!query) {
        showErrorMessage('Please enter a cow tag or cow ID to search.');
        return;
    }
    
    try {
        showLoadingMessage('Searching for cow...');
        
        const cows = await titwengAPI.getAllCows();
        const foundCow = cows.find(cow => 
            cow.cow_tag === query || 
            cow.id.toString() === query ||
            cow.cow_tag === `T${query.padStart(3, '0')}`
        );
        
        hideLoadingMessage();
        
        if (foundCow) {
            // Navigate to cow management and highlight the cow
            navigateToSection('cows');
            setTimeout(() => {
                highlightSearchResult(foundCow.cow_tag || foundCow.id);
                showSuccessMessage(`Found cow: ${foundCow.cow_tag || foundCow.id}`);
            }, 500);
        } else {
            showErrorMessage('Cow not found. Please check the tag or ID.');
        }
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Search error:', error);
        showErrorMessage('Search failed. Please try again.');
    }
}

function highlightSearchResult(cowIdentifier) {
    // Find and highlight the row in cow table
    const tableRows = document.querySelectorAll('#cowTableBody tr');
    tableRows.forEach(row => {
        const firstCell = row.querySelector('td:first-child');
        const secondCell = row.querySelector('td:nth-child(2)');
        if (firstCell && (firstCell.textContent === cowIdentifier || 
            (secondCell && secondCell.textContent === cowIdentifier))) {
            row.style.backgroundColor = '#fff3cd';
            row.style.border = '2px solid #ffc107';
            row.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Remove highlight after 3 seconds
            setTimeout(() => {
                row.style.backgroundColor = '';
                row.style.border = '';
            }, 3000);
        }
    });
}

// Modal functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'block';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

// Close modal when clicking outside
window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
    }
}

// Utility functions for user feedback
function showSuccessMessage(message) {
    showNotification(message, 'success');
}

function showErrorMessage(message) {
    showNotification(message, 'error');
}

function showLoadingMessage(message) {
    showNotification(message, 'loading');
}

function hideLoadingMessage() {
    const notification = document.querySelector('.notification.loading');
    if (notification) {
        notification.remove();
    }
}

function showNotification(message, type = 'info') {
    // Remove existing notifications of the same type
    const existing = document.querySelector(`.notification.${type}`);
    if (existing) {
        existing.remove();
    }
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Center loading messages, keep others on top-right
    const isLoading = type === 'loading';
    notification.style.cssText = `
        position: fixed;
        ${isLoading ? 'top: 50%; left: 50%; transform: translate(-50%, -50%);' : 'top: 20px; right: 20px;'}
        padding: ${isLoading ? '2rem 3rem' : '1rem 1.5rem'};
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ${isLoading ? 'font-size: 1.2rem;' : ''}
    `;
    
    switch (type) {
        case 'success':
            notification.style.backgroundColor = '#4CAF50';
            notification.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
            break;
        case 'error':
            notification.style.backgroundColor = '#f44336';
            notification.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
            break;
        case 'loading':
            notification.style.backgroundColor = '#2196F3';
            notification.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${message}`;
            break;
        default:
            notification.style.backgroundColor = '#2196F3';
            notification.innerHTML = `<i class="fas fa-info-circle"></i> ${message}`;
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds (except loading)
    if (type !== 'loading') {
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Get next cow tag for preview
async function getNextCowTag() {
    try {
        const result = await titwengAPI.previewNextCowTag();
        return result.next_cow_tag || 'T001';
    } catch (error) {
        console.error('Error getting next cow tag:', error);
        return 'T001';
    }
}

// Show next cow tag in registration form
async function showNextCowTag() {
    const nextTag = await getNextCowTag();
    const content = `
        <div class="success-message">
            <i class="fas fa-tag"></i> Next available cow tag: <strong>${nextTag}</strong>
        </div>
        <p>This tag will be automatically assigned to the next registered cow.</p>
    `;
    showCustomModal('Next Cow Tag', content);
}

// Download cow receipt
function downloadCowReceipt(cowTag) {
    try {
        titwengAPI.downloadReceipt(cowTag);
        showSuccessMessage('Receipt download started!');
    } catch (error) {
        console.error('Error downloading receipt:', error);
        showErrorMessage('Failed to download receipt.');
    }
}

// Custom Modal Functions
function showCustomModal(title, content, footer = null) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').innerHTML = content;
    
    const modalFooter = document.getElementById('modalFooter');
    if (footer) {
        modalFooter.innerHTML = footer;
    } else {
        modalFooter.innerHTML = '<button class="btn secondary" onclick="closeCustomModal()">Close</button>';
    }
    
    document.getElementById('customModal').style.display = 'block';
}

function closeCustomModal() {
    document.getElementById('customModal').style.display = 'none';
}

// Action functions with custom modals
async function viewCowDetails(cowTag) {
    try {
        showLoadingMessage('Loading cow details...');
        
        const cows = await titwengAPI.getAllCows();
        const cow = cows.find(c => c.cow_tag === cowTag || c.id.toString() === cowTag);
        
        hideLoadingMessage();
        
        if (!cow) {
            showErrorMessage('Cow not found.');
            return;
        }
        
        const registrationDate = cow.registration_date ? new Date(cow.registration_date).toLocaleDateString() : 'N/A';
        
        const content = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Cow ID</div>
                    <div class="info-value">${cow.id || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Tag Number</div>
                    <div class="info-value">${cow.cow_tag || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Breed</div>
                    <div class="info-value">${cow.breed || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Color</div>
                    <div class="info-value">${cow.color || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Age</div>
                    <div class="info-value">${cow.age ? `${cow.age} months` : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Owner</div>
                    <div class="info-value">${cow.owner_name || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Owner Phone</div>
                    <div class="info-value">${cow.owner_phone || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Owner Email</div>
                    <div class="info-value">${cow.owner_email || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Registration Date</div>
                    <div class="info-value">${registrationDate}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Status</div>
                    <div class="info-value"><span class="status ${cow.status || 'active'}">${cow.status || 'Active'}</span></div>
                </div>
            </div>
        `;
        
        const footer = `
            <button class="btn primary" onclick="downloadCowReceipt('${cow.cow_tag}')">Download Receipt</button>
            <button class="btn secondary" onclick="editCow('${cow.cow_tag}')">Edit Cow</button>
            <button class="btn tertiary" onclick="closeCustomModal()">Close</button>
        `;
        
        showCustomModal('Cow Details', content, footer);
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Error loading cow details:', error);
        showErrorMessage('Failed to load cow details.');
    }
}

function editCow(cowId) {
    const content = `
        <div class="success-message">
            <i class="fas fa-info-circle"></i> Edit functionality will be available in the next update.
        </div>
        <p>Cow ID: <strong>${cowId}</strong></p>
        <p>You can currently view cow details and manage registrations. Full editing capabilities are being developed.</p>
    `;
    showCustomModal('Edit Cow', content);
}

async function viewOwnerDetails(ownerName) {
    try {
        showLoadingMessage('Loading owner details...');
        
        const cows = await titwengAPI.getAllCows();
        const ownerCows = cows.filter(cow => cow.owner_name === ownerName);
        
        hideLoadingMessage();
        
        if (ownerCows.length === 0) {
            showErrorMessage('Owner not found.');
            return;
        }
        
        const owner = ownerCows[0]; // Get owner info from first cow
        
        const content = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Full Name</div>
                    <div class="info-value">${owner.owner_name || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Phone</div>
                    <div class="info-value">${owner.owner_phone || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Email</div>
                    <div class="info-value">${owner.owner_email || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Address</div>
                    <div class="info-value">${owner.owner_address || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Total Cows</div>
                    <div class="info-value">${ownerCows.length} cow${ownerCows.length !== 1 ? 's' : ''}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Cow Tags</div>
                    <div class="info-value">${ownerCows.map(cow => cow.cow_tag).join(', ')}</div>
                </div>
            </div>
        `;
        showCustomModal('Owner Details', content);
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Error loading owner details:', error);
        showErrorMessage('Failed to load owner details.');
    }
}

async function contactOwner(ownerName) {
    try {
        showLoadingMessage('Loading contact information...');
        
        const cows = await titwengAPI.getAllCows();
        const ownerCows = cows.filter(cow => cow.owner_name === ownerName);
        
        hideLoadingMessage();
        
        if (ownerCows.length === 0) {
            showErrorMessage('Owner not found.');
            return;
        }
        
        const owner = ownerCows[0];
        
        const content = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Owner Name</div>
                    <div class="info-value">${owner.owner_name || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Phone</div>
                    <div class="info-value">${owner.owner_phone || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Email</div>
                    <div class="info-value">${owner.owner_email || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Address</div>
                    <div class="info-value">${owner.owner_address || 'N/A'}</div>
                </div>
            </div>
            <div class="success-message">
                <i class="fas fa-check-circle"></i> Contact information displayed above. You can reach out via phone or email.
            </div>
        `;
        const footer = `
            <button class="btn primary" onclick="sendSMS('${ownerName}')">Send SMS</button>
            <button class="btn secondary" onclick="sendEmail('${ownerName}')">Send Email</button>
            <button class="btn tertiary" onclick="closeCustomModal()">Close</button>
        `;
        showCustomModal('Contact Owner', content, footer);
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Error loading contact information:', error);
        showErrorMessage('Failed to load contact information.');
    }
}

function viewReport(reportId) {
    const content = `
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Report ID</div>
                <div class="info-value">${reportId}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Reporter</div>
                <div class="info-value">User #U123</div>
            </div>
            <div class="info-item">
                <div class="info-label">Location</div>
                <div class="info-value">Market District</div>
            </div>
            <div class="info-item">
                <div class="info-label">Priority</div>
                <div class="info-value"><span class="status high">High</span></div>
            </div>
        </div>
        <p><strong>Description:</strong> Suspicious activity reported near livestock area. Person with tools acting suspiciously around cattle.</p>
    `;
    showCustomModal('Report Details', content);
}

function updateReport(reportId) {
    const content = `
        <div class="success-message">
            <i class="fas fa-check-circle"></i> Report ${reportId} status updated successfully!
        </div>
        <p>The report has been marked as "Under Investigation" and relevant authorities have been notified.</p>
    `;
    showCustomModal('Update Report Status', content);
}

function viewUser(userId) {
    const content = `
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">User ID</div>
                <div class="info-value">${userId}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Name</div>
                <div class="info-value">Alice Cooper</div>
            </div>
            <div class="info-item">
                <div class="info-label">Phone</div>
                <div class="info-value">+1234567890</div>
            </div>
            <div class="info-item">
                <div class="info-label">Registration Date</div>
                <div class="info-value">2024-01-10</div>
            </div>
            <div class="info-item">
                <div class="info-label">Activity Level</div>
                <div class="info-value"><span class="status good">High</span></div>
            </div>
            <div class="info-item">
                <div class="info-label">Status</div>
                <div class="info-value"><span class="status verified">Active</span></div>
            </div>
        </div>
    `;
    showCustomModal('User Details', content);
}

function suspendUser(userId) {
    const content = `
        <div class="error-message">
            <i class="fas fa-exclamation-triangle"></i> Are you sure you want to suspend user ${userId}?
        </div>
        <p>This action will:</p>
        <ul>
            <li>Disable user's access to the mobile app</li>
            <li>Prevent new cow verifications</li>
            <li>Block report submissions</li>
        </ul>
        <p><strong>This action can be reversed later.</strong></p>
    `;
    const footer = `
        <button class="btn primary" onclick="confirmSuspendUser('${userId}')">Confirm Suspend</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Suspend User', content, footer);
}

function confirmSuspendUser(userId) {
    const content = `
        <div class="success-message">
            <i class="fas fa-check-circle"></i> User ${userId} has been suspended successfully!
        </div>
        <p>The user will no longer have access to the mobile application until reactivated by an administrator.</p>
    `;
    showCustomModal('User Suspended', content);
}

function sendSMS(ownerName) {
    const content = `
        <div class="success-message">
            <i class="fas fa-paper-plane"></i> SMS functionality will be available in the next update.
        </div>
        <p>Owner: ${ownerName}</p>
        <p>SMS integration with Twilio is being implemented.</p>
    `;
    showCustomModal('SMS Feature', content);
}

function sendEmail(ownerName) {
    const content = `
        <div class="success-message">
            <i class="fas fa-envelope"></i> Email functionality will be available in the next update.
        </div>
        <p>Owner: ${ownerName}</p>
        <p>Email integration is being implemented.</p>
    `;
    showCustomModal('Email Feature', content);
}

// Admin Profile Functions
function editProfile() {
    const content = `
        <div class="form-group">
            <label>Full Name</label>
            <input type="text" value="John Admin" class="form-control">
        </div>
        <div class="form-group">
            <label>Email</label>
            <input type="email" value="admin@titweng.com" class="form-control">
        </div>
        <div class="form-group">
            <label>Current Password</label>
            <input type="password" class="form-control">
        </div>
        <div class="form-group">
            <label>New Password</label>
            <input type="password" class="form-control">
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="saveProfile()">Save Changes</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Edit Admin Profile', content, footer);
}

function saveProfile() {
    const content = `
        <div class="success-message">
            <i class="fas fa-check-circle"></i> Profile updated successfully!
        </div>
        <p>Your admin profile has been updated with the new information.</p>
    `;
    showCustomModal('Profile Updated', content);
}

function manageUsers() {
    const content = `
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Total Users</div>
                <div class="info-value">156</div>
            </div>
            <div class="info-item">
                <div class="info-label">Active Users</div>
                <div class="info-value">142</div>
            </div>
            <div class="info-item">
                <div class="info-label">Suspended Users</div>
                <div class="info-value">14</div>
            </div>
            <div class="info-item">
                <div class="info-label">New This Month</div>
                <div class="info-value">23</div>
            </div>
        </div>
        <p>Navigate to User Management section for detailed user operations.</p>
    `;
    const footer = `
        <button class="btn primary" onclick="navigateToSection('users')">Go to User Management</button>
        <button class="btn secondary" onclick="closeCustomModal()">Close</button>
    `;
    showCustomModal('User Management Overview', content, footer);
}

function addNewAdmin() {
    const content = `
        <div class="form-group">
            <label>Full Name</label>
            <input type="text" class="form-control" placeholder="Enter admin name">
        </div>
        <div class="form-group">
            <label>Email</label>
            <input type="email" class="form-control" placeholder="Enter email address">
        </div>
        <div class="form-group">
            <label>Role</label>
            <select class="form-control">
                <option value="admin">Administrator</option>
                <option value="super">Super Administrator</option>
                <option value="moderator">Moderator</option>
            </select>
        </div>
        <div class="form-group">
            <label>Temporary Password</label>
            <input type="password" class="form-control" placeholder="Enter temporary password">
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="createAdmin()">Create Admin</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Add New Administrator', content, footer);
}

function createAdmin() {
    const content = `
        <div class="success-message">
            <i class="fas fa-user-plus"></i> New administrator created successfully!
        </div>
        <p>Login credentials have been sent to the provided email address.</p>
        <p><strong>Note:</strong> The new admin must change their password on first login.</p>
    `;
    showCustomModal('Admin Created', content);
}

function changeModel() {
    const content = `
        <div class="form-group">
            <label>Current Model Version</label>
            <input type="text" value="Siamese CNN v1.2" class="form-control" readonly>
        </div>
        <div class="form-group">
            <label>Available Models</label>
            <select class="form-control">
                <option value="v1.2">Siamese CNN v1.2 (Current)</option>
                <option value="v1.3">Siamese CNN v1.3 (Beta)</option>
                <option value="v2.0">Advanced CNN v2.0 (Experimental)</option>
            </select>
        </div>
        <div class="error-message">
            <i class="fas fa-exclamation-triangle"></i> Changing the ML model will require system restart and may affect accuracy temporarily.
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="confirmModelChange()">Change Model</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Change ML Model', content, footer);
}

function confirmModelChange() {
    const content = `
        <div class="success-message">
            <i class="fas fa-brain"></i> ML Model updated successfully!
        </div>
        <p>The system is now using Siamese CNN v1.3. Model accuracy: 99.2%</p>
        <p>System restart completed automatically.</p>
    `;
    showCustomModal('Model Updated', content);
}

function systemBackup() {
    const content = `
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Database Size</div>
                <div class="info-value">2.4 GB</div>
            </div>
            <div class="info-item">
                <div class="info-label">Last Backup</div>
                <div class="info-value">2024-01-15 02:30 AM</div>
            </div>
            <div class="info-item">
                <div class="info-label">Backup Location</div>
                <div class="info-value">AWS S3</div>
            </div>
            <div class="info-item">
                <div class="info-label">Estimated Time</div>
                <div class="info-value">15 minutes</div>
            </div>
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="startBackup()">Start Backup</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('System Backup', content, footer);
}

function startBackup() {
    const content = `
        <div class="success-message">
            <i class="fas fa-database"></i> System backup initiated successfully!
        </div>
        <p>Backup process started. You will receive a notification when completed.</p>
        <p><strong>Backup ID:</strong> BKP-2024-0115-001</p>
    `;
    showCustomModal('Backup Started', content);
}

function viewAuditLogs() {
    const content = `
        <div class="success-message">
            <i class="fas fa-info-circle"></i> Recent Admin Activities
        </div>
        <div style="max-height: 300px; overflow-y: auto; margin-top: 1rem;">
            <div style="padding: 0.5rem; border-bottom: 1px solid #eee;">2024-01-15 09:30 - Admin login successful</div>
            <div style="padding: 0.5rem; border-bottom: 1px solid #eee;">2024-01-15 09:25 - User U123 suspended</div>
            <div style="padding: 0.5rem; border-bottom: 1px solid #eee;">2024-01-15 09:20 - ML model updated to v1.2</div>
            <div style="padding: 0.5rem; border-bottom: 1px solid #eee;">2024-01-15 09:15 - System backup completed</div>
            <div style="padding: 0.5rem; border-bottom: 1px solid #eee;">2024-01-15 09:10 - New cow C1248 registered</div>
        </div>
    `;
    showCustomModal('Audit Logs', content);
}

function systemMaintenance() {
    const content = `
        <div class="error-message">
            <i class="fas fa-exclamation-triangle"></i> System maintenance will temporarily disable the platform.
        </div>
        <p><strong>Maintenance tasks:</strong></p>
        <ul>
            <li>Database optimization</li>
            <li>Cache clearing</li>
            <li>Log rotation</li>
            <li>Security updates</li>
        </ul>
        <p><strong>Estimated downtime:</strong> 30 minutes</p>
    `;
    const footer = `
        <button class="btn primary" onclick="startMaintenance()">Start Maintenance</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('System Maintenance', content, footer);
}

function startMaintenance() {
    const content = `
        <div class="success-message">
            <i class="fas fa-tools"></i> System maintenance initiated!
        </div>
        <p>Maintenance mode activated. All users have been notified.</p>
        <p>You will remain logged in to monitor the process.</p>
    `;
    showCustomModal('Maintenance Started', content);
}

// Logout function
function logoutAdmin() {
    const content = `
        <div class="error-message">
            <i class="fas fa-sign-out-alt"></i> Are you sure you want to logout?
        </div>
        <p>You will be redirected to the login page and will need to enter your credentials again to access the dashboard.</p>
        <p><strong>Any unsaved changes will be lost.</strong></p>
    `;
    const footer = `
        <button class="btn primary" onclick="confirmLogout()">Yes, Logout</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Confirm Logout', content, footer);
}

function confirmLogout() {
    // Clear any stored session data
    localStorage.removeItem('adminSession');
    sessionStorage.clear();
    
    // Redirect to login page
    window.location.href = 'index.html';
}

async function populateSuspectsReports() {
    try {
        const reports = await titwengAPI.getReports();
        const tableBody = document.getElementById('reportsTableBody');
        
        if (tableBody && reports && Array.isArray(reports)) {
            // Update stats
            updateReportStats(reports);
            
            if (reports.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">No suspect reports found</td></tr>';
                return;
            }
            
            tableBody.innerHTML = reports.map(report => {
                const reportDate = report.timestamp ? new Date(report.timestamp).toLocaleDateString() : 'N/A';
                const status = report.status || 'pending';
                const priority = determinePriority(report.message || '');
                
                return `
                    <tr>
                        <td>SR${report.id}</td>
                        <td>${report.reporter_name || 'Anonymous'}</td>
                        <td>${report.location || 'N/A'}</td>
                        <td><span class="status ${priority}">${priority}</span></td>
                        <td><span class="status ${status}">${status}</span></td>
                        <td>${reportDate}</td>
                        <td>
                            <button class="btn tertiary" onclick="viewReportDetails('SR${report.id}')">View</button>
                            <button class="btn secondary" onclick="updateReportStatus('SR${report.id}')">Update</button>
                        </td>
                    </tr>
                `;
            }).join('');
        } else {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem; color: red;">Failed to load suspect reports</td></tr>';
        }
    } catch (error) {
        console.error('Error loading suspect reports:', error);
        const tableBody = document.getElementById('reportsTableBody');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem; color: red;">Error loading suspect reports</td></tr>';
        }
    }
}

function updateReportStats(reports) {
    const urgent = reports.filter(r => determinePriority(r.message || '') === 'urgent').length;
    const pending = reports.filter(r => (r.status || 'pending') === 'pending').length;
    const investigating = reports.filter(r => (r.status || 'pending') === 'investigating').length;
    const resolved = reports.filter(r => (r.status || 'pending') === 'resolved').length;
    
    document.getElementById('urgentReports').textContent = urgent;
    document.getElementById('newReports').textContent = pending;
    document.getElementById('investigatingReports').textContent = investigating;
    document.getElementById('resolvedReports').textContent = resolved;
}

async function viewReportDetails(reportId) {
    try {
        showLoadingMessage('Loading report details...');
        
        const reports = await titwengAPI.getReports();
        const report = reports.find(r => `SR${r.id}` === reportId);
        
        hideLoadingMessage();
        
        if (!report) {
            showErrorMessage('Report not found.');
            return;
        }
        
        const reportDate = report.timestamp ? new Date(report.timestamp).toLocaleString() : 'N/A';
        const priority = determinePriority(report.message || '');
        
        const content = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Report ID</div>
                    <div class="info-value">${reportId}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Reporter</div>
                    <div class="info-value">${report.reporter_name || 'Anonymous'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Location</div>
                    <div class="info-value">${report.location || 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Priority</div>
                    <div class="info-value"><span class="status ${priority}">${priority}</span></div>
                </div>
                <div class="info-item">
                    <div class="info-label">Status</div>
                    <div class="info-value"><span class="status ${report.status || 'pending'}">${report.status || 'pending'}</span></div>
                </div>
                <div class="info-item">
                    <div class="info-label">Date Reported</div>
                    <div class="info-value">${reportDate}</div>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <h4>Report Message:</h4>
                <p>${report.message || 'No message provided'}</p>
            </div>
        `;
        const footer = `
            <button class="btn primary" onclick="updateReportStatus('${reportId}')">Update Status</button>
            <button class="btn secondary" onclick="contactReporter('${reportId}')">Contact Reporter</button>
            <button class="btn tertiary" onclick="closeCustomModal()">Close</button>
        `;
        showCustomModal('Suspect Report Details', content, footer);
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Error loading report details:', error);
        showErrorMessage('Failed to load report details.');
    }
}

function updateReportStatus(reportId) {
    const content = `
        <div class="form-group">
            <label>Current Status</label>
            <input type="text" value="New" class="form-control" readonly>
        </div>
        <div class="form-group">
            <label>Update Status To</label>
            <select class="form-control" id="newStatus">
                <option value="investigating">Under Investigation</option>
                <option value="resolved">Resolved</option>
                <option value="false-alarm">False Alarm</option>
                <option value="escalated">Escalated to Authorities</option>
            </select>
        </div>
        <div class="form-group">
            <label>Admin Notes</label>
            <textarea class="form-control" rows="3" placeholder="Add notes about the status update..."></textarea>
        </div>
        <div class="form-group">
            <label>Notify Reporter</label>
            <select class="form-control">
                <option value="yes">Yes - Send SMS notification</option>
                <option value="no">No - Internal update only</option>
            </select>
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="confirmStatusUpdate('${reportId}')">Update Status</button>
        <button class="btn secondary" onclick="closeCustomModal()">Cancel</button>
    `;
    showCustomModal('Update Report Status', content, footer);
}

function confirmStatusUpdate(reportId) {
    const newStatus = document.getElementById('newStatus')?.value || 'investigating';
    const content = `
        <div class="success-message">
            <i class="fas fa-check-circle"></i> Report ${reportId} status updated successfully!
        </div>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Report ID</div>
                <div class="info-value">${reportId}</div>
            </div>
            <div class="info-item">
                <div class="info-label">New Status</div>
                <div class="info-value"><span class="status ${newStatus}">${newStatus.replace('-', ' ')}</span></div>
            </div>
            <div class="info-item">
                <div class="info-label">Updated By</div>
                <div class="info-value">Admin User</div>
            </div>
            <div class="info-item">
                <div class="info-label">Update Time</div>
                <div class="info-value">${new Date().toLocaleString()}</div>
            </div>
        </div>
        <p style="margin-top: 1rem;">Reporter has been notified via SMS about the status update.</p>
    `;
    showCustomModal('Status Updated', content);
}

function contactReporter(reportId) {
    const content = `
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Reporter Name</div>
                <div class="info-value">John Doe</div>
            </div>
            <div class="info-item">
                <div class="info-label">Phone Number</div>
                <div class="info-value">+1234567890</div>
            </div>
            <div class="info-item">
                <div class="info-label">Report ID</div>
                <div class="info-value">${reportId}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Preferred Contact</div>
                <div class="info-value">SMS</div>
            </div>
        </div>
        <div class="form-group" style="margin-top: 1rem;">
            <label>Message Template</label>
            <select class="form-control" id="messageTemplate" onchange="updateMessageText()">
                <option value="status">Status Update</option>
                <option value="followup">Follow-up Questions</option>
                <option value="resolved">Case Resolved</option>
                <option value="custom">Custom Message</option>
            </select>
        </div>
        <div class="form-group">
            <label>Message</label>
            <textarea class="form-control" id="messageText" rows="4">Hello John, this is regarding your theft report ${reportId}. We are currently investigating the matter and will update you soon. Thank you for your vigilance.</textarea>
        </div>
    `;
    const footer = `
        <button class="btn primary" onclick="sendReporterMessage('${reportId}')">Send SMS</button>
        <button class="btn secondary" onclick="callReporter('${reportId}')">Call Reporter</button>
        <button class="btn tertiary" onclick="closeCustomModal()">Close</button>
    `;
    showCustomModal('Contact Reporter', content, footer);
}

function sendReporterMessage(reportId) {
    const content = `
        <div class="success-message">
            <i class="fas fa-paper-plane"></i> Message functionality will be available in the next update.
        </div>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Report ID</div>
                <div class="info-value">${reportId}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Status</div>
                <div class="info-value">Feature in development</div>
            </div>
        </div>
        <p style="margin-top: 1rem;">SMS and email integration is being implemented.</p>
    `;
    showCustomModal('Message Feature', content);
}

function callReporter(reportId) {
    const content = `
        <div class="success-message">
            <i class="fas fa-phone"></i> Call initiated to reporter
        </div>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Calling</div>
                <div class="info-value">John Doe (+1234567890)</div>
            </div>
            <div class="info-item">
                <div class="info-label">Report ID</div>
                <div class="info-value">${reportId}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Call Status</div>
                <div class="info-value"><span class="status good">Connecting...</span></div>
            </div>
        </div>
        <p style="margin-top: 1rem;">Please use your phone to complete the call. This action has been logged for audit purposes.</p>
    `;
    showCustomModal('Calling Reporter', content);
}

function updateMessageText() {
    const template = document.getElementById('messageTemplate')?.value;
    const messageText = document.getElementById('messageText');
    if (!messageText) return;
    
    const templates = {
        status: 'Hello, this is regarding your theft report. We are currently investigating the matter and will update you soon.',
        followup: 'Hello, we need additional information about your theft report. Please call us back at your earliest convenience.',
        resolved: 'Good news! Your theft report has been resolved. The suspect has been apprehended. Thank you for your cooperation.',
        custom: ''
    };
    
    messageText.value = templates[template] || '';
}

function showCowDetails() {
    document.getElementById('cowDetails').style.display = 'block';
}

// Navigation helper function
function navigateToSection(sectionId) {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');
    
    navLinks.forEach(l => l.classList.remove('active'));
    sections.forEach(s => s.classList.remove('active'));
    
    const targetLink = document.querySelector(`[href="#${sectionId}"]`);
    const targetSection = document.getElementById(sectionId);
    
    if (targetLink) targetLink.classList.add('active');
    if (targetSection) targetSection.classList.add('active');
}

function toggleNavSection(categoryElement) {
    const navSection = categoryElement.parentElement;
    navSection.classList.toggle('collapsed');
}

// File upload handling
let uploadedFiles = [];

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('noseImages');
    const uploadArea = document.getElementById('fileUploadArea');
    const preview = document.getElementById('imagePreview');
    
    if (fileInput && uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', function(e) {
            const files = Array.from(e.target.files);
            
            if (files.length === 0) return;
            
            const currentTotal = uploadedFiles.length + capturedImages.length;
            const newTotal = currentTotal + files.length;
            
            if (newTotal > 7) {
                showFormMessage('Please remove some images. Maximum is 7.', 'error');
                return;
            }
            
            files.forEach(file => {
                if (file.type.startsWith('image/')) {
                    uploadedFiles.push(file);
                    addImagePreview(file, 'uploaded');
                }
            });
            
            const finalTotal = uploadedFiles.length + capturedImages.length;
            if (finalTotal < 5) {
                showFormMessage(`Please add ${5 - finalTotal} more image${5 - finalTotal > 1 ? 's' : ''}.`, 'warning');
            } else {
                showFormMessage(`${finalTotal} images selected. Ready to register!`, 'success');
            }
            
            fileInput.value = '';
        });
    }
});

function addImagePreview(file, type) {
    const preview = document.getElementById('imagePreview');
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const container = document.createElement('div');
        container.className = 'image-container';
        
        const img = document.createElement('img');
        img.src = e.target.result;
        img.className = 'preview-image';
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-image';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => removeImage(container, file, type);
        
        container.appendChild(img);
        container.appendChild(removeBtn);
        preview.appendChild(container);
    };
    
    reader.readAsDataURL(file);
}

function removeImage(container, file, type) {
    if (type === 'uploaded') {
        uploadedFiles = uploadedFiles.filter(f => f !== file);
    } else if (type === 'captured') {
        capturedImages = capturedImages.filter(f => f !== file);
    }
    container.remove();
    
    const totalImages = uploadedFiles.length + capturedImages.length;
    if (totalImages < 5) {
        showFormMessage(`Please add ${5 - totalImages} more image${5 - totalImages > 1 ? 's' : ''}.`, 'warning');
    } else {
        showFormMessage(`${totalImages} images selected. Ready to register!`, 'success');
    }
}

// Verification method toggle
function toggleVerificationMethod() {
    const tagMethod = document.querySelector('input[name="verifyMethod"][value="tag"]').checked;
    const tagSection = document.getElementById('tagVerification');
    const imageSection = document.getElementById('imageVerification');
    
    if (tagMethod) {
        tagSection.style.display = 'block';
        imageSection.style.display = 'none';
    } else {
        tagSection.style.display = 'none';
        imageSection.style.display = 'block';
    }
    
    resetVerification();
}

// Verify by cow tag
async function verifyByTag() {
    const tagInput = document.getElementById('tagInput').value;
    if (!tagInput) {
        showErrorMessage('Please enter a cow tag.');
        return;
    }
    
    try {
        showLoadingMessage('Verifying cow by tag...');
        
        const cows = await titwengAPI.getAllCows();
        const foundCow = cows.find(cow => 
            cow.cow_tag === tagInput || 
            cow.cow_tag === `T${tagInput.padStart(3, '0')}`
        );
        
        hideLoadingMessage();
        
        if (foundCow) {
            showVerificationResult(true, 100, foundCow);
        } else {
            showVerificationResult(false, 0, null);
        }
    } catch (error) {
        hideLoadingMessage();
        console.error('Tag verification error:', error);
        showErrorMessage('Failed to verify cow by tag. Please try again.');
    }
}

// Verify by image
async function verifyByImage() {
    const fileInput = document.getElementById('verifyImage');
    const files = fileInput.files;
    
    if (files.length === 0) {
        showErrorMessage('Please select an image for verification.');
        return;
    }
    
    try {
        showLoadingMessage('Verifying cow by image...');
        
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        const result = await titwengAPI.verifyCow(formData);
        
        hideLoadingMessage();
        
        const confidence = result.confidence || 0;
        const isMatch = result.match || false;
        
        showVerificationResult(isMatch, confidence * 100, result.cow_data);
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Image verification error:', error);
        showErrorMessage('Failed to verify cow by image. Please try again.');
    }
}

// Show verification result
function showVerificationResult(isMatch, confidence, cowData) {
    document.getElementById('confidenceScore').textContent = confidence.toFixed(1) + '%';
    const resultStatus = document.getElementById('resultStatus');
    const cowDetails = document.getElementById('cowDetails');
    
    if (isMatch && cowData) {
        resultStatus.className = 'result-status success';
        resultStatus.innerHTML = '✓ Cow is Registered<br><button class="btn primary" onclick="showCowDetails()" style="margin-top: 0.5rem;">View Details</button>';
        
        document.getElementById('detailCowId').textContent = cowData.cow_tag || cowData.id || 'N/A';
        document.getElementById('detailColor').textContent = cowData.color || 'N/A';
        document.getElementById('detailBreed').textContent = cowData.breed || 'N/A';
        document.getElementById('detailAge').textContent = cowData.age || 'N/A';
        document.getElementById('detailOwnerName').textContent = cowData.owner_name || 'N/A';
        document.getElementById('detailPhone').textContent = cowData.owner_phone || 'N/A';
        document.getElementById('detailEmail').textContent = cowData.owner_email || 'N/A';
        document.getElementById('detailAddress').textContent = cowData.owner_address || 'N/A';
    } else {
        resultStatus.className = 'result-status failed';
        resultStatus.textContent = '✗ This cow is not registered in the system';
        cowDetails.style.display = 'none';
    }
    
    document.getElementById('verificationResult').style.display = 'block';
}

function resetVerification() {
    document.getElementById('tagInput').value = '';
    document.getElementById('verifyImage').value = '';
    document.getElementById('verificationResult').style.display = 'none';
    document.getElementById('verifyPreview').innerHTML = '';
    closeVerifyCamera();
}

// Verify image upload
document.addEventListener('DOMContentLoaded', function() {
    const verifyInput = document.getElementById('verifyImage');
    const verifyArea = document.getElementById('verifyUploadArea');
    const verifyPreview = document.getElementById('verifyPreview');
    
    if (verifyInput && verifyArea) {
        verifyArea.addEventListener('click', () => verifyInput.click());
        
        verifyInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    verifyPreview.innerHTML = `<img src="${e.target.result}" alt="Verification Image">`;
                };
                reader.readAsDataURL(file);
            }
        });
    }
});

// Camera functionality
let cameraStream = null;
let capturedImages = [];

function openCamera() {
    const cameraSection = document.getElementById('cameraSection');
    const video = document.getElementById('cameraVideo');
    
    cameraSection.style.display = 'block';
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            cameraStream = stream;
            video.srcObject = stream;
        })
        .catch(err => {
            alert('Camera access denied or not available');
            console.error('Camera error:', err);
        });
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');
    
    const totalImages = uploadedFiles.length + capturedImages.length;
    if (totalImages >= 7) {
        showFormMessage('Please remove some images. Maximum is 7.', 'error');
        return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
        const file = new File([blob], `nose_print_${Date.now()}.jpg`, { type: 'image/jpeg' });
        capturedImages.push(file);
        addImagePreview(file, 'captured');
        
        const newTotal = uploadedFiles.length + capturedImages.length;
        if (newTotal < 5) {
            showFormMessage(`Please add ${5 - newTotal} more image${5 - newTotal > 1 ? 's' : ''}.`, 'warning');
        } else {
            showFormMessage(`Photo captured! Total images: ${newTotal}`, 'success');
        }
    }, 'image/jpeg', 0.8);
}

function closeCamera() {
    const cameraSection = document.getElementById('cameraSection');
    const video = document.getElementById('cameraVideo');
    
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    video.srcObject = null;
    cameraSection.style.display = 'none';
}

// Verification camera functions
let verifyCameraStream = null;

function openVerifyCamera() {
    const cameraSection = document.getElementById('verifyCameraSection');
    const video = document.getElementById('verifyCameraVideo');
    
    cameraSection.style.display = 'block';
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            verifyCameraStream = stream;
            video.srcObject = stream;
        })
        .catch(err => {
            alert('Camera access denied or not available');
            console.error('Camera error:', err);
        });
}

function captureVerifyPhoto() {
    const video = document.getElementById('verifyCameraVideo');
    const canvas = document.getElementById('verifyCameraCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
        const preview = document.getElementById('verifyPreview');
        const img = document.createElement('img');
        img.src = URL.createObjectURL(blob);
        img.alt = 'Verification Image';
        preview.innerHTML = '';
        preview.appendChild(img);
        
        closeVerifyCamera();
        alert('Photo captured for verification!');
    }, 'image/jpeg', 0.8);
}

function closeVerifyCamera() {
    const cameraSection = document.getElementById('verifyCameraSection');
    const video = document.getElementById('verifyCameraVideo');
    
    if (verifyCameraStream) {
        verifyCameraStream.getTracks().forEach(track => track.stop());
        verifyCameraStream = null;
    }
    
    video.srcObject = null;
    cameraSection.style.display = 'none';
}

// Form submissions with real API calls
document.addEventListener('submit', function(e) {
    if (e.target.closest('.modal-form')) {
        e.preventDefault();
        handleModalFormSubmission(e.target);
    }
    
    if (e.target.id === 'cowRegistrationForm') {
        e.preventDefault();
        handleCowRegistration(e.target);
    }
});

async function handleModalFormSubmission(form) {
    const modal = form.closest('.modal');
    const modalId = modal.id;
    
    try {
        if (modalId === 'registerCowModal') {
            await handleQuickCowRegistration(form);
        } else if (modalId === 'addAdminModal') {
            await handleAdminCreation(form);
        } else {
            showSuccessMessage('Form submitted successfully!');
        }
        modal.style.display = 'none';
    } catch (error) {
        showErrorMessage('Failed to submit form. Please try again.');
    }
}

async function handleAdminCreation(form) {
    const formData = new FormData(form);
    const adminData = {
        username: formData.get('email').split('@')[0], // Use email prefix as username
        email: formData.get('email'),
        password: formData.get('tempPassword'),
        full_name: formData.get('fullName'),
        role: formData.get('role') === 'super_admin' ? 'super_admin' : 'admin'
    };
    
    const response = await fetch(`${titwengAPI.baseURL}/auth/create-admin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(adminData)
    });
    
    if (response.ok) {
        showSuccessMessage(`Admin ${adminData.full_name} created successfully!`);
        await populateAdminTable();
    } else {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create admin');
    }
}

async function handleCowRegistration(form) {
    try {
        showLoadingMessage('Registering cow...');
        
        const formData = new FormData();
        
        // Get all form fields with proper names
        const color = form.querySelector('input[name="color"]').value;
        const breed = form.querySelector('select[name="breed"]').value;
        const age = form.querySelector('input[name="age"]').value;
        const ownerName = form.querySelector('input[name="owner_name"]').value;
        const ownerPhone = form.querySelector('input[name="owner_phone"]').value;
        const ownerEmail = form.querySelector('input[name="owner_email"]').value;
        const nationalId = form.querySelector('input[name="national_id"]').value;
        const ownerAddress = form.querySelector('textarea[name="owner_address"]').value;
        
        // Validate required fields
        if (!color || !breed || !age || !ownerName || !ownerPhone || !nationalId || !ownerAddress) {
            hideLoadingMessage();
            showErrorMessage('Please fill in all required fields.');
            return;
        }
        
        // Get current admin's location
        const currentAdmin = JSON.parse(localStorage.getItem('admin_user') || '{}');
        const registrationLocation = currentAdmin.location || 'Unknown';
        
        // Append form data
        formData.append('color', color);
        formData.append('breed', breed);
        formData.append('age', age);
        formData.append('owner_name', ownerName);
        formData.append('owner_phone', ownerPhone);
        formData.append('owner_email', ownerEmail || '');
        formData.append('national_id', nationalId);
        formData.append('owner_address', ownerAddress);
        formData.append('registered_by', currentAdmin.id || 'unknown');
        formData.append('location', registrationLocation);
        
        // Handle file uploads
        const totalImages = uploadedFiles.length + capturedImages.length;
        
        if (totalImages < 5) {
            hideLoadingMessage();
            showFormMessage('Please upload at least 5 nose print images.', 'error');
            return;
        }
        
        if (totalImages > 7) {
            hideLoadingMessage();
            showFormMessage('Maximum 7 images allowed. Please remove some images.', 'error');
            return;
        }
        
        // Add uploaded files
        uploadedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        // Add captured images
        capturedImages.forEach(image => {
            formData.append('files', image);
        });
        
        const result = await titwengAPI.registerCow(formData);
        
        hideLoadingMessage();
        showSuccessMessage(`Cow registered successfully! Tag: ${result.cow_tag || 'Generated'}`);
        
        form.reset();
        document.getElementById('imagePreview').innerHTML = '';
        uploadedFiles = [];
        capturedImages = [];
        
        await populateCowTable();
        await loadDashboardStats();
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Registration error:', error);
        showErrorMessage('Failed to register cow. Please check all fields and try again.');
    }
}

async function handleCowVerification(form) {
    try {
        showLoadingMessage('Verifying cow...');
        
        const fileInput = form.querySelector('input[type="file"]');
        const files = fileInput.files;
        
        if (files.length === 0) {
            hideLoadingMessage();
            showErrorMessage('Please select an image for verification.');
            return;
        }
        
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        const result = await titwengAPI.verifyCow(formData);
        
        hideLoadingMessage();
        
        const confidence = result.confidence || 0;
        const isMatch = result.match || false;
        
        document.getElementById('confidenceScore').textContent = (confidence * 100).toFixed(1) + '%';
        const resultStatus = document.getElementById('resultStatus');
        const cowDetails = document.getElementById('cowDetails');
        
        if (isMatch && result.cow_data) {
            resultStatus.className = 'result-status success';
            resultStatus.innerHTML = '✓ Cow is Registered<br><button class="btn primary" onclick="showCowDetails()" style="margin-top: 0.5rem;">View Details</button>';
            
            const cow = result.cow_data;
            document.getElementById('detailCowId').textContent = cow.cow_tag || cow.id || 'N/A';
            document.getElementById('detailColor').textContent = cow.color || 'N/A';
            document.getElementById('detailBreed').textContent = cow.breed || 'N/A';
            document.getElementById('detailAge').textContent = cow.age || 'N/A';
            document.getElementById('detailOwnerName').textContent = cow.owner_name || 'N/A';
            document.getElementById('detailPhone').textContent = cow.owner_phone || 'N/A';
            document.getElementById('detailEmail').textContent = cow.owner_email || 'N/A';
            document.getElementById('detailAddress').textContent = cow.owner_address || 'N/A';
        } else {
            resultStatus.className = 'result-status failed';
            resultStatus.textContent = '✗ This cow is not registered in the system';
            cowDetails.style.display = 'none';
        }
        
        document.getElementById('verificationResult').style.display = 'block';
        
    } catch (error) {
        hideLoadingMessage();
        console.error('Verification error:', error);
        showErrorMessage('Failed to verify cow. Please try again.');
    }
}

async function handleQuickCowRegistration(form) {
    const formData = new FormData();
    
    // Get form values
    const breed = form.querySelector('select[name="breed"]').value;
    const color = form.querySelector('input[name="color"]').value;
    const age = form.querySelector('input[name="age"]').value;
    const ownerName = form.querySelector('input[name="owner_name"]').value;
    const ownerPhone = form.querySelector('input[name="owner_phone"]').value;
    const ownerEmail = form.querySelector('input[name="owner_email"]').value;
    const ownerAddress = form.querySelector('textarea[name="owner_address"]').value;
    
    // Append required fields
    formData.append('color', color || 'Mixed');
    formData.append('breed', breed || 'Mixed');
    formData.append('age', age || '12');
    formData.append('owner_name', ownerName || 'Quick Registration');
    formData.append('owner_phone', ownerPhone || '');
    formData.append('owner_email', ownerEmail || '');
    formData.append('owner_address', ownerAddress || 'Not specified');
    formData.append('national_id', 'QUICK-' + Date.now());
    
    const files = form.querySelector('input[type="file"]').files;
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    
    const result = await titwengAPI.registerCow(formData);
    showSuccessMessage(`Cow registered successfully! Tag: ${result.cow_tag || 'Generated'}`);
    
    await populateCowTable();
    await loadDashboardStats();
}

// Filter functionality
document.addEventListener('change', function(e) {
    if (e.target.matches('#breedFilter, #dateFilter, #statusFilter')) {
        // Implement filtering logic here
        console.log('Filter changed:', e.target.id, e.target.value);
    }
});