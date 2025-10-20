// API Configuration for Titweng Dashboard
const API_CONFIG = {
    BASE_URL: 'https://titweng-capstone-project.onrender.com',
    ENDPOINTS: {
        // Admin Dashboard
        DASHBOARD: '/admin/dashboard',
        REPORTS: '/admin/reports', 
        COWS: '/admin/cows',
        UPDATE_COW: '/admin/cows/{cow_tag}',
        DELETE_COW: '/admin/cows/{cow_tag}',
        
        // Cow Management
        REGISTER_COW: '/register-cow',
        VERIFY_COW: '/verify-cow',
        PREVIEW_NEXT_TAG: '/preview-next-cow-tag',
        
        // Mobile App
        MOBILE_VERIFY: '/mobile/verify-cow',
        SUBMIT_REPORT: '/submit-report',
        
        // Utility
        GET_IMAGE: '/image/{embedding_id}',
        DOWNLOAD_RECEIPT: '/download-receipt/{cow_tag}',
        
        // Status
        HEALTH: '/health',
        ROOT: '/'
    }
};

// API Helper Functions
class TitwengAPI {
    constructor() {
        this.baseURL = API_CONFIG.BASE_URL;
    }

    // Generic API call method
    async apiCall(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const finalOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }

    // Dashboard Statistics
    async getDashboardStats() {
        return await this.apiCall(API_CONFIG.ENDPOINTS.DASHBOARD);
    }

    // Get all reports
    async getReports() {
        return await this.apiCall(API_CONFIG.ENDPOINTS.REPORTS);
    }

    // Get all cows
    async getAllCows() {
        return await this.apiCall(API_CONFIG.ENDPOINTS.COWS);
    }

    // Register new cow
    async registerCow(formData) {
        return await this.apiCall(API_CONFIG.ENDPOINTS.REGISTER_COW, {
            method: 'POST',
            body: formData,
            headers: {} // Remove Content-Type to let browser set it for FormData
        });
    }

    // Verify cow
    async verifyCow(formData) {
        return await this.apiCall(API_CONFIG.ENDPOINTS.VERIFY_COW, {
            method: 'POST',
            body: formData,
            headers: {} // Remove Content-Type to let browser set it for FormData
        });
    }

    // Update cow
    async updateCow(cowTag, formData) {
        const endpoint = API_CONFIG.ENDPOINTS.UPDATE_COW.replace('{cow_tag}', cowTag);
        return await this.apiCall(endpoint, {
            method: 'PUT',
            body: formData,
            headers: {} // Remove Content-Type to let browser set it for FormData
        });
    }

    // Delete cow
    async deleteCow(cowTag) {
        const endpoint = API_CONFIG.ENDPOINTS.DELETE_COW.replace('{cow_tag}', cowTag);
        return await this.apiCall(endpoint, {
            method: 'DELETE'
        });
    }

    // Preview next cow tag
    async previewNextCowTag() {
        return await this.apiCall(API_CONFIG.ENDPOINTS.PREVIEW_NEXT_TAG);
    }

    // Submit report
    async submitReport(reportData) {
        return await this.apiCall(API_CONFIG.ENDPOINTS.SUBMIT_REPORT, {
            method: 'POST',
            body: new URLSearchParams(reportData),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });
    }

    // Get image by embedding ID
    async getImage(embeddingId) {
        const endpoint = API_CONFIG.ENDPOINTS.GET_IMAGE.replace('{embedding_id}', embeddingId);
        return await this.apiCall(endpoint);
    }

    // Download receipt
    async downloadReceipt(cowTag) {
        const endpoint = API_CONFIG.ENDPOINTS.DOWNLOAD_RECEIPT.replace('{cow_tag}', cowTag);
        const url = `${this.baseURL}${endpoint}`;
        window.open(url, '_blank');
    }

    // Health check
    async healthCheck() {
        return await this.apiCall(API_CONFIG.ENDPOINTS.HEALTH);
    }
}

// Create global API instance
const titwengAPI = new TitwengAPI();

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API_CONFIG, TitwengAPI, titwengAPI };
}