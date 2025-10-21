# Titweng Capstone Project - Database Schema

## Database Overview
- **Platform:** PostgreSQL on Render Cloud
- **Extensions:** pgvector (for ML embeddings)
- **Total Tables:** 7

## Table Structure

### 1. `owners` - Cattle Owners
| Column | Type | Description |
|--------|------|-------------|
| owner_id | INTEGER (PK) | Unique owner identifier |
| full_name | VARCHAR(100) | Owner's full name |
| email | VARCHAR(120) | Contact email |
| phone | VARCHAR(20) | Phone number |
| address | TEXT | Physical address |
| national_id | VARCHAR(50) | National ID number |
| created_at | TIMESTAMP | Registration date |

### 2. `cows` - Registered Cattle
| Column | Type | Description |
|--------|------|-------------|
| cow_id | INTEGER (PK) | Unique cow identifier |
| cow_tag | VARCHAR(50) (UNIQUE) | Secure cow tag (e.g., TWAB0001) |
| breed | VARCHAR(100) | Cattle breed |
| color | VARCHAR(50) | Cattle color |
| age | INTEGER | Age in months |
| owner_id | INTEGER (FK) | Links to owners table |
| qr_code_link | TEXT | QR code URL |
| receipt_pdf_link | TEXT | PDF receipt path |
| registration_date | TIMESTAMP | When registered |

### 3. `embeddings` - Siamese CNN Vectors
| Column | Type | Description |
|--------|------|-------------|
| embedding_id | INTEGER (PK) | Unique embedding ID |
| cow_id | INTEGER (FK) | Links to cows table |
| embedding | VECTOR(256) | 256-dimensional ML vector |
| image_path | TEXT | Image file path |
| image_data | BYTEA | Raw image bytes |
| created_at | TIMESTAMP | When created |

### 4. `users` - Mobile App Users
| Column | Type | Description |
|--------|------|-------------|
| user_id | INTEGER (PK) | Unique user ID |
| username | VARCHAR(100) | Username |
| email | VARCHAR(120) | Email address |
| phone | VARCHAR(20) | Phone number |
| password_hash | TEXT | Encrypted password |
| role | VARCHAR(20) | User role |
| created_at | TIMESTAMP | Registration date |

### 5. `reports` - Theft/Suspect Reports
| Column | Type | Description |
|--------|------|-------------|
| report_id | INTEGER (PK) | Unique report ID |
| user_id | INTEGER (FK) | Reporter user ID |
| report_type | VARCHAR(50) | Type of report |
| description | TEXT | Report description |
| location | VARCHAR(200) | Incident location |
| status | VARCHAR(20) | pending/investigating/resolved |
| created_at | TIMESTAMP | Report date |

### 6. `verification_logs` - ML Verification History
| Column | Type | Description |
|--------|------|-------------|
| log_id | INTEGER (PK) | Unique log ID |
| user_id | INTEGER (FK) | User who verified |
| cow_id | INTEGER (FK) | Cow being verified |
| similarity_score | FLOAT | ML confidence score |
| verified | BOOLEAN | Verification result |
| created_at | TIMESTAMP | Verification time |

### 7. `admins` - Dashboard Authentication
| Column | Type | Description |
|--------|------|-------------|
| admin_id | INTEGER (PK) | Unique admin ID |
| username | VARCHAR | Login username |
| email | VARCHAR | Admin email |
| password_hash | VARCHAR | Encrypted password |
| full_name | VARCHAR | Admin's full name |
| role | VARCHAR | admin/super_admin |
| is_active | BOOLEAN | Account status |
| created_at | TIMESTAMP | Account creation |
| last_login | TIMESTAMP | Last login time |

## Key Features
- **Foreign Key Constraints:** Maintain data integrity
- **Cascade Deletes:** Clean up related data automatically
- **Vector Storage:** pgvector extension for ML embeddings
- **Binary Storage:** Store nose print images directly
- **Timestamps:** Track all data creation/modification
- **Unique Constraints:** Prevent duplicate cow tags/emails

## Sample Data
- **3 Registered Cows** with nose print embeddings
- **3 Cattle Owners** with contact information
- **2 Suspect Reports** from mobile app users
- **1 Admin Account** for dashboard access

## Live Database URL
- **API:** https://titweng-capstone-project.onrender.com
- **Dashboard:** https://titweng-admin.netlify.app
- **Health Check:** https://titweng-capstone-project.onrender.com/health