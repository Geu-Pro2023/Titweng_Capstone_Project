# Titweng Capstone Project - Database Demo Script

## For Professor Presentation

### 1. **Show Live Database Connection** (30 seconds)
- Open: https://titweng-capstone-project.onrender.com/health
- **Point out:** `"database_connected": true`
- **Explain:** "This confirms our PostgreSQL database is live on Render cloud"

### 2. **Show Database Statistics** (1 minute)
- Open: https://titweng-capstone-project.onrender.com/admin/dashboard
- **Show JSON response:**
  ```json
  {
    "total_cows": 3,
    "total_owners": 3, 
    "pending_reports": 2
  }
  ```
- **Explain:** "Real data from our live PostgreSQL database"

### 3. **Show Detailed Cow Data** (2 minutes)
- Open: https://titweng-capstone-project.onrender.com/admin/cows
- **Point out:**
  - Cow tags (TWAB0001, etc.)
  - Owner information
  - Registration dates
  - Embeddings count (ML vectors)
- **Explain:** "Each cow has 5-7 nose print embeddings for Siamese CNN"

### 4. **Show Admin Dashboard** (3 minutes)
- Login: https://titweng-admin.netlify.app
- **Credentials:** titweng/titweng@2025
- **Navigate through:**
  - Dashboard (statistics)
  - Cow Management (table view)
  - Owner Management
  - Reports section
- **Explain:** "Dashboard pulls real-time data from our database"

### 5. **Show Database Schema** (2 minutes)
- Open: `DATABASE_SCHEMA.md` file
- **Highlight:**
  - 7 tables with relationships
  - pgvector for ML embeddings
  - Foreign key constraints
  - Data types and structure

### 6. **Demonstrate Data Flow** (2 minutes)
- **Register a new cow** through dashboard
- **Show:** Data appears immediately in database
- **Explain:** "Data flows: Dashboard → API → PostgreSQL → Persistent storage"

## Key Points to Emphasize:
✅ **Real cloud database** (not local/fake)
✅ **Live data** with proper relationships
✅ **ML integration** (vector embeddings)
✅ **Professional structure** with constraints
✅ **Scalable architecture** (cloud-hosted)
✅ **Data persistence** (survives restarts)

## Backup Demo (if internet issues):
- Show local code files
- Explain database models
- Show API endpoints in code
- Reference health check screenshot