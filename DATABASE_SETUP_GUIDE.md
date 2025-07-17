# Malnutrition Assessment System - Database Integration Guide

## 🎯 Overview
This enhanced malnutrition assessment system now includes comprehensive database integration with multi-user support, authentication, and analytics capabilities.

## 🚀 Quick Start

### 1. Install Database Dependencies
```bash
pip install psycopg2-binary mysql-connector-python bcrypt python-dotenv
```

### 2. Initialize Database
Run the setup script to create the database with sample data:
```bash
python setup_database.py
```

### 3. Launch Enhanced Application
```bash
streamlit run app_with_database.py
```

## 🔑 Demo Login Credentials

### Admin Access
- **Username:** `admin`
- **Password:** `admin123`
- **Capabilities:** Full system access, user management, analytics

### Nutritionist Access
- **Username:** `dr_maria`
- **Password:** `nutri123`
- **Capabilities:** Patient management, assessments, treatment protocols

### Parent Access
- **Username:** `juan_dela_cruz`
- **Password:** `parent123`
- **Capabilities:** View children's records, assessment history

## 📊 Database Features

### Multi-Database Support
- **SQLite:** Default for development and testing
- **PostgreSQL:** Production-ready with advanced features
- **MySQL:** Enterprise-grade with high performance

### User Management
- Role-based access control (Admin, Nutritionist, Parent)
- Secure password hashing with bcrypt
- User authentication and session management

### Patient Records
- Complete patient demographics
- Parent-child relationships
- Nutritionist assignments
- Assessment history tracking

### Assessment Storage
- Full nutritional assessment data
- WHO protocol compliance
- Treatment recommendations
- Progress tracking
- Analytics and reporting

## 🏗️ System Architecture

### Database Layer (`database_manager.py`)
- Multi-database abstraction
- User authentication
- Patient management
- Assessment storage
- Analytics engine

### Business Logic (`enhanced_treatment_protocols.py`)
- WHO treatment protocols
- Age-specific considerations
- Risk assessment
- Treatment recommendations

### Presentation Layer (`app_with_database.py`)
- Role-based dashboards
- Interactive assessments
- Data visualization
- User management

## 📋 Database Schema

### Users Table
- user_id (Primary Key)
- username (Unique)
- email
- password_hash
- role (admin/nutritionist/parent)
- full_name
- created_at

### Patients Table
- patient_id (Primary Key)
- full_name
- date_of_birth
- sex
- municipality
- parent_user_id (Foreign Key)
- assigned_nutritionist_id (Foreign Key)
- created_at

### Assessments Table
- assessment_id (Primary Key)
- patient_id (Foreign Key)
- assessed_by_user_id (Foreign Key)
- assessment_date
- Complete nutritional data
- Treatment recommendations
- Notes and observations

## 🔧 Configuration

### Environment Variables (.env)
```env
# Database Configuration
DB_TYPE=sqlite
DB_HOST=localhost
DB_PORT=5432
DB_NAME=malnutrition_assessment
DB_USER=username
DB_PASSWORD=password

# Security
SECRET_KEY=your_secret_key_here
```

### SQLite Configuration (Default)
- Database file: `malnutrition_assessment.db`
- No additional setup required
- Perfect for development and testing

### PostgreSQL Configuration
```python
db = DatabaseManager(
    db_type='postgresql',
    host='localhost',
    port=5432,
    database='malnutrition_assessment',
    user='your_username',
    password='your_password'
)
```

### MySQL Configuration
```python
db = DatabaseManager(
    db_type='mysql',
    host='localhost',
    port=3306,
    database='malnutrition_assessment',
    user='your_username',
    password='your_password'
)
```

## 👥 User Roles & Permissions

### Admin Role
- ✅ Full system access
- ✅ User management
- ✅ System analytics
- ✅ Database administration
- ✅ All assessment data

### Nutritionist Role
- ✅ Patient management
- ✅ Assessment creation/editing
- ✅ Treatment protocols
- ✅ Assigned patients only
- ✅ Clinical reports

### Parent Role
- ✅ View children's records
- ✅ Assessment history
- ✅ Treatment updates
- ❌ Cannot modify assessments
- ❌ Limited to own children

## 📊 Analytics Features

### System Statistics
- Total users by role
- Patient demographics
- Assessment trends
- Nutritional status distribution
- MAM/SAM prevalence rates

### Clinical Insights
- Growth trajectory analysis
- Treatment outcome tracking
- Risk factor correlations
- Population health metrics
- Intervention effectiveness

## 🔒 Security Features

### Authentication
- Secure password hashing (bcrypt)
- Session management
- Role-based access control
- Input validation and sanitization

### Data Protection
- SQL injection prevention
- Parameterized queries
- Access logging
- Error handling without data exposure

## 🚀 Deployment Options

### Local Development
1. Use SQLite (default)
2. Run setup script
3. Launch Streamlit app

### Production Deployment
1. Configure PostgreSQL/MySQL
2. Set environment variables
3. Run database migrations
4. Deploy with appropriate hosting

### Cloud Deployment
- AWS RDS for database
- Heroku/Railway for application
- Environment-based configuration

## 🔄 Data Migration

### From Existing System
```python
# Example migration script
def migrate_existing_data():
    db = DatabaseManager()
    
    # Import existing assessments
    for assessment in old_assessments:
        db.save_assessment(assessment)
    
    # Create user accounts
    for user in existing_users:
        db.create_user(user)
```

### Backup and Restore
```bash
# SQLite backup
cp malnutrition_assessment.db backup_$(date +%Y%m%d).db

# PostgreSQL backup
pg_dump malnutrition_assessment > backup_$(date +%Y%m%d).sql
```

## 📈 Performance Optimization

### Database Indexing
- User lookups by username/email
- Patient queries by parent/nutritionist
- Assessment queries by date/patient
- Status-based filtering

### Query Optimization
- Efficient joins for related data
- Pagination for large datasets
- Caching for frequent queries
- Connection pooling

## 🛠️ Troubleshooting

### Common Issues

#### Database Connection Error
```bash
# Check database status
systemctl status postgresql  # For PostgreSQL
systemctl status mysql       # For MySQL
```

#### Permission Denied
```bash
# Check file permissions
chmod 664 malnutrition_assessment.db
```

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements_database.txt
```

### Error Logs
Check application logs for detailed error information:
```python
# Enable logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For technical support or feature requests:
1. Check the troubleshooting guide
2. Review error logs
3. Contact system administrator
4. Submit issue to development team

---

## 🎉 Success!
Your malnutrition assessment system is now enhanced with full database integration, multi-user support, and comprehensive analytics capabilities!
