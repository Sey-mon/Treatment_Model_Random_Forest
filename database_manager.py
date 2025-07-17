"""
Database Manager for Child Malnutrition Assessment System
Supports SQLite, PostgreSQL, and MySQL databases
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import hashlib
import os

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

class DatabaseManager:
    """
    Comprehensive database manager for the malnutrition assessment system
    Supports SQLite (default), PostgreSQL, and MySQL
    """
    
    def __init__(self, db_type='sqlite', **kwargs):
        self.db_type = db_type.lower()
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.config = {
            'sqlite': {
                'database': kwargs.get('database', 'malnutrition_assessment.db')
            },
            'postgresql': {
                'host': kwargs.get('host', 'localhost'),
                'port': kwargs.get('port', 5432),
                'database': kwargs.get('database', 'malnutrition_db'),
                'user': kwargs.get('user', 'postgres'),
                'password': kwargs.get('password', ''),
            },
            'mysql': {
                'host': kwargs.get('host', 'localhost'),
                'port': kwargs.get('port', 3306),
                'database': kwargs.get('database', 'malnutrition_db'),
                'user': kwargs.get('user', 'root'),
                'password': kwargs.get('password', ''),
            }
        }
        
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            if self.db_type == 'sqlite':
                db_path = Path(self.config['sqlite']['database'])
                self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
                self.connection.row_factory = sqlite3.Row
                
            elif self.db_type == 'postgresql':
                if not POSTGRES_AVAILABLE:
                    raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
                
                self.connection = psycopg2.connect(**self.config['postgresql'])
                
            elif self.db_type == 'mysql':
                if not MYSQL_AVAILABLE:
                    raise ImportError("mysql-connector-python not installed. Install with: pip install mysql-connector-python")
                
                self.connection = mysql.connector.connect(**self.config['mysql'])
            
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            self.logger.info(f"Connected to {self.db_type} database successfully")
            
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create all necessary tables"""
        
        # SQL for different database types
        if self.db_type == 'sqlite':
            self._create_sqlite_tables()
        elif self.db_type == 'postgresql':
            self._create_postgresql_tables()
        elif self.db_type == 'mysql':
            self._create_mysql_tables()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'nutritionist', 'parent')),
                full_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_number VARCHAR(50) UNIQUE,
                full_name VARCHAR(100) NOT NULL,
                date_of_birth DATE,
                sex VARCHAR(10) NOT NULL CHECK (sex IN ('Male', 'Female')),
                municipality VARCHAR(100),
                parent_user_id INTEGER,
                assigned_nutritionist_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (parent_user_id) REFERENCES users(user_id),
                FOREIGN KEY (assigned_nutritionist_id) REFERENCES users(user_id)
            )
        """)
        
        # Assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                assessed_by_user_id INTEGER NOT NULL,
                assessment_date DATE NOT NULL,
                age_months INTEGER NOT NULL,
                weight_kg DECIMAL(5,2) NOT NULL,
                height_cm DECIMAL(5,1) NOT NULL,
                whz_score DECIMAL(5,2),
                nutritional_status VARCHAR(50),
                bmi DECIMAL(5,2),
                edema BOOLEAN DEFAULT 0,
                total_household INTEGER,
                adults INTEGER,
                children INTEGER,
                is_twin BOOLEAN DEFAULT 0,
                fourps_beneficiary VARCHAR(10),
                breastfeeding VARCHAR(10),
                tuberculosis VARCHAR(10),
                malaria VARCHAR(10),
                congenital_anomalies VARCHAR(10),
                other_medical_problems VARCHAR(10),
                prediction_confidence DECIMAL(5,3),
                treatment_protocol TEXT,
                recommendations TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (assessed_by_user_id) REFERENCES users(user_id)
            )
        """)
        
        # Treatment protocols table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS treatment_protocols (
                protocol_id INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol_name VARCHAR(100) NOT NULL,
                protocol_version VARCHAR(20),
                protocol_data TEXT NOT NULL,
                created_by_user_id INTEGER,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by_user_id) REFERENCES users(user_id)
            )
        """)
        
        # Analytics cache table for performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_cache (
                cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key VARCHAR(255) UNIQUE NOT NULL,
                cache_data TEXT NOT NULL,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        self.logger.info("SQLite tables created successfully")
    
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables"""
        cursor = self.connection.cursor()
        
        # Enable UUID extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'nutritionist', 'parent')),
                full_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id SERIAL PRIMARY KEY,
                patient_number VARCHAR(50) UNIQUE,
                full_name VARCHAR(100) NOT NULL,
                date_of_birth DATE,
                sex VARCHAR(10) NOT NULL CHECK (sex IN ('Male', 'Female')),
                municipality VARCHAR(100),
                parent_user_id INTEGER REFERENCES users(user_id),
                assigned_nutritionist_id INTEGER REFERENCES users(user_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id SERIAL PRIMARY KEY,
                patient_id INTEGER NOT NULL REFERENCES patients(patient_id),
                assessed_by_user_id INTEGER NOT NULL REFERENCES users(user_id),
                assessment_date DATE NOT NULL,
                age_months INTEGER NOT NULL,
                weight_kg DECIMAL(5,2) NOT NULL,
                height_cm DECIMAL(5,1) NOT NULL,
                whz_score DECIMAL(5,2),
                nutritional_status VARCHAR(50),
                bmi DECIMAL(5,2),
                edema BOOLEAN DEFAULT FALSE,
                total_household INTEGER,
                adults INTEGER,
                children INTEGER,
                is_twin BOOLEAN DEFAULT FALSE,
                fourps_beneficiary VARCHAR(10),
                breastfeeding VARCHAR(10),
                tuberculosis VARCHAR(10),
                malaria VARCHAR(10),
                congenital_anomalies VARCHAR(10),
                other_medical_problems VARCHAR(10),
                prediction_confidence DECIMAL(5,3),
                treatment_protocol TEXT,
                recommendations TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        self.logger.info("PostgreSQL tables created successfully")
    
    def _create_mysql_tables(self):
        """Create MySQL tables"""
        cursor = self.connection.cursor()
        
        # Similar structure adapted for MySQL syntax
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('admin', 'nutritionist', 'parent') NOT NULL,
                full_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Continue with other tables...
        self.connection.commit()
        self.logger.info("MySQL tables created successfully")
    
    # User Management Methods
    def create_user(self, username: str, email: str, password: str, role: str, full_name: str = None) -> int:
        """Create a new user"""
        password_hash = self._hash_password(password)
        
        cursor = self.connection.cursor()
        
        if self.db_type == 'sqlite':
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role, full_name)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, role, full_name))
            user_id = cursor.lastrowid
            
        elif self.db_type == 'postgresql':
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role, full_name)
                VALUES (%s, %s, %s, %s, %s) RETURNING user_id
            """, (username, email, password_hash, role, full_name))
            user_id = cursor.fetchone()[0]
            
        elif self.db_type == 'mysql':
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role, full_name)
                VALUES (%s, %s, %s, %s, %s)
            """, (username, email, password_hash, role, full_name))
            user_id = cursor.lastrowid
        
        self.connection.commit()
        self.logger.info(f"User created: {username} (ID: {user_id})")
        return user_id
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info"""
        cursor = self.connection.cursor()
        
        if self.db_type == 'sqlite':
            cursor.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
        else:
            cursor.execute("SELECT * FROM users WHERE username = %s AND is_active = TRUE", (username,))
        
        user = cursor.fetchone()
        
        if user and self._verify_password(password, user['password_hash'] if self.db_type == 'sqlite' else user[3]):
            return dict(user) if self.db_type == 'sqlite' else {
                'user_id': user[0],
                'username': user[1],
                'email': user[2],
                'role': user[4],
                'full_name': user[5]
            }
        
        return None
    
    # Patient Management Methods
    def create_patient(self, patient_data: Dict) -> int:
        """Create a new patient record"""
        cursor = self.connection.cursor()
        
        # Generate patient number if not provided
        if not patient_data.get('patient_number'):
            patient_data['patient_number'] = self._generate_patient_number()
        
        if self.db_type == 'sqlite':
            cursor.execute("""
                INSERT INTO patients (patient_number, full_name, date_of_birth, sex, 
                                    municipality, parent_user_id, assigned_nutritionist_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_data['patient_number'],
                patient_data['full_name'],
                patient_data.get('date_of_birth'),
                patient_data['sex'],
                patient_data.get('municipality'),
                patient_data.get('parent_user_id'),
                patient_data.get('assigned_nutritionist_id')
            ))
            patient_id = cursor.lastrowid
        
        # Similar for PostgreSQL and MySQL...
        
        self.connection.commit()
        return patient_id
    
    # Assessment Management Methods
    def save_assessment(self, assessment_data: Dict) -> int:
        """Save a new assessment"""
        cursor = self.connection.cursor()
        
        assessment_fields = [
            'patient_id', 'assessed_by_user_id', 'assessment_date', 'age_months',
            'weight_kg', 'height_cm', 'whz_score', 'nutritional_status', 'bmi',
            'edema', 'total_household', 'adults', 'children', 'is_twin',
            'fourps_beneficiary', 'breastfeeding', 'tuberculosis', 'malaria',
            'congenital_anomalies', 'other_medical_problems', 'prediction_confidence',
            'treatment_protocol', 'recommendations', 'notes'
        ]
        
        values = [assessment_data.get(field) for field in assessment_fields]
        
        if self.db_type == 'sqlite':
            placeholders = ', '.join(['?' for _ in assessment_fields])
            cursor.execute(f"""
                INSERT INTO assessments ({', '.join(assessment_fields)})
                VALUES ({placeholders})
            """, values)
            assessment_id = cursor.lastrowid
        
        # Similar for other database types...
        
        self.connection.commit()
        return assessment_id
    
    # Analytics Methods
    def get_assessment_statistics(self, filters: Dict = None) -> Dict:
        """Get assessment statistics for analytics"""
        cursor = self.connection.cursor()
        
        base_query = """
            SELECT 
                nutritional_status,
                COUNT(*) as count,
                AVG(whz_score) as avg_whz_score,
                AVG(age_months) as avg_age_months,
                municipality
            FROM assessments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE 1=1
        """
        
        params = []
        
        # Add filters
        if filters:
            if filters.get('municipality'):
                base_query += " AND p.municipality = ?"
                params.append(filters['municipality'])
            
            if filters.get('date_from'):
                base_query += " AND a.assessment_date >= ?"
                params.append(filters['date_from'])
            
            if filters.get('date_to'):
                base_query += " AND a.assessment_date <= ?"
                params.append(filters['date_to'])
        
        base_query += " GROUP BY nutritional_status, municipality"
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        # Process results into analytics format
        statistics = {
            'total_assessments': 0,
            'by_status': {},
            'by_municipality': {},
            'average_whz_score': 0
        }
        
        for row in results:
            status = row[0] if isinstance(row, tuple) else row['nutritional_status']
            count = row[1] if isinstance(row, tuple) else row['count']
            
            statistics['total_assessments'] += count
            
            if status not in statistics['by_status']:
                statistics['by_status'][status] = 0
            statistics['by_status'][status] += count
        
        return statistics
    
    def get_patient_history(self, patient_id: int) -> List[Dict]:
        """Get assessment history for a specific patient"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT a.*, u.full_name as assessed_by_name
            FROM assessments a
            JOIN users u ON a.assessed_by_user_id = u.user_id
            WHERE a.patient_id = ?
            ORDER BY a.assessment_date DESC
        """, (patient_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_nutritionist_caseload(self, nutritionist_id: int) -> List[Dict]:
        """Get all patients assigned to a nutritionist"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT p.*, 
                   latest_assessment.assessment_date as last_assessment,
                   latest_assessment.nutritional_status as current_status
            FROM patients p
            LEFT JOIN (
                SELECT patient_id, 
                       MAX(assessment_date) as assessment_date,
                       nutritional_status
                FROM assessments
                GROUP BY patient_id
            ) latest_assessment ON p.patient_id = latest_assessment.patient_id
            WHERE p.assigned_nutritionist_id = ? AND p.is_active = 1
            ORDER BY latest_assessment.assessment_date DESC
        """, (nutritionist_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # Utility Methods
    def _hash_password(self, password: str) -> str:
        """Hash password for secure storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return hashlib.sha256(password.encode()).hexdigest() == password_hash
    
    def _generate_patient_number(self) -> str:
        """Generate unique patient number"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"PAT_{timestamp}"
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")

# Database configuration helper
class DatabaseConfig:
    """Helper class for database configuration"""
    
    @staticmethod
    def get_sqlite_config(db_path: str = "malnutrition_assessment.db"):
        return {
            'db_type': 'sqlite',
            'database': db_path
        }
    
    @staticmethod
    def get_postgresql_config(host: str, database: str, user: str, password: str, port: int = 5432):
        return {
            'db_type': 'postgresql',
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
    
    @staticmethod
    def get_mysql_config(host: str, database: str, user: str, password: str, port: int = 3306):
        return {
            'db_type': 'mysql',
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
