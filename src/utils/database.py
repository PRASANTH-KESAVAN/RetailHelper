# src/utils/database.py

import pandas as pd
import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker
from loguru import logger
import yaml
from datetime import datetime
import shutil

class DatabaseManager:
    """
    Database manager for retail customer analytics
    """
    
    def __init__(self, config_path: str = "config/database_config.yaml"):
        """Initialize database manager"""
        self.config = self._load_config(config_path)
        self.engine = None
        self.session = None
        self.metadata = MetaData()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading database config: {e}")
            # Return default SQLite config
            return {
                'database': {
                    'type': 'sqlite',
                    'name': 'retail_analytics.db'
                },
                'connection_strings': {
                    'sqlite': 'sqlite:///data/database/retail_analytics.db'
                }
            }
    
    def create_connection(self, connection_string: str = None) -> None:
        """
        Create database connection
        
        Args:
            connection_string: Custom connection string (optional)
        """
        try:
            if connection_string is None:
                db_type = self.config['database']['type']
                connection_string = self.config['connection_strings'][db_type]
            
            # Ensure directory exists for SQLite
            if connection_string.startswith('sqlite'):
                db_path = Path(connection_string.replace('sqlite:///', ''))
                db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.engine = create_engine(connection_string)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            logger.info(f"Database connection established: {db_type}")
            
        except Exception as e:
            logger.error(f"Error creating database connection: {e}")
            raise
    
    def create_tables(self) -> None:
        """Create database tables based on configuration"""
        try:
            # Customers table
            customers = Table(
                'customers', self.metadata,
                Column('customer_id', String(50), primary_key=True),
                Column('age', Integer),
                Column('gender', String(10)),
                Column('location', String(100)),
                Column('subscription_status', String(10)),
                Column('created_date', DateTime, default=datetime.utcnow),
                Column('updated_date', DateTime, default=datetime.utcnow)
            )
            
            # Transactions table
            transactions = Table(
                'transactions', self.metadata,
                Column('transaction_id', String(50), primary_key=True),
                Column('customer_id', String(50)),
                Column('item_purchased', String(200)),
                Column('category', String(100)),
                Column('purchase_amount', Float),
                Column('purchase_date', DateTime),
                Column('review_rating', Float),
                Column('payment_method', String(50)),
                Column('created_date', DateTime, default=datetime.utcnow)
            )
            
            # Customer segments table
            segments = Table(
                'customer_segments', self.metadata,
                Column('segment_id', Integer, primary_key=True),
                Column('customer_id', String(50)),
                Column('segment_label', Integer),
                Column('segment_name', String(100)),
                Column('rfm_score', String(10)),
                Column('created_date', DateTime, default=datetime.utcnow),
                Column('model_version', String(20))
            )
            
            # Model predictions table
            predictions = Table(
                'model_predictions', self.metadata,
                Column('prediction_id', Integer, primary_key=True),
                Column('customer_id', String(50)),
                Column('model_name', String(50)),
                Column('prediction_type', String(50)),
                Column('prediction_value', Float),
                Column('prediction_probability', Float),
                Column('prediction_date', DateTime, default=datetime.utcnow),
                Column('model_version', String(20))
            )
            
            # Create all tables
            self.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def insert_data(self, table_name: str, data: Union[Dict, pd.DataFrame]) -> None:
        """
        Insert data into database table
        
        Args:
            table_name: Name of the table
            data: Data to insert (dict or DataFrame)
        """
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            data.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"Inserted {len(data)} records into {table_name}")
            
        except Exception as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            raise
    
    def update_data(self, table_name: str, update_data: Dict, condition: str) -> None:
        """
        Update data in database table
        
        Args:
            table_name: Name of the table
            update_data: Data to update
            condition: WHERE condition
        """
        try:
            set_clause = ", ".join([f"{k} = '{v}'" for k, v in update_data.items()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
            
            self.engine.execute(text(query))
            logger.info(f"Updated data in {table_name}")
            
        except Exception as e:
            logger.error(f"Error updating data in {table_name}: {e}")
            raise
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            Query results as DataFrame
        """
        try:
            if params:
                result = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                result = pd.read_sql_query(query, self.engine)
            
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_customer_data(self, customer_id: str = None) -> pd.DataFrame:
        """
        Get customer data from database
        
        Args:
            customer_id: Specific customer ID (optional)
            
        Returns:
            Customer data as DataFrame
        """
        try:
            if customer_id:
                query = "SELECT * FROM customers WHERE customer_id = :customer_id"
                params = {'customer_id': customer_id}
            else:
                query = "SELECT * FROM customers"
                params = None
            
            return self.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Error getting customer data: {e}")
            raise
    
    def get_transaction_data(self, customer_id: str = None, 
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get transaction data from database
        
        Args:
            customer_id: Specific customer ID (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            Transaction data as DataFrame
        """
        try:
            query = "SELECT * FROM transactions WHERE 1=1"
            params = {}
            
            if customer_id:
                query += " AND customer_id = :customer_id"
                params['customer_id'] = customer_id
            
            if start_date:
                query += " AND purchase_date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND purchase_date <= :end_date"
                params['end_date'] = end_date
            
            return self.execute_query(query, params if params else None)
            
        except Exception as e:
            logger.error(f"Error getting transaction data: {e}")
            raise
    
    def save_customer_segments(self, segments_data: pd.DataFrame, model_version: str = "1.0") -> None:
        """
        Save customer segments to database
        
        Args:
            segments_data: DataFrame with customer segments
            model_version: Version of the segmentation model
        """
        try:
            segments_data['model_version'] = model_version
            segments_data['created_date'] = datetime.utcnow()
            
            self.insert_data('customer_segments', segments_data)
            logger.info(f"Saved {len(segments_data)} customer segments")
            
        except Exception as e:
            logger.error(f"Error saving customer segments: {e}")
            raise
    
    def save_model_predictions(self, predictions_data: pd.DataFrame, 
                             model_name: str, prediction_type: str, 
                             model_version: str = "1.0") -> None:
        """
        Save model predictions to database
        
        Args:
            predictions_data: DataFrame with predictions
            model_name: Name of the model
            prediction_type: Type of prediction
            model_version: Version of the model
        """
        try:
            predictions_data['model_name'] = model_name
            predictions_data['prediction_type'] = prediction_type
            predictions_data['model_version'] = model_version
            predictions_data['prediction_date'] = datetime.utcnow()
            
            self.insert_data('model_predictions', predictions_data)
            logger.info(f"Saved {len(predictions_data)} model predictions")
            
        except Exception as e:
            logger.error(f"Error saving model predictions: {e}")
            raise
    
    def backup_database(self, backup_path: str = None) -> None:
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file (optional)
        """
        try:
            if backup_path is None:
                backup_dir = Path("backups/database")
                backup_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"retail_analytics_backup_{timestamp}.db"
            
            # For SQLite databases
            if self.config['database']['type'] == 'sqlite':
                db_path = self.config['connection_strings']['sqlite'].replace('sqlite:///', '')
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backup created: {backup_path}")
            else:
                logger.warning("Backup not implemented for non-SQLite databases")
                
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table information dictionary
        """
        try:
            # Get table schema
            schema_query = f"PRAGMA table_info({table_name})"
            schema_info = self.execute_query(schema_query)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            row_count = self.execute_query(count_query)['row_count'].iloc[0]
            
            return {
                'table_name': table_name,
                'columns': schema_info.to_dict('records'),
                'row_count': row_count
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            raise
    
    def close_connection(self) -> None:
        """Close database connection"""
        try:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connection closed")
            
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

# Convenience functions
def create_connection(config_path: str = "config/database_config.yaml") -> DatabaseManager:
    """
    Create and return database connection
    
    Args:
        config_path: Path to database configuration
        
    Returns:
        DatabaseManager instance
    """
    db_manager = DatabaseManager(config_path)
    db_manager.create_connection()
    return db_manager

def execute_query(query: str, config_path: str = "config/database_config.yaml") -> pd.DataFrame:
    """
    Execute query and return results
    
    Args:
        query: SQL query to execute
        config_path: Path to database configuration
        
    Returns:
        Query results as DataFrame
    """
    db_manager = create_connection(config_path)
    try:
        result = db_manager.execute_query(query)
        return result
    finally:
        db_manager.close_connection()

def insert_data(table_name: str, data: Union[Dict, pd.DataFrame], 
               config_path: str = "config/database_config.yaml") -> None:
    """
    Insert data into table
    
    Args:
        table_name: Name of the table
        data: Data to insert
        config_path: Path to database configuration
    """
    db_manager = create_connection(config_path)
    try:
        db_manager.insert_data(table_name, data)
    finally:
        db_manager.close_connection()

def update_data(table_name: str, update_data: Dict, condition: str,
               config_path: str = "config/database_config.yaml") -> None:
    """
    Update data in table
    
    Args:
        table_name: Name of the table
        update_data: Data to update
        condition: WHERE condition
        config_path: Path to database configuration
    """
    db_manager = create_connection(config_path)
    try:
        db_manager.update_data(table_name, update_data, condition)
    finally:
        db_manager.close_connection()

def create_tables(config_path: str = "config/database_config.yaml") -> None:
    """
    Create database tables
    
    Args:
        config_path: Path to database configuration
    """
    db_manager = create_connection(config_path)
    try:
        db_manager.create_tables()
    finally:
        db_manager.close_connection()

def backup_database(backup_path: str = None, 
                   config_path: str = "config/database_config.yaml") -> None:
    """
    Create database backup
    
    Args:
        backup_path: Path for backup file
        config_path: Path to database configuration
    """
    db_manager = create_connection(config_path)
    try:
        db_manager.backup_database(backup_path)
    finally:
        db_manager.close_connection()