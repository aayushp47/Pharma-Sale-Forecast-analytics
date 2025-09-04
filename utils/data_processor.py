#!/usr/bin/env python3
"""
Data Processing Utilities for Healthcare Sales Analysis
Handles data cleaning, validation, and synthetic data generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Comprehensive data processing utilities"""
    
    def __init__(self):
        self.required_columns = ['Date', 'Drug', 'Revenue']
        self.optional_columns = ['Units_Sold', 'Price_Per_Unit', 'Region', 'Territory']
        
    def process_data(self, raw_data):
        """Main data processing pipeline"""
        try:
            # Make a copy to avoid modifying original data
            data = raw_data.copy()
            
            # Basic validation
            self._validate_raw_data(data)
            
            # Clean and standardize data
            data = self._clean_data(data)
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Validate data types
            data = self._validate_data_types(data)
            
            # Add derived columns
            data = self._add_derived_columns(data)
            
            # Remove outliers
            data = self._remove_outliers(data)
            
            # Sort by date
            data = data.sort_values(['Date', 'Drug']).reset_index(drop=True)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Data processing failed: {str(e)}")
    
    def generate_demo_data(self, years=4, drugs=None, regions=None):
        """Generate realistic synthetic pharmaceutical sales data"""
        try:
            # Default drug list
            if drugs is None:
                drugs = [
                    'Aspirin', 'Ibuprofen', 'Lisinopril', 'Metformin', 'Amlodipine',
                    'Omeprazole', 'Atorvastatin', 'Levothyroxine', 'Simvastatin', 
                    'Warfarin', 'Metoprolol', 'Hydrochlorothiazide'
                ]
            
            # Default regions
            if regions is None:
                regions = ['North America', 'Europe', 'Asia-Pacific', 'Latin America']
            
            # Generate date range
            start_date = datetime.now() - timedelta(days=years * 365)
            end_date = datetime.now()
            
            data_points = []
            
            # Drug characteristics (affects baseline sales)
            drug_characteristics = {
                'Aspirin': {'base_price': 25, 'seasonality': 0.1, 'trend': 0.02},
                'Ibuprofen': {'base_price': 35, 'seasonality': 0.15, 'trend': 0.03},
                'Lisinopril': {'base_price': 45, 'seasonality': 0.05, 'trend': 0.01},
                'Metformin': {'base_price': 55, 'seasonality': 0.08, 'trend': 0.04},
                'Amlodipine': {'base_price': 40, 'seasonality': 0.12, 'trend': 0.02},
                'Omeprazole': {'base_price': 65, 'seasonality': 0.2, 'trend': -0.01},
                'Atorvastatin': {'base_price': 75, 'seasonality': 0.1, 'trend': 0.05},
                'Levothyroxine': {'base_price': 30, 'seasonality': 0.08, 'trend': 0.03},
                'Simvastatin': {'base_price': 50, 'seasonality': 0.12, 'trend': 0.01},
                'Warfarin': {'base_price': 80, 'seasonality': 0.15, 'trend': -0.02},
                'Metoprolol': {'base_price': 60, 'seasonality': 0.1, 'trend': 0.02},
                'Hydrochlorothiazide': {'base_price': 25, 'seasonality': 0.08, 'trend': 0.01}
            }
            
            # Generate monthly data points
            current_date = start_date.replace(day=1)  # Start from first of month
            
            while current_date < end_date:
                for drug in drugs:
                    for region in regions:
                        # Get drug characteristics
                        char = drug_characteristics.get(drug, {
                            'base_price': random.uniform(30, 80),
                            'seasonality': random.uniform(0.05, 0.2),
                            'trend': random.uniform(-0.02, 0.05)
                        })
                        
                        # Calculate base units sold
                        base_units = random.randint(800, 2500)
                        
                        # Apply seasonal effects
                        month = current_date.month
                        seasonal_multiplier = 1 + char['seasonality'] * np.sin(2 * np.pi * month / 12)
                        
                        # Apply trend (years since start)
                        years_elapsed = (current_date - start_date).days / 365.25
                        trend_multiplier = 1 + char['trend'] * years_elapsed
                        
                        # Apply random noise
                        noise_multiplier = random.uniform(0.8, 1.2)
                        
                        # Special events (random market disruptions)
                        special_event_multiplier = 1.0
                        if random.random() < 0.02:  # 2% chance of special event
                            special_event_multiplier = random.uniform(0.3, 1.8)
                        
                        # Calculate final units
                        units_sold = int(base_units * seasonal_multiplier * trend_multiplier * 
                                       noise_multiplier * special_event_multiplier)
                        units_sold = max(0, units_sold)  # Ensure non-negative
                        
                        # Calculate price with some variation
                        price_variation = random.uniform(0.9, 1.1)
                        price_per_unit = char['base_price'] * price_variation
                        
                        # Calculate revenue
                        revenue = units_sold * price_per_unit
                        
                        # Add some regional variation
                        regional_multipliers = {
                            'North America': 1.2,
                            'Europe': 1.0,
                            'Asia-Pacific': 0.8,
                            'Latin America': 0.6
                        }
                        regional_mult = regional_multipliers.get(region, 1.0)
                        revenue *= regional_mult
                        units_sold = int(units_sold * regional_mult)
                        
                        data_points.append({
                            'Date': current_date.strftime('%Y-%m-%d'),
                            'Drug': drug,
                            'Units_Sold': units_sold,
                            'Price_Per_Unit': round(price_per_unit, 2),
                            'Revenue': round(revenue, 2),
                            'Region': region,
                            'Territory': f"{region}-{random.randint(1, 5)}",
                            'Market_Share': round(random.uniform(5, 25), 1),
                            'Competition_Index': round(random.uniform(0.3, 0.9), 2)
                        })
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            # Convert to DataFrame
            demo_data = pd.DataFrame(data_points)
            
            # Add some data quality issues to make it more realistic
            demo_data = self._add_realistic_data_issues(demo_data)
            
            return demo_data
            
        except Exception as e:
            raise ValueError(f"Demo data generation failed: {str(e)}")
    
    def _validate_raw_data(self, data):
        """Validate raw data structure and content"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data is empty")
        
        # Check for at least some required columns
        available_cols = data.columns.tolist()
        
        # Try to map common column variations
        column_mapping = self._detect_column_mapping(available_cols)
        
        if not column_mapping.get('date') or not column_mapping.get('drug'):
            raise ValueError(
                f"Could not find required columns. Available columns: {available_cols}. "
                "Expected columns like: Date, Drug, Revenue, Sales, Amount, etc."
            )
    
    def _detect_column_mapping(self, columns):
        """Detect and map column names to standard format"""
        mapping = {}
        columns_lower = [col.lower() for col in columns]
        
        # Date column detection
        date_patterns = ['date', 'time', 'period', 'month', 'year']
        for pattern in date_patterns:
            matches = [col for col in columns if pattern in col.lower()]
            if matches:
                mapping['date'] = matches[0]
                break
        
        # Drug/Product column detection
        drug_patterns = ['drug', 'product', 'medicine', 'pharmaceutical', 'name', 'item']
        for pattern in drug_patterns:
            matches = [col for col in columns if pattern in col.lower()]
            if matches:
                mapping['drug'] = matches[0]
                break
        
        # Revenue column detection
        revenue_patterns = ['revenue', 'sales', 'amount', 'value', 'income', 'earnings']
        for pattern in revenue_patterns:
            matches = [col for col in columns if pattern in col.lower()]
            if matches:
                mapping['revenue'] = matches[0]
                break
        
        # Units column detection
        units_patterns = ['units', 'quantity', 'volume', 'count', 'sold', 'qty']
        for pattern in units_patterns:
            matches = [col for col in columns if pattern in col.lower()]
            if matches:
                mapping['units'] = matches[0]
                break
        
        return mapping
    
    def _clean_data(self, data):
        """Clean and standardize data"""
        data = data.copy()
        
        # Detect column mapping
        column_mapping = self._detect_column_mapping(data.columns.tolist())
        
        # Rename columns to standard format
        rename_dict = {}
        if column_mapping.get('date'):
            rename_dict[column_mapping['date']] = 'Date'
        if column_mapping.get('drug'):
            rename_dict[column_mapping['drug']] = 'Drug'
        if column_mapping.get('revenue'):
            rename_dict[column_mapping['revenue']] = 'Revenue'
        if column_mapping.get('units'):
            rename_dict[column_mapping['units']] = 'Units_Sold'
        
        data = data.rename(columns=rename_dict)
        
        # Clean text columns
        text_columns = data.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col != 'Date':  # Don't clean date column yet
                data[col] = data[col].astype(str).str.strip()
                data[col] = data[col].replace(['', 'nan', 'None', 'null'], np.nan)
        
        # Clean numeric columns
        numeric_columns = ['Revenue', 'Units_Sold', 'Price_Per_Unit']
        for col in numeric_columns:
            if col in data.columns:
                # Remove currency symbols and commas
                if data[col].dtype == 'object':
                    data[col] = data[col].astype(str).str.replace(r'[\$,£€¥]', '', regex=True)
                    data[col] = data[col].str.replace(',', '')
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values appropriately"""
        data = data.copy()
        
        # Remove rows with missing critical information
        critical_columns = ['Date', 'Drug']
        data = data.dropna(subset=critical_columns)
        
        # Handle missing Revenue
        if 'Revenue' in data.columns:
            # If Units_Sold and Price_Per_Unit exist, calculate Revenue
            if 'Units_Sold' in data.columns and 'Price_Per_Unit' in data.columns:
                mask = data['Revenue'].isna() & data['Units_Sold'].notna() & data['Price_Per_Unit'].notna()
                data.loc[mask, 'Revenue'] = data.loc[mask, 'Units_Sold'] * data.loc[mask, 'Price_Per_Unit']
            
            # Fill remaining missing revenue with 0
            data['Revenue'] = data['Revenue'].fillna(0)
        
        # Handle missing Units_Sold
        if 'Units_Sold' in data.columns:
            data['Units_Sold'] = data['Units_Sold'].fillna(0)
        
        # Handle missing categorical variables
        categorical_columns = ['Region', 'Territory']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        
        return data
    
    def _validate_data_types(self, data):
        """Validate and convert data types"""
        data = data.copy()
        
        # Convert Date column
        if 'Date' in data.columns:
            try:
                data['Date'] = pd.to_datetime(data['Date'])
            except Exception as e:
                raise ValueError(f"Could not parse Date column: {str(e)}")
        
        # Convert numeric columns
        numeric_columns = ['Revenue', 'Units_Sold', 'Price_Per_Unit', 'Market_Share', 'Competition_Index']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Ensure Drug is string
        if 'Drug' in data.columns:
            data['Drug'] = data['Drug'].astype(str)
        
        return data
    
    def _add_derived_columns(self, data):
        """Add useful derived columns"""
        data = data.copy()
        
        # Add time-based columns
        if 'Date' in data.columns:
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Quarter'] = data['Date'].dt.quarter
            data['Weekday'] = data['Date'].dt.day_name()
            data['Month_Name'] = data['Date'].dt.month_name()
        
        # Calculate Price_Per_Unit if missing but have Revenue and Units_Sold
        if ('Price_Per_Unit' not in data.columns and 
            'Revenue' in data.columns and 
            'Units_Sold' in data.columns):
            
            mask = (data['Units_Sold'] > 0)
            data.loc[mask, 'Price_Per_Unit'] = data.loc[mask, 'Revenue'] / data.loc[mask, 'Units_Sold']
            data['Price_Per_Unit'] = data['Price_Per_Unit'].fillna(0)
        
        # Add revenue per unit if not exists
        if ('Revenue_Per_Unit' not in data.columns and 
            'Revenue' in data.columns and 
            'Units_Sold' in data.columns):
            
            mask = (data['Units_Sold'] > 0)
            data.loc[mask, 'Revenue_Per_Unit'] = data.loc[mask, 'Revenue'] / data.loc[mask, 'Units_Sold']
            data['Revenue_Per_Unit'] = data['Revenue_Per_Unit'].fillna(0)
        
        return data
    
    def _remove_outliers(self, data, method='iqr'):
        """Remove outliers using specified method"""
        data = data.copy()
        
        if method == 'iqr':
            # Remove outliers based on IQR for Revenue column
            if 'Revenue' in data.columns and len(data) > 10:
                Q1 = data['Revenue'].quantile(0.25)
                Q3 = data['Revenue'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Only remove if outliers are extreme (to preserve data)
                extreme_lower = Q1 - 3 * IQR
                extreme_upper = Q3 + 3 * IQR
                
                data = data[
                    (data['Revenue'] >= extreme_lower) & 
                    (data['Revenue'] <= extreme_upper)
                ]
        
        return data
    
    def _add_realistic_data_issues(self, data):
        """Add realistic data quality issues to demo data"""
        data = data.copy()
        
        # Randomly introduce some missing values (1-2%)
        missing_rate = 0.015
        for col in ['Units_Sold', 'Price_Per_Unit']:
            if col in data.columns:
                missing_mask = np.random.random(len(data)) < missing_rate
                data.loc[missing_mask, col] = np.nan
        
        # Add some duplicate entries (rare)
        if len(data) > 100:
            duplicate_count = max(1, int(len(data) * 0.005))  # 0.5% duplicates
            duplicate_indices = np.random.choice(data.index, duplicate_count, replace=False)
            duplicates = data.loc[duplicate_indices].copy()
            data = pd.concat([data, duplicates], ignore_index=True)
        
        # Add some data entry errors (very rare)
        if 'Revenue' in data.columns:
            error_mask = np.random.random(len(data)) < 0.002  # 0.2% error rate
            error_indices = data[error_mask].index
            
            for idx in error_indices:
                if np.random.random() < 0.5:
                    # Decimal place error
                    data.loc[idx, 'Revenue'] *= 10
                else:
                    # Missing digit error
                    data.loc[idx, 'Revenue'] /= 10
        
        return data
    
    def validate_processed_data(self, data):
        """Final validation of processed data"""
        issues = []
        
        # Check required columns
        required_cols = ['Date', 'Drug', 'Revenue']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'Date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['Date']):
            issues.append("Date column is not datetime type")
        
        if 'Revenue' in data.columns and not pd.api.types.is_numeric_dtype(data['Revenue']):
            issues.append("Revenue column is not numeric")
        
        # Check for empty data
        if data.empty:
            issues.append("Processed data is empty")
        
        # Check for negative values where inappropriate
        if 'Revenue' in data.columns and (data['Revenue'] < 0).any():
            issues.append("Found negative revenue values")
        
        if 'Units_Sold' in data.columns and (data['Units_Sold'] < 0).any():
            issues.append("Found negative units sold values")
        
        # Return validation results
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'rows': len(data),
            'columns': list(data.columns),
            'date_range': {
                'start': data['Date'].min().strftime('%Y-%m-%d') if 'Date' in data.columns else None,
                'end': data['Date'].max().strftime('%Y-%m-%d') if 'Date' in data.columns else None
            } if 'Date' in data.columns else None
        }