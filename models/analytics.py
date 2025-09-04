#!/usr/bin/env python3
"""
Analytics Engine for Healthcare Sales Analysis
Provides comprehensive data analysis and metrics calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnalyticsEngine:
    """Comprehensive analytics engine for sales data analysis"""
    
    def __init__(self):
        self.required_columns = ['Date', 'Drug', 'Revenue']
        
    def calculate_metrics(self, data):
        """Calculate comprehensive summary metrics"""
        try:
            self._validate_data(data)
            
            total_revenue = data['Revenue'].sum()
            total_units = data.get('Units_Sold', pd.Series([0])).sum()
            unique_drugs = data['Drug'].nunique()
            date_range = (data['Date'].max() - data['Date'].min()).days
            
            drug_revenues = data.groupby('Drug')['Revenue'].sum()
            top_drug = drug_revenues.idxmax() if not drug_revenues.empty else 'N/A'
            top_drug_revenue = drug_revenues.max() if not drug_revenues.empty else 0
            top_drug_share = (top_drug_revenue / total_revenue * 100) if total_revenue > 0 else 0
            
            monthly_data = self._get_monthly_aggregation(data)
            growth_rate = self._calculate_growth_rate(monthly_data)
            
            performance_stats = self._calculate_performance_distribution(data)
            seasonal_strength = self._calculate_seasonal_strength(data)
            
            return {
                'total_revenue': round(total_revenue, 2),
                'total_units': int(total_units),
                'unique_drugs': unique_drugs,
                'date_range_days': date_range,
                'top_drug': {
                    'name': top_drug,
                    'revenue': round(top_drug_revenue, 2),
                    'market_share': round(top_drug_share, 1)
                },
                'growth_rate': round(growth_rate, 2),
                'performance_distribution': performance_stats,
                'seasonal_strength': round(seasonal_strength, 2),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")
    
    def get_trends_data(self, data, period='monthly'):
        """Get sales trends data aggregated by specified period"""
        try:
            self._validate_data(data)
            
            if period == 'monthly':
                grouped_data = self._get_monthly_aggregation(data)
            elif period == 'quarterly':
                grouped_data = self._get_quarterly_aggregation(data)
            elif period == 'yearly':
                grouped_data = self._get_yearly_aggregation(data)
            else:
                raise ValueError(f"Unsupported period: {period}")
            
            top_drugs = data.groupby('Drug')['Revenue'].sum().nlargest(5).index.tolist()
            
            trends_data = []
            for drug in top_drugs:
                drug_data = grouped_data[grouped_data['Drug'] == drug]
                trends_data.append({
                    'drug': drug,
                    'data': drug_data[['period', 'Revenue']].to_dict('records')
                })
            
            return {
                'trends': trends_data,
                'period': period,
                'total_periods': len(grouped_data['period'].unique()),
                'date_range': {
                    'start': data['Date'].min().strftime('%Y-%m-%d'),
                    'end': data['Date'].max().strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating trends: {str(e)}")
    
    # ... (The rest of the file remains unchanged. You can copy it from your original file or use the full version from a previous conversation.)
    # The helper methods below are still required for the class to function.
    
    def _validate_data(self, data):
        """Validate input data structure"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if data.empty:
            raise ValueError("Data is empty")
        
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'])
    
    def _get_monthly_aggregation(self, data):
        """Aggregate data by month"""
        monthly_data = data.copy()
        monthly_data['period'] = monthly_data['Date'].dt.to_period('M').astype(str)
        
        agg_dict = {'Revenue': 'sum'}
        if 'Units_Sold' in data.columns:
            agg_dict['Units_Sold'] = 'sum'
            
        result = monthly_data.groupby(['period', 'Drug']).agg(agg_dict).reset_index()
        return result
    
    def _get_quarterly_aggregation(self, data):
        """Aggregate data by quarter"""
        quarterly_data = data.copy()
        quarterly_data['period'] = quarterly_data['Date'].dt.to_period('Q').astype(str)
        agg_dict = {'Revenue': 'sum'}
        if 'Units_Sold' in data.columns:
            agg_dict['Units_Sold'] = 'sum'
        result = quarterly_data.groupby(['period', 'Drug']).agg(agg_dict).reset_index()
        return result
    
    def _get_yearly_aggregation(self, data):
        """Aggregate data by year"""
        yearly_data = data.copy()
        yearly_data['period'] = yearly_data['Date'].dt.year.astype(str)
        agg_dict = {'Revenue': 'sum'}
        if 'Units_Sold' in data.columns:
            agg_dict['Units_Sold'] = 'sum'
        result = yearly_data.groupby(['period', 'Drug']).agg(agg_dict).reset_index()
        return result
    
    def _calculate_growth_rate(self, monthly_data):
        """Calculate overall growth rate from monthly data"""
        period_totals = monthly_data.groupby('period')['Revenue'].sum().sort_index()
        if len(period_totals) < 2:
            return 0
        growth_rates = period_totals.pct_change().dropna()
        return growth_rates.mean() * 12 * 100
    
    def _calculate_performance_distribution(self, data):
        """Calculate performance distribution statistics"""
        drug_revenues = data.groupby('Drug')['Revenue'].sum()
        if drug_revenues.empty:
            return {}
        return {
            'mean': round(drug_revenues.mean(), 2),
            'median': round(drug_revenues.median(), 2),
            'std': round(drug_revenues.std(), 2),
            'min': round(drug_revenues.min(), 2),
            'max': round(drug_revenues.max(), 2),
            'q25': round(drug_revenues.quantile(0.25), 2),
            'q75': round(drug_revenues.quantile(0.75), 2)
        }
    
    def _calculate_seasonal_strength(self, data):
        """Calculate seasonal strength in the data"""
        try:
            monthly_data = self._get_monthly_aggregation(data)
            period_totals = monthly_data.groupby('period')['Revenue'].sum()
            if len(period_totals) < 12:
                return 0
            
            monthly_series = period_totals.values
            mean_value = np.mean(monthly_series)
            if mean_value == 0: return 0
            
            seasonal_cv = np.std(monthly_series) / mean_value
            return min(seasonal_cv * 100, 100)
        except Exception:
            return 0