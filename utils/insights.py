#!/usr/bin/env python3
"""
AI-Powered Insights Generator for Healthcare Sales Analysis
Generates actionable business insights from pharmaceutical sales data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InsightsGenerator:
    """AI-powered insights generator for sales data analysis"""
    
    def __init__(self):
        self.insight_templates = {
            'top_performer': "Drug {drug} is the top performer with {revenue:,.0f} in revenue, representing {share:.1f}% of total sales.",
            'declining_trend': "Drug {drug} shows a declining trend with {decline:.1f}% decrease over the past {months} months.",
            'growth_opportunity': "Drug {drug} shows strong growth potential with {growth:.1f}% increase over the past {months} months.",
            'seasonal_pattern': "Drug {drug} exhibits strong seasonal patterns with peak sales in {peak_month}.",
            'market_concentration': "The top {count} drugs control {percentage:.1f}% of the market, indicating {level} market concentration.",
            'revenue_volatility': "Drug {drug} shows high revenue volatility with a coefficient of variation of {cv:.2f}.",
            'underperformer': "Drug {drug} is underperforming with only {share:.1f}% market share despite historical importance.",
            'price_opportunity': "Drug {drug} has pricing optimization opportunity with current price of ${price:.2f} per unit.",
            'regional_insight': "Region {region} shows the strongest performance with {percentage:.1f}% of total revenue.",
            'forecast_accuracy': "Forecasting models show {accuracy:.1f}% accuracy, indicating {reliability} predictive reliability."
        }
        
        self.thresholds = {
            'high_growth': 15.0,  # >15% annual growth
            'decline': -10.0,     # <-10% annual decline
            'high_volatility': 0.3,  # CV > 0.3
            'seasonal_strength': 0.2,  # Seasonal component > 20%
            'market_concentration_high': 60.0,  # Top 3 drugs >60%
            'market_concentration_medium': 40.0,  # Top 3 drugs >40%
            'min_market_share': 5.0   # Minimum share for significance
        }
    
    def generate_insights(self, data, max_insights=8):
        """Generate comprehensive AI-powered insights"""
        try:
            insights = []
            
            # Validate data
            if data.empty or 'Date' not in data.columns or 'Drug' not in data.columns:
                return [{'title': 'Data Quality Issue', 'description': 'Insufficient data for insights generation.'}]
            
            # Generate different types of insights
            insights.extend(self._analyze_top_performers(data))
            insights.extend(self._analyze_trends(data))
            insights.extend(self._analyze_seasonal_patterns(data))
            insights.extend(self._analyze_market_concentration(data))
            insights.extend(self._analyze_volatility(data))
            insights.extend(self._analyze_regional_performance(data))
            insights.extend(self._analyze_pricing_opportunities(data))
            insights.extend(self._generate_strategic_recommendations(data))
            
            # Sort by importance score and limit
            insights = sorted(insights, key=lambda x: x.get('importance', 0), reverse=True)
            
            return insights[:max_insights]
            
        except Exception as e:
            return [{'title': 'Error', 'description': f'Error generating insights: {str(e)}'}]
    
    def _analyze_top_performers(self, data):
        """Analyze top performing drugs"""
        insights = []
        
        try:
            drug_revenues = data.groupby('Drug')['Revenue'].sum().sort_values(ascending=False)
            total_revenue = drug_revenues.sum()
            
            if total_revenue > 0 and len(drug_revenues) > 0:
                # Top performer
                top_drug = drug_revenues.index[0]
                top_revenue = drug_revenues.iloc[0]
                top_share = (top_revenue / total_revenue) * 100
                
                insights.append({
                    'title': 'Top Performer Analysis',
                    'description': self.insight_templates['top_performer'].format(
                        drug=top_drug, 
                        revenue=top_revenue, 
                        share=top_share
                    ),
                    'type': 'performance',
                    'importance': 9,
                    'actionable': True
                })
                
                # Underperformers
                if len(drug_revenues) > 5:
                    bottom_drugs = drug_revenues.tail(3)
                    for drug, revenue in bottom_drugs.items():
                        share = (revenue / total_revenue) * 100
                        if share > 0.5:  # Only flag if they have some significance
                            insights.append({
                                'title': 'Underperformance Alert',
                                'description': self.insight_templates['underperformer'].format(
                                    drug=drug, 
                                    share=share
                                ),
                                'type': 'risk',
                                'importance': 6,
                                'actionable': True
                            })
            
        except Exception:
            pass
        
        return insights
    
    def _analyze_trends(self, data):
        """Analyze growth and decline trends"""
        insights = []
        
        try:
            # Calculate monthly trends for each drug
            data_monthly = data.copy()
            data_monthly['YearMonth'] = data_monthly['Date'].dt.to_period('M')
            monthly_revenue = data_monthly.groupby(['Drug', 'YearMonth'])['Revenue'].sum().reset_index()
            
            for drug in data['Drug'].unique():
                drug_data = monthly_revenue[monthly_revenue['Drug'] == drug]
                if len(drug_data) >= 6:  # Need at least 6 months of data
                    
                    # Calculate trend
                    drug_data = drug_data.sort_values('YearMonth')
                    revenues = drug_data['Revenue'].values
                    
                    # Simple linear trend calculation
                    x = np.arange(len(revenues))
                    if len(revenues) > 1 and np.std(revenues) > 0:
                        correlation = np.corrcoef(x, revenues)[0, 1]
                        slope = np.polyfit(x, revenues, 1)[0]
                        
                        # Calculate percentage change
                        if revenues[0] > 0:
                            total_change = ((revenues[-1] - revenues[0]) / revenues[0]) * 100
                            monthly_change = total_change / len(revenues)
                            annual_change = monthly_change * 12
                            
                            # Identify significant trends
                            if annual_change > self.thresholds['high_growth']:
                                insights.append({
                                    'title': 'Growth Opportunity Identified',
                                    'description': self.insight_templates['growth_opportunity'].format(
                                        drug=drug, 
                                        growth=annual_change, 
                                        months=len(revenues)
                                    ),
                                    'type': 'opportunity',
                                    'importance': 8,
                                    'actionable': True
                                })
                            
                            elif annual_change < self.thresholds['decline']:
                                insights.append({
                                    'title': 'Declining Performance Alert',
                                    'description': self.insight_templates['declining_trend'].format(
                                        drug=drug, 
                                        decline=abs(annual_change), 
                                        months=len(revenues)
                                    ),
                                    'type': 'risk',
                                    'importance': 9,
                                    'actionable': True
                                })
        
        except Exception:
            pass
        
        return insights
    
    def _analyze_seasonal_patterns(self, data):
        """Analyze seasonal patterns in sales data"""
        insights = []
        
        try:
            # Calculate monthly averages
            data_with_month = data.copy()
            data_with_month['Month'] = data_with_month['Date'].dt.month
            monthly_avg = data_with_month.groupby(['Drug', 'Month'])['Revenue'].mean().reset_index()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for drug in data['Drug'].unique():
                drug_monthly = monthly_avg[monthly_avg['Drug'] == drug]
                if len(drug_monthly) >= 8:  # Need data for most months
                    
                    revenues = drug_monthly['Revenue'].values
                    if np.std(revenues) > 0:
                        # Calculate seasonal strength
                        seasonal_cv = np.std(revenues) / np.mean(revenues)
                        
                        if seasonal_cv > self.thresholds['seasonal_strength']:
                            peak_month_idx = drug_monthly.loc[drug_monthly['Revenue'].idxmax(), 'Month']
                            peak_month = month_names[peak_month_idx - 1]
                            
                            insights.append({
                                'title': 'Seasonal Pattern Detected',
                                'description': self.insight_templates['seasonal_pattern'].format(
                                    drug=drug, 
                                    peak_month=peak_month
                                ),
                                'type': 'pattern',
                                'importance': 7,
                                'actionable': True
                            })
        
        except Exception:
            pass
        
        return insights
    
    def _analyze_market_concentration(self, data):
        """Analyze market concentration and competitive landscape"""
        insights = []
        
        try:
            drug_revenues = data.groupby('Drug')['Revenue'].sum().sort_values(ascending=False)
            total_revenue = drug_revenues.sum()
            
            if total_revenue > 0 and len(drug_revenues) >= 3:
                # Calculate concentration ratios
                top3_revenue = drug_revenues.head(3).sum()
                top5_revenue = drug_revenues.head(5).sum()
                
                top3_share = (top3_revenue / total_revenue) * 100
                top5_share = (top5_revenue / total_revenue) * 100
                
                # Assess concentration level
                if top3_share > self.thresholds['market_concentration_high']:
                    concentration_level = "high"
                elif top3_share > self.thresholds['market_concentration_medium']:
                    concentration_level = "moderate"
                else:
                    concentration_level = "low"
                
                insights.append({
                    'title': 'Market Concentration Analysis',
                    'description': self.insight_templates['market_concentration'].format(
                        count=3, 
                        percentage=top3_share, 
                        level=concentration_level
                    ),
                    'type': 'market_structure',
                    'importance': 6,
                    'actionable': True
                })
                
                # Identify potential consolidation or diversification needs
                if concentration_level == "high":
                    insights.append({
                        'title': 'Portfolio Diversification Opportunity',
                        'description': f"High market concentration ({top3_share:.1f}% in top 3 drugs) suggests need for portfolio diversification to reduce risk.",
                        'type': 'strategic',
                        'importance': 7,
                        'actionable': True
                    })
        
        except Exception:
            pass
        
        return insights
    
    def _analyze_volatility(self, data):
        """Analyze revenue volatility for risk assessment"""
        insights = []
        
        try:
            # Calculate monthly volatility for each drug
            data_monthly = data.copy()
            data_monthly['YearMonth'] = data_monthly['Date'].dt.to_period('M')
            monthly_revenue = data_monthly.groupby(['Drug', 'YearMonth'])['Revenue'].sum().reset_index()
            
            for drug in data['Drug'].unique():
                drug_data = monthly_revenue[monthly_revenue['Drug'] == drug]
                if len(drug_data) >= 6:
                    
                    revenues = drug_data['Revenue'].values
                    if len(revenues) > 1 and np.mean(revenues) > 0:
                        cv = np.std(revenues) / np.mean(revenues)
                        
                        if cv > self.thresholds['high_volatility']:
                            insights.append({
                                'title': 'High Revenue Volatility Detected',
                                'description': self.insight_templates['revenue_volatility'].format(
                                    drug=drug, 
                                    cv=cv
                                ),
                                'type': 'risk',
                                'importance': 6,
                                'actionable': True
                            })
        
        except Exception:
            pass
        
        return insights
    
    def _analyze_regional_performance(self, data):
        """Analyze regional performance patterns"""
        insights = []
        
        try:
            if 'Region' in data.columns:
                regional_revenue = data.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
                total_revenue = regional_revenue.sum()
                
                if total_revenue > 0 and len(regional_revenue) > 1:
                    top_region = regional_revenue.index[0]
                    top_region_share = (regional_revenue.iloc[0] / total_revenue) * 100
                    
                    insights.append({
                        'title': 'Regional Performance Leader',
                        'description': self.insight_templates['regional_insight'].format(
                            region=top_region, 
                            percentage=top_region_share
                        ),
                        'type': 'geographic',
                        'importance': 5,
                        'actionable': True
                    })
                    
                    # Identify underperforming regions
                    bottom_region = regional_revenue.index[-1]
                    bottom_region_share = (regional_revenue.iloc[-1] / total_revenue) * 100
                    
                    if bottom_region_share < 10 and len(regional_revenue) > 2:
                        insights.append({
                            'title': 'Regional Growth Opportunity',
                            'description': f"Region {bottom_region} shows potential for growth with only {bottom_region_share:.1f}% of total revenue.",
                            'type': 'opportunity',
                            'importance': 6,
                            'actionable': True
                        })
        
        except Exception:
            pass
        
        return insights
    
    def _analyze_pricing_opportunities(self, data):
        """Analyze pricing optimization opportunities"""
        insights = []
        
        try:
            if 'Price_Per_Unit' in data.columns and 'Units_Sold' in data.columns:
                drug_metrics = data.groupby('Drug').agg({
                    'Price_Per_Unit': 'mean',
                    'Units_Sold': 'sum',
                    'Revenue': 'sum'
                }).reset_index()
                
                # Calculate price elasticity indicators
                for _, row in drug_metrics.iterrows():
                    drug = row['Drug']
                    price = row['Price_Per_Unit']
                    units = row['Units_Sold']
                    revenue = row['Revenue']
                    
                    # Simple pricing opportunity detection
                    if price < drug_metrics['Price_Per_Unit'].median() and units > drug_metrics['Units_Sold'].median():
                        insights.append({
                            'title': 'Pricing Optimization Opportunity',
                            'description': self.insight_templates['price_opportunity'].format(
                                drug=drug, 
                                price=price
                            ),
                            'type': 'opportunity',
                            'importance': 7,
                            'actionable': True
                        })
        
        except Exception:
            pass
        
        return insights
    
    def _generate_strategic_recommendations(self, data):
        """Generate high-level strategic recommendations"""
        insights = []
        
        try:
            # Portfolio diversification analysis
            drug_count = data['Drug'].nunique()
            drug_revenues = data.groupby('Drug')['Revenue'].sum()
            revenue_distribution = drug_revenues / drug_revenues.sum()
            
            # Gini coefficient for inequality measure
            gini = self._calculate_gini_coefficient(revenue_distribution.values)
            
            if gini > 0.7:  # High inequality
                insights.append({
                    'title': 'Portfolio Rebalancing Recommended',
                    'description': f"Revenue distribution shows high concentration (Gini: {gini:.2f}). Consider diversifying portfolio to reduce risk.",
                    'type': 'strategic',
                    'importance': 8,
                    'actionable': True
                })
            
            # Market expansion opportunities
            if 'Region' in data.columns:
                region_count = data['Region'].nunique()
                if region_count < 4:
                    insights.append({
                        'title': 'Market Expansion Opportunity',
                        'description': f"Currently operating in {region_count} regions. Consider expanding to new geographic markets for growth.",
                        'type': 'strategic',
                        'importance': 6,
                        'actionable': True
                    })
            
            # Innovation pipeline assessment
            recent_data = data[data['Date'] >= (data['Date'].max() - timedelta(days=365))]
            if not recent_data.empty:
                recent_drugs = set(recent_data['Drug'].unique())
                all_drugs = set(data['Drug'].unique())
                
                if len(recent_drugs) == len(all_drugs):
                    insights.append({
                        'title': 'Innovation Pipeline Assessment',
                        'description': "No new drug launches detected in the past year. Consider strengthening R&D pipeline for future growth.",
                        'type': 'strategic',
                        'importance': 7,
                        'actionable': True
                    })
        
        except Exception:
            pass
        
        return insights
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        try:
            # Sort values
            sorted_values = np.sort(values)
            n = len(sorted_values)
            
            # Calculate Gini coefficient
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
            
            return max(0, min(1, gini))  # Ensure between 0 and 1
        
        except Exception:
            return 0
    
    def generate_forecast_insights(self, historical_data, forecast_data, accuracy_metrics):
        """Generate insights specific to forecasting results"""
        insights = []
        
        try:
            if accuracy_metrics and 'accuracy_percentage' in accuracy_metrics:
                accuracy = accuracy_metrics['accuracy_percentage']
                
                if accuracy > 85:
                    reliability = "high"
                elif accuracy > 70:
                    reliability = "moderate"
                else:
                    reliability = "low"
                
                insights.append({
                    'title': 'Forecast Reliability Assessment',
                    'description': self.insight_templates['forecast_accuracy'].format(
                        accuracy=accuracy, 
                        reliability=reliability
                    ),
                    'type': 'forecast',
                    'importance': 8,
                    'actionable': True
                })
                
                # Specific recommendations based on accuracy
                if reliability == "low":
                    insights.append({
                        'title': 'Data Quality Improvement Needed',
                        'description': f"Low forecast accuracy ({accuracy:.1f}%) suggests need for better data quality or additional external factors consideration.",
                        'type': 'data_quality',
                        'importance': 9,
                        'actionable': True
                    })
            
            # Analyze forecast trends
            if forecast_data and 'values' in forecast_data:
                forecast_values = forecast_data['values']
                if len(forecast_values) > 1:
                    forecast_trend = (forecast_values[-1] - forecast_values[0]) / forecast_values[0] * 100
                    
                    if forecast_trend > 10:
                        insights.append({
                            'title': 'Positive Forecast Trend',
                            'description': f"Forecast shows positive growth trend of {forecast_trend:.1f}% over the prediction period.",
                            'type': 'forecast',
                            'importance': 7,
                            'actionable': True
                        })
                    elif forecast_trend < -10:
                        insights.append({
                            'title': 'Declining Forecast Warning',
                            'description': f"Forecast indicates potential decline of {abs(forecast_trend):.1f}% over the prediction period. Consider intervention strategies.",
                            'type': 'forecast',
                            'importance': 8,
                            'actionable': True
                        })
        
        except Exception:
            pass
        
        return insights
    
    def prioritize_insights(self, insights):
        """Prioritize insights based on business impact and actionability"""
        try:
            # Define priority weights
            type_weights = {
                'risk': 1.2,
                'opportunity': 1.1,
                'strategic': 1.0,
                'performance': 0.9,
                'forecast': 0.8,
                'pattern': 0.7,
                'market_structure': 0.6,
                'geographic': 0.5,
                'data_quality': 1.3
            }
            
            # Calculate priority scores
            for insight in insights:
                base_importance = insight.get('importance', 5)
                insight_type = insight.get('type', 'general')
                is_actionable = insight.get('actionable', False)
                
                type_weight = type_weights.get(insight_type, 1.0)
                actionable_bonus = 1.1 if is_actionable else 1.0
                
                priority_score = base_importance * type_weight * actionable_bonus
                insight['priority_score'] = priority_score
            
            # Sort by priority score
            return sorted(insights, key=lambda x: x.get('priority_score', 0), reverse=True)
        
        except Exception:
            return insights