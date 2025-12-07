"""
Invoice Analyzer
Advanced analysis functions for invoice data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class InvoiceAnalyzer:
    def __init__(self, df):
        """Initialize analyzer with a dataframe"""
        self.df = df.copy()
        self.results = {}
    
    def spending_summary(self):
        """Get overall spending summary"""
        summary = {
            'total_invoices': len(self.df),
            'total_amount': self.df['amount'].sum(),
            'average_amount': self.df['amount'].mean(),
            'median_amount': self.df['amount'].median(),
            'min_amount': self.df['amount'].min(),
            'max_amount': self.df['amount'].max(),
            'std_amount': self.df['amount'].std()
        }
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            summary['date_range_start'] = self.df['date'].min()
            summary['date_range_end'] = self.df['date'].max()
            summary['date_span_days'] = (summary['date_range_end'] - summary['date_range_start']).days
        
        if 'vendor' in self.df.columns:
            summary['unique_vendors'] = self.df['vendor'].nunique()
            summary['most_frequent_vendor'] = self.df['vendor'].value_counts().index[0]
            summary['top_vendor_count'] = self.df['vendor'].value_counts().iloc[0]
        
        return summary
    
    def vendor_analysis(self):
        """Analyze spending by vendor"""
        if 'vendor' not in self.df.columns:
            return None
        
        vendor_stats = self.df.groupby('vendor').agg({
            'amount': ['count', 'sum', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)
        
        vendor_stats.columns = ['invoice_count', 'total_spent', 'avg_amount', 
                                'median_amount', 'std_amount', 'min_amount', 'max_amount']
        vendor_stats = vendor_stats.sort_values('total_spent', ascending=False)
        
        # Add percentage of total spending
        vendor_stats['pct_of_total'] = (vendor_stats['total_spent'] / 
                                         vendor_stats['total_spent'].sum() * 100).round(2)
        
        return vendor_stats
    
    def time_series_analysis(self):
        """Analyze spending over time"""
        if 'date' not in self.df.columns:
            return None
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Daily aggregation
        daily = self.df.groupby(self.df['date'].dt.date).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        daily.columns = ['total_amount', 'invoice_count', 'avg_amount']
        
        # Monthly aggregation
        monthly = self.df.groupby(self.df['date'].dt.to_period('M')).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        monthly.columns = ['total_amount', 'invoice_count', 'avg_amount']
        
        # Quarterly aggregation
        quarterly = self.df.groupby(self.df['date'].dt.to_period('Q')).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        quarterly.columns = ['total_amount', 'invoice_count', 'avg_amount']
        
        return {
            'daily': daily,
            'monthly': monthly,
            'quarterly': quarterly
        }
    
    def seasonal_patterns(self):
        """Identify seasonal spending patterns"""
        if 'date' not in self.df.columns:
            return None
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Day of week patterns
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        dow_stats = self.df.groupby('day_of_week')['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        # Month patterns
        self.df['month'] = self.df['date'].dt.month_name()
        month_stats = self.df.groupby('month')['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        # Quarter patterns
        self.df['quarter'] = 'Q' + self.df['date'].dt.quarter.astype(str)
        quarter_stats = self.df.groupby('quarter')['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        return {
            'day_of_week': dow_stats,
            'month': month_stats,
            'quarter': quarter_stats
        }
    
    def detect_anomalies(self, threshold=2.5):
        """Detect anomalous invoices based on Z-score"""
        results = []
        
        if 'vendor' in self.df.columns:
            # Detect anomalies per vendor
            for vendor in self.df['vendor'].unique():
                vendor_data = self.df[self.df['vendor'] == vendor]
                
                if len(vendor_data) > 2:  # Need at least 3 invoices
                    mean = vendor_data['amount'].mean()
                    std = vendor_data['amount'].std()
                    
                    if std > 0:
                        vendor_data['z_score'] = (vendor_data['amount'] - mean) / std
                        anomalies = vendor_data[abs(vendor_data['z_score']) > threshold]
                        
                        for idx, row in anomalies.iterrows():
                            results.append({
                                'invoice_number': row.get('invoice_number', idx),
                                'vendor': vendor,
                                'amount': row['amount'],
                                'expected_range': f"${mean - threshold*std:.2f} - ${mean + threshold*std:.2f}",
                                'z_score': row['z_score'],
                                'date': row.get('date', 'N/A')
                            })
        else:
            # Global anomaly detection
            mean = self.df['amount'].mean()
            std = self.df['amount'].std()
            
            if std > 0:
                self.df['z_score'] = (self.df['amount'] - mean) / std
                anomalies = self.df[abs(self.df['z_score']) > threshold]
                
                for idx, row in anomalies.iterrows():
                    results.append({
                        'invoice_number': row.get('invoice_number', idx),
                        'vendor': row.get('vendor', 'N/A'),
                        'amount': row['amount'],
                        'expected_range': f"${mean - threshold*std:.2f} - ${mean + threshold*std:.2f}",
                        'z_score': row['z_score'],
                        'date': row.get('date', 'N/A')
                    })
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def category_analysis(self):
        """Analyze spending by category if available"""
        if 'category' not in self.df.columns:
            return None
        
        category_stats = self.df.groupby('category').agg({
            'amount': ['count', 'sum', 'mean', 'median']
        }).round(2)
        
        category_stats.columns = ['invoice_count', 'total_spent', 'avg_amount', 'median_amount']
        category_stats = category_stats.sort_values('total_spent', ascending=False)
        
        # Add percentage
        category_stats['pct_of_total'] = (category_stats['total_spent'] / 
                                           category_stats['total_spent'].sum() * 100).round(2)
        
        return category_stats
    
    def payment_terms_analysis(self):
        """Analyze payment terms if available"""
        if 'payment_terms' not in self.df.columns:
            return None
        
        terms_stats = self.df.groupby('payment_terms').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        
        terms_stats.columns = ['invoice_count', 'total_amount', 'avg_amount']
        
        return terms_stats
    
    def tax_analysis(self):
        """Analyze tax patterns"""
        if 'tax' not in self.df.columns or 'amount' not in self.df.columns:
            return None
        
        # Calculate effective tax rates
        self.df['subtotal'] = self.df['amount'] - self.df['tax']
        self.df['tax_rate'] = (self.df['tax'] / self.df['subtotal'] * 100).replace([np.inf, -np.inf], 0)
        
        tax_stats = {
            'total_tax_paid': self.df['tax'].sum(),
            'avg_tax_rate': self.df['tax_rate'].mean(),
            'median_tax_rate': self.df['tax_rate'].median(),
            'common_tax_rates': self.df['tax_rate'].value_counts().head(5).to_dict()
        }
        
        return tax_stats
    
    def top_spending_periods(self, top_n=10):
        """Identify top spending days/months"""
        if 'date' not in self.df.columns:
            return None
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Top days
        top_days = self.df.groupby(self.df['date'].dt.date)['amount'].sum().sort_values(ascending=False).head(top_n)
        
        # Top months
        top_months = self.df.groupby(self.df['date'].dt.to_period('M'))['amount'].sum().sort_values(ascending=False).head(top_n)
        
        return {
            'top_days': top_days,
            'top_months': top_months
        }
    
    def spending_trends(self):
        """Calculate spending trends"""
        if 'date' not in self.df.columns:
            return None
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        monthly = self.df.groupby(self.df['date'].dt.to_period('M'))['amount'].sum()
        
        if len(monthly) < 2:
            return None
        
        # Calculate month-over-month growth
        mom_growth = monthly.pct_change() * 100
        
        # Calculate trend (simple linear regression)
        x = np.arange(len(monthly))
        y = monthly.values
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            trend = 'Increasing' if slope > 0 else 'Decreasing'
        else:
            slope = 0
            trend = 'Stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'avg_mom_growth': mom_growth.mean(),
            'mom_growth_series': mom_growth
        }
    
    def compare_periods(self, period1_start, period1_end, period2_start, period2_end):
        """Compare spending between two time periods"""
        if 'date' not in self.df.columns:
            return None
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        period1 = self.df[(self.df['date'] >= period1_start) & (self.df['date'] <= period1_end)]
        period2 = self.df[(self.df['date'] >= period2_start) & (self.df['date'] <= period2_end)]
        
        comparison = {
            'period1': {
                'total': period1['amount'].sum(),
                'count': len(period1),
                'average': period1['amount'].mean()
            },
            'period2': {
                'total': period2['amount'].sum(),
                'count': len(period2),
                'average': period2['amount'].mean()
            }
        }
        
        # Calculate changes
        comparison['change'] = {
            'total_pct': ((comparison['period2']['total'] - comparison['period1']['total']) / 
                         comparison['period1']['total'] * 100) if comparison['period1']['total'] > 0 else 0,
            'count_pct': ((comparison['period2']['count'] - comparison['period1']['count']) / 
                         comparison['period1']['count'] * 100) if comparison['period1']['count'] > 0 else 0,
            'average_pct': ((comparison['period2']['average'] - comparison['period1']['average']) / 
                           comparison['period1']['average'] * 100) if comparison['period1']['average'] > 0 else 0
        }
        
        return comparison
    
    def generate_full_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'summary': self.spending_summary(),
            'vendor_analysis': self.vendor_analysis(),
            'time_series': self.time_series_analysis(),
            'seasonal_patterns': self.seasonal_patterns(),
            'anomalies': self.detect_anomalies(),
            'category_analysis': self.category_analysis(),
            'tax_analysis': self.tax_analysis(),
            'top_periods': self.top_spending_periods(),
            'trends': self.spending_trends()
        }
        
        return report


def quick_analysis(df):
    """Quick analysis function for convenience"""
    analyzer = InvoiceAnalyzer(df)
    return analyzer.generate_full_report()


def main():
    """Test the analyzer with sample data"""
    print("=" * 60)
    print("INVOICE ANALYZER TEST")
    print("=" * 60)
    
    # Load sample data
    from pdf_extractor import InvoiceExtractor
    extractor = InvoiceExtractor()
    df = extractor.create_sample_data()
    
    # Run analysis
    analyzer = InvoiceAnalyzer(df)
    
    print("\n📊 SPENDING SUMMARY:")
    summary = analyzer.spending_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n🏢 TOP 5 VENDORS:")
    vendor_analysis = analyzer.vendor_analysis()
    if vendor_analysis is not None:
        print(vendor_analysis.head())
    
    print("\n🚨 ANOMALIES DETECTED:")
    anomalies = analyzer.detect_anomalies()
    if not anomalies.empty:
        print(f"  Found {len(anomalies)} anomalies")
        print(anomalies.head())
    else:
        print("  No anomalies detected")
    
    print("\n📈 SPENDING TRENDS:")
    trends = analyzer.spending_trends()
    if trends:
        print(f"  Trend: {trends['trend']}")
        print(f"  Avg Month-over-Month Growth: {trends['avg_mom_growth']:.2f}%")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()