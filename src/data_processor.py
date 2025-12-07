"""
Flexible Data Processor
Works with ANY invoice data structure dynamically
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FlexibleDataProcessor:
    def __init__(self, input_file='data/processed/sample_invoices.csv'):
        """Initialize flexible data processor"""
        self.input_file = Path(input_file)
        self.df = None
        self.column_mapping = {}
        self.detected_columns = {
            'date_col': None,
            'amount_col': None,
            'vendor_col': None,
            'tax_col': None,
            'invoice_id_col': None,
            'category_col': None
        }
        
    def load_data(self):
        """Load the invoice data"""
        if not self.input_file.exists():
            print(f"❌ File not found: {self.input_file}")
            return False
        
        self.df = pd.read_csv(self.input_file)
        print(f"✅ Loaded {len(self.df)} records from {self.input_file}")
        print(f"📊 Columns found: {list(self.df.columns)}")
        return True
    
    def detect_column_types(self):
        """Intelligently detect which columns contain what data"""
        print("\n🔍 Auto-detecting column types...")
        
        columns_lower = {col: col.lower() for col in self.df.columns}
        
        # Detect date column
        date_keywords = ['date', 'time', 'created', 'issued', 'timestamp']
        for col, col_lower in columns_lower.items():
            if any(keyword in col_lower for keyword in date_keywords):
                self.detected_columns['date_col'] = col
                break
        
        # Try to parse datetime for undetected date columns
        if not self.detected_columns['date_col']:
            for col in self.df.columns:
                try:
                    pd.to_datetime(self.df[col], errors='coerce')
                    if self.df[col].notna().sum() > len(self.df) * 0.5:
                        self.detected_columns['date_col'] = col
                        break
                except:
                    continue
        
        # Detect amount/total column
        amount_keywords = ['amount', 'total', 'sum', 'price', 'cost', 'value', 'grand']
        for col, col_lower in columns_lower.items():
            if any(keyword in col_lower for keyword in amount_keywords):
                if pd.api.types.is_numeric_dtype(self.df[col]) or self.df[col].dtype == 'object':
                    self.detected_columns['amount_col'] = col
                    break
        
        # Detect vendor column
        vendor_keywords = ['vendor', 'supplier', 'company', 'client', 'customer', 'seller', 'merchant']
        for col, col_lower in columns_lower.items():
            if any(keyword in col_lower for keyword in vendor_keywords):
                self.detected_columns['vendor_col'] = col
                break
        
        # Detect tax column
        tax_keywords = ['tax', 'vat', 'gst', 'duty']
        for col, col_lower in columns_lower.items():
            if any(keyword in col_lower for keyword in tax_keywords):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.detected_columns['tax_col'] = col
                    break
        
        # Detect invoice ID column
        id_keywords = ['invoice', 'id', 'number', 'reference', 'order']
        for col, col_lower in columns_lower.items():
            if any(keyword in col_lower for keyword in id_keywords) and 'date' not in col_lower:
                self.detected_columns['invoice_id_col'] = col
                break
        
        # Detect category column
        category_keywords = ['category', 'type', 'class', 'department', 'group']
        for col, col_lower in columns_lower.items():
            if any(keyword in col_lower for keyword in category_keywords):
                self.detected_columns['category_col'] = col
                break
        
        # Print detection results
        print("=" * 60)
        print("DETECTED COLUMNS:")
        for key, value in self.detected_columns.items():
            status = f"✅ {value}" if value else "⚠️ Not found"
            print(f"  {key.replace('_', ' ').title()}: {status}")
        print("=" * 60)
        
        return self.detected_columns
    
    def standardize_columns(self):
        """Create standardized column names"""
        print("\n🔧 Standardizing columns...")
        
        # Map detected columns to standard names
        if self.detected_columns['date_col']:
            self.df['date'] = pd.to_datetime(self.df[self.detected_columns['date_col']], errors='coerce')
        else:
            self.df['date'] = pd.Timestamp.now()
            print("  ⚠️ No date column found, using current date")
        
        if self.detected_columns['amount_col']:
            # Handle amount as string with currency symbols
            amount_col = self.df[self.detected_columns['amount_col']]
            if amount_col.dtype == 'object':
                self.df['amount'] = amount_col.str.replace(r'[$,£€¥]', '', regex=True).astype(float)
            else:
                self.df['amount'] = amount_col.astype(float)
        else:
            # Try to find any numeric column
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.df['amount'] = self.df[numeric_cols[0]]
                print(f"  ⚠️ Using {numeric_cols[0]} as amount column")
            else:
                raise ValueError("❌ No numeric column found for amounts")
        
        if self.detected_columns['vendor_col']:
            self.df['vendor'] = self.df[self.detected_columns['vendor_col']].astype(str)
        else:
            self.df['vendor'] = 'Unknown'
            print("  ⚠️ No vendor column found, using 'Unknown'")
        
        if self.detected_columns['tax_col']:
            tax_col = self.df[self.detected_columns['tax_col']]
            if tax_col.dtype == 'object':
                self.df['tax'] = tax_col.str.replace(r'[$,£€¥]', '', regex=True).astype(float)
            else:
                self.df['tax'] = tax_col.astype(float)
        else:
            self.df['tax'] = self.df['amount'] * 0.1
            print("  ⚠️ No tax column found, estimating as 10% of amount")
        
        if self.detected_columns['invoice_id_col']:
            self.df['invoice_number'] = self.df[self.detected_columns['invoice_id_col']].astype(str)
        else:
            self.df['invoice_number'] = [f'INV-{i:06d}' for i in range(len(self.df))]
            print("  ⚠️ No invoice ID found, generating sequential IDs")
        
        if self.detected_columns['category_col']:
            self.df['category'] = self.df[self.detected_columns['category_col']].astype(str)
        
        print(f"✅ Standardized to {len(self.df.columns)} columns")
        return self.df
    
    def clean_data(self):
        """Clean and validate the data"""
        print("\n🧹 Cleaning data...")
        
        initial_count = len(self.df)
        
        # Remove duplicates
        if 'invoice_number' in self.df.columns:
            self.df.drop_duplicates(subset=['invoice_number'], inplace=True)
        else:
            self.df.drop_duplicates(inplace=True)
        print(f"  - Removed {initial_count - len(self.df)} duplicates")
        
        # Handle missing values
        if 'vendor' in self.df.columns:
            self.df['vendor'].fillna('Unknown', inplace=True)
        
        if 'amount' in self.df.columns:
            self.df['amount'].fillna(self.df['amount'].median(), inplace=True)
            self.df = self.df[self.df['amount'] > 0]
        
        if 'tax' in self.df.columns:
            self.df['tax'].fillna(0, inplace=True)
        
        if 'date' in self.df.columns:
            self.df = self.df.dropna(subset=['date'])
        
        print(f"  - Final dataset: {len(self.df)} records")
        return self.df
    
    def feature_engineering(self):
        """Create features dynamically based on available data"""
        print("\n🔧 Engineering features...")
        
        # Time-based features (if date exists)
        if 'date' in self.df.columns:
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['day_of_month'] = self.df['date'].dt.day
            self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            self.df['is_month_start'] = (self.df['day_of_month'] <= 5).astype(int)
            self.df['is_month_end'] = (self.df['day_of_month'] >= 25).astype(int)
            self.df['is_quarter_end'] = self.df['month'].isin([3, 6, 9, 12]).astype(int)
        
        # Financial features
        if 'amount' in self.df.columns and 'tax' in self.df.columns:
            self.df['subtotal'] = self.df['amount'] - self.df['tax']
            self.df['tax_rate'] = (self.df['tax'] / self.df['subtotal'] * 100).replace([np.inf, -np.inf], 0)
            self.df['total_amount'] = self.df['amount']
        
        # Amount categorization
        if 'amount' in self.df.columns:
            self.df['amount_category'] = pd.cut(
                self.df['amount'],
                bins=[0, 100, 500, 1000, 5000, float('inf')],
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
            )
            
            # Amount quantiles
            self.df['amount_quantile'] = pd.qcut(
                self.df['amount'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop'
            )
        
        # Vendor-based features (if vendor exists)
        if 'vendor' in self.df.columns:
            self.df = self.df.sort_values('date') if 'date' in self.df.columns else self.df
            
            # Vendor statistics
            vendor_stats = self.df.groupby('vendor')['amount'].agg(['mean', 'std', 'count', 'sum']).reset_index()
            vendor_stats.columns = ['vendor', 'vendor_avg_amount', 'vendor_std_amount', 
                                   'vendor_invoice_count', 'vendor_total_spent']
            self.df = self.df.merge(vendor_stats, on='vendor', how='left')
            
            # Fill NaN std with 0 for vendors with single invoice
            self.df['vendor_std_amount'].fillna(0, inplace=True)
            
            # Days since last invoice per vendor
            if 'date' in self.df.columns:
                self.df['days_since_last_invoice'] = self.df.groupby('vendor')['date'].diff().dt.days
                self.df['days_since_last_invoice'].fillna(0, inplace=True)
            
            # Vendor frequency
            self.df['vendor_frequency'] = self.df.groupby('vendor')['vendor'].transform('count')
        
        # Category-based features (if category exists)
        if 'category' in self.df.columns:
            category_stats = self.df.groupby('category')['amount'].agg(['mean', 'count']).reset_index()
            category_stats.columns = ['category', 'category_avg_amount', 'category_count']
            self.df = self.df.merge(category_stats, on='category', how='left')
        
        # Rolling statistics (if date exists)
        if 'date' in self.df.columns:
            self.df = self.df.sort_values('date')
            self.df['rolling_avg_7d'] = self.df['amount'].rolling(window=7, min_periods=1).mean()
            self.df['rolling_avg_30d'] = self.df['amount'].rolling(window=30, min_periods=1).mean()
        
        print(f"  - Created {len(self.df.columns)} total features")
        return self.df
    
    def create_target_variable(self, prediction_type='amount'):
        """Create target variable for prediction"""
        print(f"\n🎯 Creating target variable: {prediction_type}")
        
        if prediction_type == 'amount':
            if 'vendor' in self.df.columns:
                self.df['target'] = self.df.groupby('vendor')['amount'].shift(-1)
            else:
                self.df['target'] = self.df['amount'].shift(-1)
                
        elif prediction_type == 'category':
            if 'amount_category' in self.df.columns:
                self.df['target'] = self.df['amount_category']
            else:
                raise ValueError("Run feature_engineering first")
                
        elif prediction_type == 'anomaly':
            if 'vendor_avg_amount' in self.df.columns:
                z_scores = (self.df['amount'] - self.df['vendor_avg_amount']) / (self.df['vendor_std_amount'] + 1e-6)
                self.df['target'] = (np.abs(z_scores) > 2).astype(int)
            else:
                # Simple anomaly detection without vendor
                mean = self.df['amount'].mean()
                std = self.df['amount'].std()
                z_scores = (self.df['amount'] - mean) / std
                self.df['target'] = (np.abs(z_scores) > 2).astype(int)
        
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['target'])
        print(f"  - Valid samples: {len(self.df)} (removed {initial_len - len(self.df)} NaN)")
        
        return self.df
    
    def get_statistics(self):
        """Get statistics about the data"""
        if self.df is None:
            return None
        
        stats = {
            'total_records': len(self.df),
            'total_amount': self.df['amount'].sum() if 'amount' in self.df.columns else 0,
            'avg_amount': self.df['amount'].mean() if 'amount' in self.df.columns else 0,
            'median_amount': self.df['amount'].median() if 'amount' in self.df.columns else 0,
            'columns': list(self.df.columns)
        }
        
        if 'date' in self.df.columns:
            stats['date_range'] = f"{self.df['date'].min()} to {self.df['date'].max()}"
        
        if 'vendor' in self.df.columns:
            stats['unique_vendors'] = self.df['vendor'].nunique()
            stats['top_vendors'] = self.df['vendor'].value_counts().head(5).to_dict()
        
        if 'category' in self.df.columns:
            stats['unique_categories'] = self.df['category'].nunique()
        
        return stats
    
    def save_processed_data(self, output_file='data/processed/processed_invoices.csv'):
        """Save the processed data"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Processed data saved to: {output_path}")
        return output_path
    
    def get_dataframe(self):
        """Return the processed dataframe"""
        return self.df


# Alias for backward compatibility
DataProcessor = FlexibleDataProcessor


def main():
    """Main execution function"""
    print("=" * 60)
    print("FLEXIBLE DATA PROCESSOR")
    print("Works with ANY invoice format!")
    print("=" * 60)
    
    processor = FlexibleDataProcessor()
    
    if not processor.load_data():
        return
    
    processor.detect_column_types()
    processor.standardize_columns()
    processor.clean_data()
    processor.feature_engineering()
    processor.create_target_variable(prediction_type='amount')
    
    stats = processor.get_statistics()
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total Records: {stats['total_records']}")
    if 'date_range' in stats:
        print(f"Date Range: {stats['date_range']}")
    print(f"Total Amount: ${stats['total_amount']:,.2f}")
    print(f"Average Amount: ${stats['avg_amount']:,.2f}")
    if 'unique_vendors' in stats:
        print(f"Unique Vendors: {stats['unique_vendors']}")
    if 'unique_categories' in stats:
        print(f"Unique Categories: {stats['unique_categories']}")
    
    processor.save_processed_data()
    
    print("\n✅ Ready for model training!")


if __name__ == "__main__":
    main()