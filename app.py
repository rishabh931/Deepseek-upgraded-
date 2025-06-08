# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import re

# Configure page
st.set_page_config(page_title="Stock P&L Analyzer", layout="wide")
st.title("📈 Indian Stock P&L Statement Analysis")

# Function to format large numbers
def format_number(num):
    if abs(num) >= 1e7:  # Crores
        return f"₹{num/1e7:,.2f} Cr"
    elif abs(num) >= 1e5:  # Lakhs
        return f"₹{num/1e5:,.2f} L"
    elif abs(num) >= 1000:  # Thousands
        return f"₹{num/1000:,.2f} K"
    return f"₹{num:,.2f}"

# Function to find stock symbol
def find_stock_symbol(query):
    # Convert to uppercase and remove extra spaces
    query = query.upper().strip()
    
    # Append .NS if not already present
    if not query.endswith('.NS'):
        query += '.NS'
    
    # Validate stock symbol format
    if not re.match(r"^[A-Z0-9.-]{1,20}\.NS$", query):
        return None
    
    return query

# Function to download financial data
def get_financials(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # Validate ticker
        if stock.info.get('regularMarketPrice') is None:
            return None, None, "Invalid stock symbol"
            
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        
        # Extract relevant data
        financial_data = {}
        
        # Revenue
        if 'Total Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Total Revenue'].head(4).values[::-1]
        elif 'Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Revenue'].head(4).values[::-1]
        else:
            return None, None, "Revenue data not available"
        
        # Operating Profit
        if 'Operating Income' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Income'].head(4).values[::-1]
        elif 'Operating Profit' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Profit'].head(4).values[::-1]
        else:
            return None, None, "Operating Profit data not available"
        
        # PBT
        if 'Pretax Income' in income_stmt.index:
            financial_data['PBT'] = income_stmt.loc['Pretax Income'].head(4).values[::-1]
        else:
            return None, None, "PBT data not available"
        
        # PAT
        if 'Net Income' in income_stmt.index:
            financial_data['PAT'] = income_stmt.loc['Net Income'].head(4).values[::-1]
        else:
            return None, None, "PAT data not available"
        
        # Shares Outstanding
        if 'Ordinary Shares Number' in balance_sheet.index:
            shares_outstanding = balance_sheet.loc['Ordinary Shares Number'].head(4).values[::-1]
        elif 'Share Issued' in balance_sheet.index:
            shares_outstanding = balance_sheet.loc['Share Issued'].head(4).values[::-1]
        else:
            # Try to get from info
            shares = stock.info.get('sharesOutstanding')
            if shares:
                shares_outstanding = np.array([shares] * 4)
            else:
                return None, None, "Shares outstanding data not available"
        
        # Calculate EPS
        financial_data['EPS'] = financial_data['PAT'] / (shares_outstanding / 1e6)  # Convert to millions
        
        # Calculate metrics
        financial_data['OPM %'] = (financial_data['Operating Profit'] / financial_data['Revenue']) * 100
        
        # Fixed EPS Growth % calculation
        eps_growth = [0]
        for i in range(1, len(financial_data['EPS'])):
            if financial_data['EPS'][i-1] != 0:
                growth = ((financial_data['EPS'][i] - financial_data['EPS'][i-1]) / 
                          abs(financial_data['EPS'][i-1]) * 100)
            else:
                growth = 0
            eps_growth.append(growth)
        financial_data['EPS Growth %'] = eps_growth
        
        # Get years
        years = income_stmt.columns[:4].strftime('%Y').values[::-1]
        
        return pd.DataFrame(financial_data, index=years), years, None
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Function to generate analysis insights
def generate_analysis(df, years):
    insights = []
    
    # Revenue analysis
    rev_growth = df['Revenue'].pct_change().dropna() * 100
    insights.append("**Revenue Analysis:**")
    insights.append(f"- Latest revenue: {format_number(df['Revenue'].iloc[-1])}")
    
    if len(df) > 1:
        cagr = ((df['Revenue'].iloc[-1] / df['Revenue'].iloc[0]) ** (1/(len(df)-1)) - 1) * 100
        insights.append(f"- {len(df)}-year CAGR: {cagr:.1f}%")
        insights.append(f"- Trend: {'Growth' if rev_growth.mean() > 0 else 'Decline'} averaging {rev_growth.mean():.1f}% YoY")
    else:
        insights.append("- Not enough data to compute growth rates")
    
    # OPM analysis
    insights.append("\n**Operating Profit Analysis:**")
    insights.append(f"- Current OPM: {df['OPM %'].iloc[-1]:.1f}%")
    
    if len(df) > 1:
        opm_trend = "improving" if df['OPM %'].iloc[-1] > df['OPM %'].iloc[0] else "declining"
        insights.append(f"- Trend: {opm_trend} ({df['OPM %'].iloc[0]:.1f}% → {df['OPM %'].iloc[-1]:.1f}%)")
        
        if df['Revenue'].pct_change().mean() != 0:
            operating_leverage = ((df['Operating Profit'].pct_change().mean() - 
                                  df['Revenue'].pct_change().mean())) * 100
            insights.append(f"- Operating leverage: {operating_leverage:.1f}%")
    else:
        insights.append("- Not enough data to analyze trends")
    
    # PAT analysis
    insights.append("\n**Profit After Tax Analysis:**")
    insights.append(f"- Latest PAT: {format_number(df['PAT'].iloc[-1])}")
    
    if len(df) > 1:
        pat_cagr = ((df['PAT'].iloc[-1] / df['PAT'].iloc[0]) ** (1/(len(df)-1)) - 1) * 100
        insights.append(f"- {len(df)}-year CAGR: {pat_cagr:.1f}%")
        
        margin_current = (df['PAT'] / df['Revenue']).iloc[-1]
        margin_initial = (df['PAT'] / df['Revenue']).iloc[0]
        margin_trend = 'Expanding' if margin_current > margin_initial else 'Contracting'
        insights.append(f"- Margin trend: {margin_trend} ({margin_initial*100:.1f}% → {margin_current*100:.1f}%)")
    else:
        insights.append("- Not enough data to analyze profit trends")
    
    # EPS analysis
    insights.append("\n**EPS Analysis:**")
    insights.append(f"- Latest EPS: ₹{df['EPS'].iloc[-1]:.1f}")
    
    if len(df) > 1:
        eps_growth = df['EPS Growth %'].dropna()
        if len(eps_growth) > 0:
            insights.append(f"- Average growth: {eps_growth.mean():.1f}%")
            insights.append(f"- Growth consistency: {'Stable' if eps_growth.std() < 15 else 'Volatile'}")
    else:
        insights.append("- Not enough data to analyze EPS growth")
    
    return "\n".join(insights)

# UI Components
st.subheader("Enter any Indian Stock Symbol")
stock_query = st.text_input("Search Stock (e.g., RELIANCE, INFY, TCS):", 
                           placeholder="Type stock name or symbol")

if stock_query:
    ticker_symbol = find_stock_symbol(stock_query)
    
    if not ticker_symbol:
        st.error("Invalid stock symbol format. Please use valid Indian stock symbols.")
    else:
        if st.button("Analyze P&L"):
            with st.spinner(f"Fetching {stock_query} financial data..."):
                financial_df, years, error = get_financials(ticker_symbol)
                
            if financial_df is not None and error is None:
                st.success(f"Data retrieved for {stock_query} ({ticker_symbol})")
                
                # Section 1: Revenue, Operating Profit, OPM%
                st.header("1. Revenue, Operating Profit & OPM% Analysis")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Financial Metrics")
                    # Create formatted display table
                    display_df = financial_df.copy()
                    display_df['Revenue'] = display_df['Revenue'].apply(format_number)
                    display_df['Operating Profit'] = display_df['Operating Profit'].apply(format_number)
                    display_df['OPM %'] = display_df['OPM %'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(display_df[['Revenue', 'Operating Profit', 'OPM %']])
                    
                    # Download button
                    csv = financial_df[['Revenue', 'Operating Profit', 'OPM %']].to_csv()
                    st.download_button(
                        label="Download Revenue Data",
                        data=csv,
                        file_name=f"{stock_query}_revenue_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.subheader("Performance Trend")
                    
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Bar plot for Revenue and Operating Profit
                    width = 0.35
                    x = np.arange(len(years))
                    ax1.bar(x - width/2, financial_df['Revenue']/1e7, width, 
                            label='Revenue (₹ Cr)', color='skyblue')
                    ax1.bar(x + width/2, financial_df['Operating Profit']/1e7, width, 
                            label='Op Profit (₹ Cr)', color='lightgreen')
                    ax1.set_ylabel('Amount (₹ Crores)')
                    
                    # Line plot for OPM%
                    ax2 = ax1.twinx()
                    ax2.plot(x, financial_df['OPM %'], 'r-o', linewidth=2, label='OPM %')
                    ax2.set_ylabel('OPM %', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    # Set y-axis limits dynamically
                    if max(financial_df['OPM %']) > 0:
                        ax2.set_ylim(0, max(financial_df['OPM %']) * 1.5)
                    
                    plt.title(f'{stock_query} Revenue & Profitability')
                    plt.xticks(x, years)
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    st.pyplot(fig)
                
                # Analysis insights
                st.subheader("Key Observations")
                st.markdown(generate_analysis(financial_df, years))
                
                # Section 2: PBT, PAT, EPS
                st.header("2. PBT, PAT & EPS Analysis")
                
                col3, col4 = st.columns([1, 2])
                
                with col3:
                    st.subheader("Profitability Metrics")
                    # Create formatted display table
                    display_df2 = financial_df.copy()
                    display_df2['PBT'] = display_df2['PBT'].apply(format_number)
                    display_df2['PAT'] = display_df2['PAT'].apply(format_number)
                    display_df2['EPS'] = display_df2['EPS'].apply(lambda x: f"₹{x:.1f}")
                    display_df2['EPS Growth %'] = display_df2['EPS Growth %'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(display_df2[['PBT', 'PAT', 'EPS', 'EPS Growth %']])
                    
                    # Download button
                    csv2 = financial_df[['PBT', 'PAT', 'EPS', 'EPS Growth %']].to_csv()
                    st.download_button(
                        label="Download Profitability Data",
                        data=csv2,
                        file_name=f"{stock_query}_profitability_data.csv",
                        mime="text/csv"
                    )
                
                with col4:
                    st.subheader("Profitability Trend")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # PAT and PBT trend
                    ax1.bar(years, financial_df['PAT']/1e7, color='lightblue', label='PAT (₹ Cr)')
                    ax1.bar(years, (financial_df['PBT'] - financial_df['PAT'])/1e7, 
                            bottom=financial_df['PAT']/1e7, color='lightcoral', label='Tax (₹ Cr)')
                    ax1.set_ylabel('Amount (₹ Crores)')
                    ax1.set_title('Profit Before Tax Composition')
                    ax1.legend()
                    
                    # EPS Growth
                    ax2.plot(years, financial_df['EPS'], 'g-o', label='EPS (₹)')
                    ax2.set_ylabel('EPS', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    
                    ax3 = ax2.twinx()
                    ax3.bar(years, financial_df['EPS Growth %'], alpha=0.3, color='purple', label='EPS Growth %')
                    ax3.set_ylabel('Growth %', color='purple')
                    ax3.tick_params(axis='y', labelcolor='purple')
                    ax3.axhline(0, color='grey', linestyle='--')
                    ax3.set_title('EPS Performance')
                    ax3.legend(loc='lower right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # GitHub resources
                st.header("📦 GitHub Resources")
                st.info("All code and data files are available in the GitHub repository:")
                st.markdown("[![GitHub](https://img.shields.io/badge/Repository-100000?logo=github)](https://github.com/yourusername/stock-pl-analysis)")
                
                resources = {
                    "app.py": "Main Streamlit application code",
                    "requirements.txt": "Python dependencies",
                    "stock_data.csv": "Sample dataset",
                    "analysis_template.ipynb": "Jupyter Notebook for analysis"
                }
                
                for file, description in resources.items():
                    with st.expander(f"Download {file}"):
                        st.write(description)
                        # Create dummy files for download
                        content = f"This is a sample {file} file for {stock_query} analysis"
                        st.download_button(
                            label=f"Download {file}",
                            data=content,
                            file_name=file,
                            mime="text/plain"
                        )
            else:
                st.error(f"Could not retrieve financial data: {error}")

# Popular stocks
st.markdown("### Popular Indian Stocks")
popular_cols = st.columns(5)
popular_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]

for i, stock in enumerate(popular_stocks):
    with popular_cols[i]:
        if st.button(stock, use_container_width=True):
            st.experimental_set_query_params(stock=stock)
            st.experimental_rerun()

# Handle query params
query_params = st.experimental_get_query_params()
if "stock" in query_params:
    stock_query = query_params["stock"][0]
    st.experimental_set_query_params()
    st.experimental_rerun()

# Add footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
- **Data Source**: Yahoo Finance
- **Financial Metrics**: Revenue, Operating Profit, OPM%, PBT, PAT, EPS
- **Analysis**: 4-year trend with visualizations and key insights
- **Number Formatting**: 
  - ₹1,000 = ₹1K 
  - ₹100,000 = ₹1L 
  - ₹10,000,000 = ₹1Cr
- **Updates**: Daily market data
""")
