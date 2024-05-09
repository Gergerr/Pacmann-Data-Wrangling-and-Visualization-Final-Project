import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
# Set page title and favicon
st.set_page_config(page_title='Credit Card Customer Analysis', page_icon=':credit_card:')

# Set custom CSS styles
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #097969;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        color: #FFD700;
    }
    .metric {
        font-size: 18px;
        font-weight: bold;
        color: #E5E4E2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to read data from a file
@st.cache_data
def read_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar for data upload and metric selection
st.sidebar.title('Data Upload')
uploaded_file = st.sidebar.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    data = read_data(uploaded_file)
    st.sidebar.success('Data uploaded successfully!')
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    # Metric selection
    metrics = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS',
               'PRC_FULL_PAYMENT', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'TENURE']
    selected_metrics = st.sidebar.multiselect('Select metrics to analyze', metrics)
    
    # Main content
    st.markdown('<div class="title">Credit Card Customer Analysis</div>', unsafe_allow_html=True)
    
    # Data overview
    st.markdown('<div class="subtitle">Data Overview</div>', unsafe_allow_html=True)
    first_data = st.text_input('Enter how many rows to display:', value='5')
    if first_data.isdigit():
        first_data = int(first_data)
        st.write(data.head(first_data))
    else:
        st.write('Please enter a valid number.')
    
    # Descriptive statistics and histogram
    if selected_metrics:
        st.markdown('<div class="subtitle">Descriptive Statistics and Histogram</div>', unsafe_allow_html=True)
        for metric in selected_metrics:
            st.markdown(f'<div class="metric">{metric}</div>', unsafe_allow_html=True)
            st.write(data[metric].describe())
            fig, ax = plt.subplots()
            sns.histplot(data[metric], kde=True, ax=ax)
            st.pyplot(fig)
    
    # Box plot
    if selected_metrics:
        st.markdown('<div class="subtitle">Box Plot</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data[selected_metrics], ax= ax)
        plt.xticks()
        st.pyplot(fig)
    
    # Balance and Credit Limit Category Distribution
    balance_credit_colorpalette = ['#097969', '#FFD700', '#E5E4E2']
    
    st.markdown('<div class="subtitle">Balance Category Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    balance_bar = sns.countplot(x='BALANCE_CATEGORY', data=data, palette=balance_credit_colorpalette, ax=ax)
    plt.title('Distribution of Balance Categories')
    plt.xlabel('Balance Categories')
    plt.ylabel('Number of Customers')
    for p in balance_bar.patches:
        balance_bar.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                             textcoords='offset points')
    st.pyplot(fig)
    
    st.markdown('<div class="subtitle">Credit Limit Category Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    credit_bar = sns.countplot(x='CREDIT_LIMIT_CATEGORY', data=data, palette=balance_credit_colorpalette, ax=ax)
    plt.title('Distribution of Credit Limit Categories')
    plt.xlabel('Credit Limit Categories')
    plt.ylabel('Number of Customers')
    for p in credit_bar.patches:
        credit_bar.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')
    st.pyplot(fig)
    
    # Mean and Median of Balance and Credit Limit Categories
    st.markdown('<div class="subtitle">Mean and Median of Balance and Credit Limit Categories</div>', unsafe_allow_html=True)
    balance_stats = data.groupby('BALANCE_CATEGORY')['BALANCE'].agg(['mean', 'median'])
    credit_stats = data.groupby('CREDIT_LIMIT_CATEGORY')['CREDIT_LIMIT'].agg(['mean', 'median'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('Balance Category Mean and Median')
        st.write(balance_stats)
    with col2:
        st.write('Credit Limit Category Mean and Median')
        st.write(credit_stats)
else:
    st.sidebar.warning('Please upload a CSV file to begin the analysis.')
    
    
    from scipy.stats import chi2_contingency

# Create tabs for each objective
tab1, tab2, tab3 = st.tabs(['Objective 1', 'Objective 2', 'Objective 3'])

with tab1:
    st.markdown('<div class="subtitle">Identifying High-Risk Customers</div>', unsafe_allow_html=True)
    
    # Categorize customers based on cash advance frequency and balance
    data['HIGH_RISK_CASH_ADVANCE'] = pd.cut(data['CASH_ADVANCE_FREQUENCY'], bins=[0, 0.1, 0.5, data['CASH_ADVANCE_FREQUENCY'].max()], labels=['Low', 'Medium', 'High'])
    data['HIGH_RISK_BALANCE'] = pd.cut(data['BALANCE'], bins=[0, 1000, 3000, data['BALANCE'].max()], labels=['Low', 'Medium', 'High'])
    
    # Group data to see average cash advance and balance in each category
    risk_group = data.groupby(['HIGH_RISK_CASH_ADVANCE', 'HIGH_RISK_BALANCE']).agg({
        'CASH_ADVANCE': 'mean',
        'BALANCE': 'mean',
        'CREDIT_LIMIT': 'mean',
        'PAYMENTS': 'mean'
    }).reset_index()
    
    # Toggle button for filtering
    balance_filter = st.selectbox('Balance Risk Filter', ['All', 'Low', 'Medium', 'High'])
    cash_advance_filter = st.selectbox('Cash Advance Risk Filter', ['All', 'Low', 'Medium', 'High'])

    if balance_filter != 'All' and cash_advance_filter != 'All':
        filtered_risk_group = data[(data['HIGH_RISK_BALANCE'] == balance_filter) & (data['HIGH_RISK_CASH_ADVANCE'] == cash_advance_filter)]
    elif balance_filter != 'All':
        filtered_risk_group = data[data['HIGH_RISK_BALANCE'] == balance_filter]
    elif cash_advance_filter != 'All':
        filtered_risk_group = data[data['HIGH_RISK_CASH_ADVANCE'] == cash_advance_filter]
    else:
        filtered_risk_group = data

    # Reorder columns to show CUST_ID, HIGH_RISK_BALANCE, and HIGH_RISK_CASH_ADVANCE first
    columns_to_show = ['CUST_ID', 'HIGH_RISK_BALANCE', 'HIGH_RISK_CASH_ADVANCE']
    remaining_columns = [col for col in data.columns if col not in columns_to_show]
    reordered_columns = columns_to_show + remaining_columns

    st.write(filtered_risk_group[reordered_columns])
    
    # Create heatmaps for average balance and credit limit
    balance_heatmap_data = data.pivot_table(index='HIGH_RISK_CASH_ADVANCE', columns='HIGH_RISK_BALANCE', values='BALANCE', aggfunc='mean')
    credit_heatmap_data = data.pivot_table(index='HIGH_RISK_CASH_ADVANCE', columns='HIGH_RISK_BALANCE', values='CREDIT_LIMIT', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(balance_heatmap_data, annot=True, fmt=".0f", cmap='Reds', cbar_kws={'label': 'Average Balance'}, ax=ax)
    ax.set_title('Risk Levels by Cash Advance Frequency and Balance')
    ax.set_xlabel('Balance Risk Level')
    ax.set_ylabel('Cash Advance Risk Level')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(credit_heatmap_data, annot=True, fmt=".0f", cmap='Blues', cbar_kws={'label': 'Average Credit Limit'}, ax=ax)
    ax.set_title('Risk Levels by Cash Advance Frequency and Credit Limit')
    ax.set_xlabel('Balance Risk Level')
    ax.set_ylabel('Cash Advance Risk Level')
    st.pyplot(fig)
    
    # Calculate risk score
    data['RISK_SCORE'] = (
        0.5 * pd.cut(data['BALANCE'], bins=3, labels=False) +
        0.3 * pd.cut(data['CASH_ADVANCE_FREQUENCY'], bins=3, labels=False) +
        0.2 * (1 - pd.cut(data['PRC_FULL_PAYMENT'], bins=3, labels=False))
    )
    
    # Visualize the distribution of the risk score
    fig, ax = plt.subplots(figsize=(10, 6))
    hist = sns.histplot(data['RISK_SCORE'], bins=10, kde=False, ax=ax)
    ax.set_title('Distribution of Composite Risk Score')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Number of Customers')
    
    # Interactive features for the risk score distribution plot
    show_counts = st.checkbox('Show Counts on Bars', value=False)
    if show_counts:
        for p in hist.patches:
            hist.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                          textcoords='offset points')
    
    color_palette = st.selectbox('Select Color Palette', ['Default', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys'])
    if color_palette != 'Default':
        hist.set_palette(sns.color_palette(color_palette))
    
    st.pyplot(fig)

    # Chi-Square Test between risk category and delinquency status
    data['DELINQUENCY_STATUS'] = (data['BALANCE'] > 3000).astype(int)
    contingency_table = pd.crosstab(data['BALANCE_CATEGORY'], data['DELINQUENCY_STATUS'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write(f"Chi-square test result -- chi2: {chi2}, p-value: {p}")
    
    # Group data by 'BALANCE_CATEGORY' and 'CREDIT_LIMIT_CATEGORY'
    grouped_data_balance = data.groupby('BALANCE_CATEGORY').agg({
        'CASH_ADVANCE_FREQUENCY': 'mean',
        'PRC_FULL_PAYMENT': 'mean',
        'CREDIT_LIMIT': 'mean',
        'PAYMENTS': 'mean'
    }).reset_index()
    
    grouped_data_credit = data.groupby('CREDIT_LIMIT_CATEGORY').agg({
        'CASH_ADVANCE_FREQUENCY': 'mean',
        'PRC_FULL_PAYMENT': 'mean',
        'BALANCE': 'mean',
        'PAYMENTS': 'mean'
    }).reset_index()
    
    # Plot average values by 'BALANCE_CATEGORY' and 'CREDIT_LIMIT_CATEGORY'
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    balance_bar1 = sns.barplot(x='BALANCE_CATEGORY', y='CASH_ADVANCE_FREQUENCY', data=grouped_data_balance, ax=axs[0, 0], palette=balance_credit_colorpalette)
    axs[0, 0].set_title('Average Cash Advance Frequency by Balance Category')
    
    balance_bar2 = sns.barplot(x='BALANCE_CATEGORY', y='PRC_FULL_PAYMENT', data=grouped_data_balance, ax=axs[0, 1], palette=balance_credit_colorpalette)
    axs[0, 1].set_title('Average Full Payment Rate by Balance Category')
    
    balance_bar3 = sns.barplot(x='BALANCE_CATEGORY', y='CREDIT_LIMIT', data=grouped_data_balance, ax=axs[1, 0], palette=balance_credit_colorpalette)
    axs[1, 0].set_title('Average Credit Limit by Balance Category')
    
    balance_bar4 = sns.barplot(x='BALANCE_CATEGORY', y='PAYMENTS', data=grouped_data_balance, ax=axs[1, 1], palette=balance_credit_colorpalette)
    axs[1, 1].set_title('Average Payments by Balance Category')
    
    plt.tight_layout()
    
    # Interactive features for the category plots
    show_counts_balance = st.checkbox('Show Counts', value=False)
    if show_counts_balance:
        for p in balance_bar1.patches:
            balance_bar1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
        for p in balance_bar2.patches:
            balance_bar2.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
        for p in balance_bar3.patches:
            balance_bar3.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
        for p in balance_bar4.patches:
            balance_bar4.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
    st.pyplot(fig)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    creditbar_1 = sns.barplot(x='CREDIT_LIMIT_CATEGORY', y='CASH_ADVANCE_FREQUENCY', data=grouped_data_credit, ax=axs[0, 0], palette=balance_credit_colorpalette)
    axs[0, 0].set_title('Average Cash Advance Frequency by Credit Limit Category')
    
    creditbar_2 = sns.barplot(x='CREDIT_LIMIT_CATEGORY', y='PRC_FULL_PAYMENT', data=grouped_data_credit, ax=axs[0, 1], palette=balance_credit_colorpalette)
    axs[0, 1].set_title('Average Full Payment Rate by Credit Limit Category')
    
    creditbar_3 = sns.barplot(x='CREDIT_LIMIT_CATEGORY', y='BALANCE', data=grouped_data_credit, ax=axs[1, 0], palette=balance_credit_colorpalette)
    axs[1, 0].set_title('Average Balance by Credit Limit Category')
    
    creditbar_4= sns.barplot(x='CREDIT_LIMIT_CATEGORY', y='PAYMENTS', data=grouped_data_credit, ax=axs[1, 1], palette=balance_credit_colorpalette)
    axs[1, 1].set_title('Average Payments by Credit Limit Category')
    
    plt.tight_layout()
    
    show_counts_credit = st.checkbox('Show Counts on Credit Limit Category Bars', value=False)
    if show_counts_credit:
        for p in creditbar_1.patches:
            creditbar_1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
        for p in creditbar_2.patches:
            creditbar_2.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
        for p in creditbar_3.patches:
            creditbar_3.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
        for p in creditbar_4.patches:
            creditbar_4.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')
    st.pyplot(fig)
    
    # Extract rows based on user input
    st.markdown('<div class="subtitle">Extract Rows by Risk Score</div>', unsafe_allow_html=True)
    risk_threshold = st.slider('Select Risk Score Threshold', min_value=0.0, max_value=data['RISK_SCORE'].max(), value=0.5, step=0.1)
    high_risk_customers = data[data['RISK_SCORE'] > risk_threshold]
    st.write(high_risk_customers)
    
with tab2:
    st.markdown('<div class="subtitle">Enhancing Customer Segmentation</div>', unsafe_allow_html=True)
    
    # Define custom bins based on the distribution observed from data['ONEOFF_PURCHASES'].describe() and similar for installments
    oneoff_edges = [-np.inf, 1, 100, 750, np.inf]  # Assuming most values are 0, set the first bin edge between 0 and 1
    installment_edges = [-np.inf, 1, 100, 600, np.inf]  # Adjust accordingly based on actual data
    
    # Categorizing customers based on custom bins
    data['SPENDING_TYPE'] = pd.cut(data['ONEOFF_PURCHASES'], bins=oneoff_edges, labels=['None', 'Low', 'Moderate', 'High'])
    data['INSTALLMENT_TYPE'] = pd.cut(data['INSTALLMENTS_PURCHASES'], bins=installment_edges, labels=['None', 'Low', 'Moderate', 'High'])
    
    # Viewing the new categorizations
     # Creating a table to show the categorization of ONEOFF_PURCHASES and INSTALLMENTS_PURCHASES
    oneoff_table_data = [
        ['≤ 1', 'NONE'],
        ['≤ 100', 'LOW'],
        ['≤ 750', 'MODERATE'],
        ['> 750', 'HIGH']
    ]
    
    installment_table_data = [
        ['≤ 1', 'NONE'],
        ['≤ 100', 'LOW'],
        ['≤ 600', 'MODERATE'],
        ['> 600', 'HIGH']
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('ONEOFF_PURCHASES Categorization')
        st.table(pd.DataFrame(oneoff_table_data, columns=['ONEOFF_PURCHASES', 'SPENDING_TYPE']))
    
    with col2:
        st.write('INSTALLMENTS_PURCHASES Categorization')
        st.table(pd.DataFrame(installment_table_data, columns=['INSTALLMENT_PURCHASES', 'INSTALLMENT_TYPE']))
    
    st.write('Note: I set a \'None\' class due to perhaps some customer has not made any one-off(single/large) purchase using their credit card.\nSame goes for the Installment type')
    # Creating credit utilization ratio
    data['UTILIZATION_RATIO'] = data['BALANCE'] / data['CREDIT_LIMIT']
    
    # Segmenting by utilization
    data['UTILIZATION_TYPE'] = pd.qcut(data['UTILIZATION_RATIO'], 4, labels=['Low', 'Moderate', 'High', 'Very High'])
    
    # Plotting the distribution of Credit Utilization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['UTILIZATION_RATIO'], bins=30, kde=True, color='purple', ax=ax)
    ax.set_title('Distribution of Credit Utilization Ratios')
    ax.set_xlabel('Credit Utilization Ratio')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)
    
    # Aggregate data by Spending and Utilization Type for visualization
    segment_analysis = data.groupby(['SPENDING_TYPE', 'UTILIZATION_TYPE']).agg({
        'BALANCE': 'mean',
        'PAYMENTS': 'mean',
        'CREDIT_LIMIT': 'mean',
        'CUST_ID': 'count'
    }).rename(columns={'CUST_ID': 'COUNT'}).reset_index()

    # Visualizing customer segments
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='UTILIZATION_TYPE', y='BALANCE', hue='SPENDING_TYPE', data=segment_analysis, ax=ax)
    ax.set_title('Customer Segments by Spending and Utilization Type')
    ax.set_ylabel('Average Balance')
    ax.set_xlabel('Utilization Type')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Filtering data based on SPENDING_TYPE and INSTALLMENT_TYPE
    st.markdown('### Filter Customers by Spending and Installment Type')
    spending_filter = st.selectbox('Select Spending Type', ['All', 'None', 'Low', 'Moderate', 'High'])
    installment_filter = st.selectbox('Select Installment Type', ['All', 'None', 'Low', 'Moderate', 'High'])
    
    if spending_filter != 'All' and installment_filter != 'All':
        filtered_data = data[(data['SPENDING_TYPE'] == spending_filter) & (data['INSTALLMENT_TYPE'] == installment_filter)]
    elif spending_filter != 'All':
        filtered_data = data[data['SPENDING_TYPE'] == spending_filter]
    elif installment_filter != 'All':
        filtered_data = data[data['INSTALLMENT_TYPE'] == installment_filter]
    else:
        filtered_data = data
    
    # Reorder columns to show CUST_ID, SPENDING_TYPE, and INSTALLMENT_TYPE first
    columns_to_show = ['CUST_ID', 'SPENDING_TYPE', 'INSTALLMENT_TYPE']
    remaining_columns = [col for col in filtered_data.columns if col not in columns_to_show]
    reordered_columns = columns_to_show + remaining_columns
    
    row_num = st.text_input('Enter how many rows to display: ', value='5')
    if row_num.isdigit():
        row_num = int(row_num)
        st.write(filtered_data[reordered_columns].head(row_num))
    else:
        st.write('Please enter a valid number.')
        
    # Code for Objective 3 goes here

with tab3:
    st.markdown('<div class="subtitle">Improving Financial Health</div>', unsafe_allow_html=True)
    
    # Analyzing credit utilization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['UTILIZATION_RATIO'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title('Credit Utilization Ratios')
    ax.set_xlabel('Utilization Ratio')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)
    
    # Identifying patterns correlating to healthy financial behavior
    financial_health_corr = data[['UTILIZATION_RATIO', 'PRC_FULL_PAYMENT', 'BALANCE', 'CREDIT_LIMIT']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(financial_health_corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation of Financial Health Indicators')
    st.pyplot(fig)
    
    # Additional analysis and visualizations for improving financial health
    st.markdown('### Customer Segmentation by Full Payment Rate')
    
    # Segment customers based on full payment rate
    full_payment_bins = [0, 0.25, 0.5, 0.75, 1]
    data['FULL_PAYMENT_CATEGORY'] = pd.cut(data['PRC_FULL_PAYMENT'], bins=full_payment_bins, labels=['Low', 'Medium', 'High', 'Very High']) 
    # Average balance and credit limit by full payment category
    full_payment_analysis = data.groupby('FULL_PAYMENT_CATEGORY').agg({
        'BALANCE': 'mean',
        'CREDIT_LIMIT': 'mean',
        'CUST_ID': 'count'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='FULL_PAYMENT_CATEGORY', y='BALANCE', data=full_payment_analysis, ax=ax)
    ax.set_title('Average Balance by Full Payment Category')
    ax.set_xlabel('Full Payment Category')
    ax.set_ylabel('Average Balance')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='FULL_PAYMENT_CATEGORY', y='CREDIT_LIMIT', data=full_payment_analysis, ax=ax)
    ax.set_title('Average Credit Limit by Full Payment Category')
    ax.set_xlabel('Full Payment Category')
    ax.set_ylabel('Average Credit Limit')
    st.pyplot(fig)
    
    # Recommendations for improving financial health
    st.markdown('### Recommendations for Improving Financial Health')
    st.write('1. Encourage customers with low full payment rates to pay their balances in full each month to avoid interest charges.')
    st.write('2. Offer financial education resources to help customers better understand credit utilization and its impact on their financial health.')
    st.write('3. Consider offering incentives or rewards for customers who maintain low credit utilization and high full payment rates.')
    st.write('4. Regularly monitor customer credit utilization and full payment rates to identify potential financial stress early on.')