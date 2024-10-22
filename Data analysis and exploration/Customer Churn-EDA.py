#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction and Analysis.
# ## Project Overview:
# 1. Identify key factors leading to customer churn using descriptive statistics and visualizations.  
# 2. Prepare the dataset for analysis by cleaning, normalizing, and encoding relevant features.  
# 3. Apply predictive modeling techniques (e.g., logistic regression, decision trees) and evaluate their performance using appropriate metrics (e.g., accuracy, precision, recall).  
# 4. Provide recommendations based on your findings that may help reduce churn rates.
# 
# ## 1. Data Exploration
# **Load and Explore the Dataset:**
# We begin by loading the dataset to get an initial understanding of its structure and the nature of the data.
# * Inspect the data to understand its structure, types, and completeness.
# * Summarize key statistics of the dataset (mean, median, missing values).

# In[2]:


# Load the dataset
import pandas as pd
data = pd.read_csv('Customer Churn.csv')
data.head()


# In[3]:


data.info()  # Check for data types and missing values
data.describe()  # Summary statistics


# **Observations:**
# * No missing values.
# * Most columns are integers except for Customer Value, which is a float.
# * The Churn column is the target variable for prediction.

# ## 2. Data Cleaning and Preparation
# * Handle missing values and duplicates (if any).

# In[6]:


# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}")


# ***The data has no missing values but has 300 duplicate rows.***

# In[8]:


# Remove duplicate rows
data_cleaned = data.drop_duplicates()

# Verify that duplicates are removed
print(f'Number of rows after removing duplicates: {data_cleaned.shape[0]}')


# In[9]:


data_cleaned.info()  # Check for data types and missing values


# In[10]:


# Clean column names (remove extra spaces and strip them)
data_cleaned.columns = data_cleaned.columns.str.replace('  ', ' ').str.strip()

# Confirm cleaned column names
print("Cleaned column names:", data_cleaned.columns)


# In[11]:


data_cleaned.describe()  # Summary statistics


# ## Descriptive Statistics:
# 
# - **Call Failure**: On average, users experience 7.80 call failures, with a maximum of 36.
# - **Complains**: The average complaint rate is 0.08, most customers don't complain (75% have 0 complaints), with very few having 1 complaint.
# - **Subscription Length**: Customers have been subscribed for an average of 32.45 months, with a maximum of 47 months.
# - **Charge Amount**: Average charge amount is low at 0.97, with a maximum of 10.
# - **Seconds of Use**: Users spend an average of 4534 seconds using the service, with some using it as much as 17,090 seconds.
# - **Frequency of Use**: The average frequency of use is 70.48, though it ranges up to 255.
# - **Frequency of SMS**: Customers send an average of 73.79 SMS, but some send as many as 522.
# - **Distinct Called Numbers**: Users call an average of 23.87 distinct numbers.
# - **Age Group**: The average falls around 2.84, likely representing customers in the 2-3 age group category.
# - **Tariff Plan**: Most customers fall under the 1st tariff plan (mean = 1.08).
# - **Status**: The average status is 1.24, indicating most customers are active.
# - **Age**: The average age of users is 31.08, with a maximum of 55 years.
# - **Customer Value**: The average lifetime value of customers is 474.99, with a maximum of 2165.28.
# - **Churn**: The average churn rate is 0.156, meaning 15.6% of customers have churned.

# #### Churn Rate Distribution:
# - **Churn Rate:**
#   - **84.4%** of customers did not churn, while **15.6%** did churn.
#   - Class imbalance observed, which could affect model performance.

# In[ ]:





# In[14]:


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Churn rate pie chart
churn_counts = data_cleaned['Churn'].value_counts()
churn_labels = ['No Churn', 'Churn']

fig_churn = px.pie(values=churn_counts, names=churn_labels,
                   title="Overall Churn Rate",
                   color_discrete_sequence=px.colors.sequential.RdBu)

fig_churn.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# Corrected countplot with 'hue'
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data_cleaned, hue='Churn', palette='coolwarm', legend=False)
plt.title('Churn Distribution')
plt.show()


# ## 3. Exploratory Data Analysis (EDA)
# These visualizations will help to understand dataset better and prepare it for predictive modeling.

# #### 1. Histograms for Distributions
# * Histograms will help to understand the distribution of numerical variables.

# In[18]:


# Get the list of numeric features
numeric_features = data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print the numeric features
print("Numeric features:", numeric_features)


# In[19]:


# Plot histograms for numeric features
import math

# Dynamic grid for subplots
num_features = len(numeric_features)
num_cols = 3
num_rows = math.ceil(num_features / num_cols)

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    sns.histplot(data=data_cleaned, x=feature, hue='Churn', multiple='stack', ax=axes[i], palette='coolwarm')
    axes[i].set_title(f'Distribution of {feature} by Churn')

plt.tight_layout()
plt.show()


# #### 2. Call Failures vs. Churn (Box Plot):

# In[21]:


# Create a histogram for Call Failures segmented by Churn status
fig = px.histogram(data_cleaned, 
                   x='Call Failure', 
                   color='Churn', 
                   barmode='overlay', 
                   labels={'Call Failure': 'Call Failures', 'Churn': 'Churn Status'},
                   title="Call Failure Distribution by Churn Status",
                   opacity=0.75)  # Optional: Adjust the opacity for better visibility

# Show the figure
fig.show()


# #### 2. Complains Distribution by Churn status

# In[23]:


# Create a histogram for Complains segmented by Churn status with 2 bins (0 and 1)
fig = px.histogram(data, 
                   x='Complains', 
                   color='Churn', 
                   barmode='overlay', 
                   nbins=3,  # Set number of bins to 2 (for values 0 and 1)
                   labels={'Complains': 'Complains', 'Churn': 'Churn Status'},
                   title="Complains Distribution by Churn Status",
                   opacity=0.75)  # Adjust the opacity for better visibility

# Update x-axis to limit the range from 0 to 1
fig.update_layout(xaxis=dict(range=[0, 1]))

# Show the figure
fig.show()


# In[24]:


# Group data by Churn and count the number of Complains (0 or 1) for each Churn status
complain_count_by_churn = data.groupby('Churn')['Complains'].sum().reset_index()

# Create a bar chart where x-axis represents Churn (0 = not churned, 1 = churned)
# and y-axis represents the count of Complains for each Churn status
fig = px.bar(complain_count_by_churn, 
             x='Churn', 
             y='Complains', 
             labels={'Churn': 'Churn Status', 'Complains': 'Number of Complaints'},
             title="Complains Distribution by Churn Status",
             text='Complains')  # Show number of complaints on the bars

# Update x-axis ticks to show 0 and 1 explicitly
fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Not Churned', 'Churned']))

# Show the figure
fig.show()


# #### 3. Seconds of Use vs. Churn

# In[26]:


# Histogram to show 'Seconds of Use' distribution by 'Churn' status
fig_seconds = px.histogram(data_cleaned, x='Seconds of Use', 
                           color='Churn', 
                           title="Seconds of Use by Churn Status",
                           nbins=50, 
                           barmode='overlay', 
                           color_discrete_sequence=px.colors.sequential.Inferno)

fig_seconds.show()


# #### 4. Frequency of SMS by Churn status

# In[28]:


# Create a histogram for Frequency of SMS segmented by Churn status
fig = px.histogram(data_cleaned, 
                   x='Frequency of SMS', 
                   color='Churn', 
                   barmode='overlay', 
                   labels={'Frequency of SMS': 'Frequency of SMS', 'Churn': 'Churn Status'},
                   title="Frequency of SMS Distribution by Churn Status",
                   opacity=0.75)  # Optional: Adjust the opacity for better visibility

# Show the figure
fig.show()


# #### 5. Frequency of Use vs. Churn

# In[30]:


# Histogram to show 'Frequency of Use' distribution by 'Churn' status
fig_frequency = px.histogram(data_cleaned, x='Frequency of use', 
                             color='Churn', 
                             title="Frequency of Use by Churn Status",
                             nbins=50, 
                             barmode='overlay', 
                             color_discrete_sequence=px.colors.sequential.Inferno)

fig_frequency.show()


# #### 6. Customer Value vs. Churn

# In[32]:


# Create a box plot for Customer Value vs. Churn
fig = px.box(data_cleaned, 
             x='Churn', 
             y='Customer Value',
             labels={'Churn': 'Churn Status', 'Customer Value': 'Customer Value'},
             title="Customer Value Distribution by Churn Status")

# Show the figure
fig.show()


# #### 7. Subscription Length vs. Churn

# In[34]:


# Create a histogram for Subscription Length vs. Churn
fig = px.histogram(data_cleaned, 
                   x='Subscription Length', 
                   color='Churn', 
                   barmode='overlay', 
                   labels={'Subscription Length': 'Subscription Length', 'Churn': 'Churn Status'},
                   title="Subscription Length Distribution by Churn Status")

# Show the figure
fig.show()


# #### 8. Histogram for Distinct Called Numbers by Churn Status

# In[36]:


# Create a histogram for Distinct Called Numbers segmented by Churn status
fig = px.histogram(data_cleaned, 
                   x='Distinct Called Numbers', 
                   color='Churn', 
                   barmode='overlay', 
                   labels={'Distinct Called Numbers': 'Distinct Called Numbers', 'Churn': 'Churn Status'},
                   title="Distinct Called Numbers Distribution by Churn Status",
                   opacity=0.75)  # Optional: Adjust the opacity for better visibility

# Show the figure
fig.show()


# #### 9. Correlation Heatmap

# In[38]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[39]:


# Sort correlations with respect to 'Churn' in descending order
churn_correlation = data_cleaned.corr()['Churn'].sort_values(ascending=False)

# Display the sorted correlation values for 'Churn'
print(churn_correlation)


# **The visualizations provide the following insights:**
# 
# 1. **Churn Distribution:** As expected, the dataset is imbalanced, with the majority of customers not churning (about 84%).
# 
# 2. **Key Feature Distributions:**
#    * **Call Failure:** Customers with more call failures tend to churn more frequently, suggesting a potential link between poor service quality and churn.
#    * **Complains:** A small portion of customers raise complaints, but those who do are more likely to churn.
#    * **Subscription Length:** Customers with shorter subscription lengths appear to churn more frequently.
#    * **Seconds of Use:** Those with lower usage seem more prone to churn.
#    * **Frequency of Use and SMS:** Lower usage and fewer SMS seem to correlate with higher churn rates.
#    * **Distinct Called Numbers:** Customers with fewer distinct numbers called are more likely to churn.
#    * **Age and Age Group:** No strong pattern, but younger customers might churn slightly more.
#    * **Customer Value:** Lower customer value tends to be associated with higher churn rates.

# ### Interactive Dashboard

# In[42]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Ensure a clean copy of the DataFrame to avoid warnings
dashboard_data = data_cleaned.copy()  # Use the renamed variable

# Map the Churn column to 'No Churn' and 'Churn' labels and convert to categorical
dashboard_data['Churn'] = dashboard_data['Churn'].map({0: 'No Churn', 1: 'Churn'}).astype('category')

# Create the initial figure layout with subplots
fig = make_subplots(
    rows=3, cols=2, 
    subplot_titles=("Frequency of Use by Churn Status", "Call Failure Distribution", 
                    "Complains Distribution", "Seconds of Use", 
                    "Subscription Length Distribution", "Distinct Called Numbers Distribution"),
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "xy"}]]
)

# 1. Frequency of Use Histogram
fig_frequency = px.histogram(dashboard_data, x='Frequency of use', 
                             color='Churn', 
                             nbins=50, 
                             barmode='overlay', 
                             color_discrete_sequence=['blue', 'orange'])  # Set specific colors

fig.add_trace(fig_frequency['data'][0], row=1, col=1)
fig.add_trace(fig_frequency['data'][1], row=1, col=1)

# 2. Call Failures Histogram
call_failure_hist = px.histogram(dashboard_data, x='Call Failure', color='Churn', barmode='overlay',
                                 labels={'Call Failure': 'Call Failures', 'Churn': 'Churn Status'}, 
                                 opacity=0.75, color_discrete_sequence=['blue', 'orange'])
fig.add_trace(call_failure_hist['data'][0], row=1, col=2)
fig.add_trace(call_failure_hist['data'][1], row=1, col=2)

# 3. Complains Histogram
complains_hist = px.histogram(dashboard_data, x='Complains', color='Churn', barmode='overlay', 
                              nbins=2, labels={'Complains': 'Complains', 'Churn': 'Churn Status'}, 
                              opacity=0.75, color_discrete_sequence=['blue', 'orange'])
fig.add_trace(complains_hist['data'][0], row=2, col=1)
fig.add_trace(complains_hist['data'][1], row=2, col=1)

# 4. Seconds of Use Histogram
seconds_use_hist = px.histogram(dashboard_data, x='Seconds of Use', color='Churn', barmode='overlay', 
                                nbins=50, labels={'Seconds of Use': 'Seconds of Use', 
                                'Churn': 'Churn Status'}, opacity=0.75, 
                                color_discrete_sequence=['blue', 'orange'])
fig.add_trace(seconds_use_hist['data'][0], row=2, col=2)
fig.add_trace(seconds_use_hist['data'][1], row=2, col=2)

# 5. Subscription Length Histogram
subscription_length_hist = px.histogram(dashboard_data, x='Subscription Length', color='Churn', 
                                        barmode='overlay', labels={'Subscription Length': 'Subscription Length', 
                                        'Churn': 'Churn Status'}, opacity=0.75, 
                                        color_discrete_sequence=['blue', 'orange'])
fig.add_trace(subscription_length_hist['data'][0], row=3, col=1)
fig.add_trace(subscription_length_hist['data'][1], row=3, col=1)

# 6. Distinct Called Numbers Histogram
distinct_numbers_hist = px.histogram(dashboard_data, x='Distinct Called Numbers', color='Churn', 
                                     barmode='overlay', labels={'Distinct Called Numbers': 'Distinct Called Numbers', 
                                     'Churn': 'Churn Status'}, opacity=0.75, 
                                     color_discrete_sequence=['blue', 'orange'])
fig.add_trace(distinct_numbers_hist['data'][0], row=3, col=2)
fig.add_trace(distinct_numbers_hist['data'][1], row=3, col=2)

# Update layout and add dropdown for interactivity
fig.update_layout(
    title_text="Customer Churn Dashboard", height=900,
    updatemenus=[
        {
            'buttons': [
                {
                    'method': 'restyle',
                    'label': 'All Customers',
                    'args': [{'visible': [True] * 12}]  # Show all traces
                },
                {
                    'method': 'restyle',
                    'label': 'No Churn',
                    'args': [{'visible': [True, False] * 6}]  # Show "No Churn" traces
                },
                {
                    'method': 'restyle',
                    'label': 'Churn',
                    'args': [{'visible': [False, True] * 6}]  # Show "Churn" traces
                }
            ],
            'direction': 'down',
            'showactive': True
        }
    ],
    showlegend=False  # Removing the legend
)

# Show the interactive dashboard
fig.show()


# # Insights from Customer Churn Dashboard
# 
# ## Overview
# The Customer Churn Dashboard provides a comprehensive overview of user behavior and churn status. By analyzing various metrics such as frequency of use, seconds of use, call failures, complaints, subscription lengths, and distinct called numbers, we gain valuable insights into the factors contributing to customer retention and churn.
# 
# ## Key Insights
# 
# ### 1. Frequency of Use
# - **Churned Customers**: Customers who churn tend to have a significantly lower frequency of use compared to those who remain.
# - **Retention Strategy**: Increasing engagement and usage frequency could be pivotal in retaining customers.
# 
# ### 2. Seconds of Use
# - **Use Duration**: Customers with higher usage duration (in seconds) are less likely to churn.
# - **Focus on High Usage**: Strategies to increase the overall usage time per customer may lead to better retention.
# 
# ### 3. Call Failures
# - **Impact of Call Failures**: A higher number of call failures correlates with increased churn rates.
# - **Improvement Areas**: Addressing technical issues and improving service reliability should be prioritized to reduce churn.
# 
# ### 4. Complaints
# - **Correlation with Churn**: Customers who register more complaints are more likely to churn.
# - **Customer Support**: Enhancing customer support and effectively addressing complaints can mitigate churn risks.
# 
# ### 5. Subscription Length
# - **Loyalty Indicator**: Longer subscription lengths are associated with lower churn rates.
# - **Incentives for Long-Term Commitment**: Offering incentives for longer subscriptions may encourage retention.
# 
# ### 6. Distinct Called Numbers
# - **Variety in Usage**: Customers who engage with a broader range of services or contacts are less likely to churn.
# - **Cross-Promotion Strategies**: Encouraging customers to explore different services can enhance their overall experience and reduce churn.
# 
# ## Recommendations
# - **Engagement Programs**: Develop targeted programs to increase the frequency and duration of use.
# - **Technical Improvements**: Focus on minimizing call failures through better infrastructure and customer service.
# - **Complaint Resolution**: Establish robust complaint management systems to improve customer satisfaction.
# - **Incentives for Long-Term Plans**: Consider introducing loyalty programs or discounts for longer subscription commitments.
# - **Cross-Functional Marketing**: Use marketing campaigns to promote the use of diverse services, ensuring customers are aware of all available options.
# 
# ## Conclusion
# The analysis highlights critical factors influencing customer churn, enabling targeted strategies to improve retention rates. By focusing on customer engagement, service reliability, and complaint resolution, we can foster a more loyal customer base and enhance overall satisfaction.
# 
