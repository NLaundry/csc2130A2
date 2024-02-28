import json
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Helper function to extract the Model value from ChatgptSharing
def extract_model(chatgpt_sharing):
    if chatgpt_sharing and 'Model' in chatgpt_sharing[0]:
        return chatgpt_sharing[0]['Model']
    return None

# Function to merge model values for simplification
def merge_model_values(model):
    if model in ['Default (GPT-3.5)', 'Default', None]:
        return '3.5'
    elif model is not None:
        return '4'

# Function to map RepoLanguage to language buckets
def map_language_to_bucket(language):
    return next((bucket for lang, bucket in bucket_mapping.items() if lang == language), 'scripting_or_web')

# Load data from JSON
with open("./snapshot_20231012/20231012_233628_pr_sharings.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data['Sources'])
df['Model'] = df['ChatgptSharing'].apply(extract_model)
df['Model'] = df['Model'].apply(merge_model_values)

# Define language buckets and invert the dictionary for easy lookup
language_buckets = {
    'systems': ['C', 'C++', 'Rust', 'Go'],
    'scripting_or_web': ['Python', 'JavaScript', 'Ruby', 'PHP', 'HTML', 'CSS', 'SCSS', 'TypeScript', 'Shell']
}
bucket_mapping = {lang: bucket for bucket, langs in language_buckets.items() for lang in langs}

# Apply the mapping to categorize RepoLanguage into buckets
df['RepoLanguageBucket'] = df['RepoLanguage'].apply(map_language_to_bucket)

# Prepare data for analysis
reduced_df = df.loc[:, ['RepoLanguageBucket', 'Model', 'State']]
reduced_df['State_bin'] = reduced_df['State'].apply(lambda x: 1 if x == 'MERGED' else 0)

# Perform the t-test
group_35 = reduced_df[reduced_df['Model'] == '3.5']['State_bin']
group_4 = reduced_df[reduced_df['Model'] == '4']['State_bin']
t_stat, p_value = ttest_ind(group_35, group_4, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Prepare and perform the chi-squared test
contingency_table = pd.crosstab(index=[reduced_df['RepoLanguageBucket'], reduced_df['Model']], columns=reduced_df['State_bin'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared: {chi2}, P-value: {p}, Degrees of freedom: {dof}, Expected frequencies: {expected}")

# Fit the logistic regression model
formula = 'State_bin ~ C(RepoLanguageBucket) * C(Model)'
model = smf.logit(formula, data=reduced_df)
result = model.fit()
print(result.summary())
