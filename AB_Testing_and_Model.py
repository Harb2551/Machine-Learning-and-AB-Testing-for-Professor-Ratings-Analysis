print("solution 1 -----------><------------")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, levene, norm
from scipy import stats


df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.loc[(df["male"]!=df["female"]) & (df["avg_rating"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"]>=3]
print(df_filtered.shape)
df_male_ratings = df_filtered[["avg_rating","male"]].loc[df["male"]==1]
l_male_ratings = list(df_male_ratings["avg_rating"])
df_female_ratings = df_filtered[["avg_rating","female"]].loc[df["female"]==1]
l_female_ratings = list(df_female_ratings["avg_rating"])
stat, p_value = mannwhitneyu(np.array(l_male_ratings), np.array(l_female_ratings), alternative='greater')
print("statistic: ",stat)
print(f"P-value: {p_value:.2e}")
if(p_value <= 0.005):
    print("Result is significant")
else:
    print("Result is not significant")

# Loading the data and preprocessing
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings", "pepper", "class_again_prop", "num_ratings_online", "male", "female"]
df_filtered = df.loc[(df["male"] != df["female"]) & (df["avg_rating"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"] >= 3]

# Data for males and females
df_male_ratings = df_filtered[["avg_rating", "male"]].loc[df["male"] == 1]
l_male_ratings = list(df_male_ratings["avg_rating"])
df_female_ratings = df_filtered[["avg_rating", "female"]].loc[df["female"] == 1]
l_female_ratings = list(df_female_ratings["avg_rating"])

# Mann-Whitney U Test
stat, p_value = mannwhitneyu(np.array(l_male_ratings), np.array(l_female_ratings), alternative='greater')
print("statistic: ", stat)
print(f"P-value: {p_value:.2e}")
if p_value <= 0.005:
    print("Result is significant")
else:
    print("Result is not significant")

# Plotting Boxplot and Violin Plot
plt.figure(figsize=(12, 6))

# Boxplot to compare male and female ratings
plt.subplot(1, 2, 1)
sns.boxplot(data=[l_male_ratings, l_female_ratings], palette="Set2")
plt.xticks([0, 1], ['Male', 'Female'])
plt.title("Boxplot of Average Ratings (Male vs Female)")

# Violin plot to show the distribution of ratings
plt.subplot(1, 2, 2)
sns.violinplot(data=[l_male_ratings, l_female_ratings], palette="Set2")
plt.xticks([0, 1], ['Male', 'Female'])
plt.title("Violin Plot of Average Ratings (Male vs Female)")

# Show plots
plt.tight_layout()
plt.show()

print("solution 2 -----------><------------")
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.loc[(df["male"]!=df["female"]) & (df["avg_rating"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"]>=3]
print(df_filtered.shape)
df_male_ratings = df_filtered[["avg_rating","male"]].loc[df["male"]==1]
l_male_ratings = list(df_male_ratings["avg_rating"])
df_female_ratings = df_filtered[["avg_rating","female"]].loc[df["female"]==1]
l_female_ratings = list(df_female_ratings["avg_rating"])
l_male_ratings = np.array(l_male_ratings)
l_female_ratings = np.array(l_female_ratings)

# Perform the KS test
stat, p_value = ks_2samp(l_male_ratings, l_female_ratings)

# Results
print("KS Statistic:", stat)
print("P-value:", p_value)

# Interpretation
if(p_value <= 0.005):
    print("Result is significant")
else:
    print("Result is not significant")

# Perform Levene's test
stat, p_value = levene(l_male_ratings, l_female_ratings, center='median')

# Results
print("Test Statistic:", stat)
print("P-value:", p_value)

# Interpretation
if(p_value <= 0.005):
    print("Result is significant")
else:
    print("Result is not significant")

# Loading and preprocessing data
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings", "pepper", "class_again_prop", "num_ratings_online", "male", "female"]
df_filtered = df.loc[(df["male"] != df["female"]) & (df["avg_rating"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"] >= 3]

# Data for male and female ratings
df_male_ratings = df_filtered[["avg_rating", "male"]].loc[df["male"] == 1]
l_male_ratings = list(df_male_ratings["avg_rating"])
df_female_ratings = df_filtered[["avg_rating", "female"]].loc[df["female"] == 1]
l_female_ratings = list(df_female_ratings["avg_rating"])

# Convert lists to arrays
l_male_ratings = np.array(l_male_ratings)
l_female_ratings = np.array(l_female_ratings)

# Perform the Kolmogorov-Smirnov (KS) test
ks_stat, ks_p_value = ks_2samp(l_male_ratings, l_female_ratings)

# Results of KS test
print("KS Statistic:", ks_stat)
print("KS P-value:", ks_p_value)

# Perform Levene's test for equality of variances
levene_stat, levene_p_value = levene(l_male_ratings, l_female_ratings, center='median')

# Results of Levene's test
print("Levene's Test Statistic:", levene_stat)
print("Levene's Test P-value:", levene_p_value)

# Interpretation of KS and Levene's tests
if ks_p_value <= 0.005:
    print("KS Test Result is significant")
else:
    print("KS Test Result is not significant")

if levene_p_value <= 0.005:
    print("Levene's Test Result is significant")
else:
    print("Levene's Test Result is not significant")

# Plotting: Histogram + KDE and Boxplot
plt.figure(figsize=(14, 6))

# Histogram + KDE Plot (for distribution)
plt.subplot(1, 2, 1)
sns.histplot(l_male_ratings, kde=True, color='blue', label='Male Ratings', stat='density', bins=20)
sns.histplot(l_female_ratings, kde=True, color='red', label='Female Ratings', stat='density', bins=20)
plt.legend()
plt.title("Distribution of Ratings (Male vs Female)")
plt.xlabel("Average Rating")
plt.ylabel("Density")

# Boxplot (for spread and median)
plt.subplot(1, 2, 2)
sns.boxplot(data=[l_male_ratings, l_female_ratings], palette="Set2")
plt.xticks([0, 1], ['Male', 'Female'])
plt.title("Boxplot of Average Ratings (Male vs Female)")

# Show the plots
plt.tight_layout()
plt.show()

print("solution 3 -----------><------------")
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.loc[(df["male"]!=df["female"]) & (df["avg_rating"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"]>=3]
print(df_filtered.shape)
df_male_ratings = df_filtered[["avg_rating","male"]].loc[df["male"]==1]
l_male_ratings = list(df_male_ratings["avg_rating"])
df_female_ratings = df_filtered[["avg_rating","female"]].loc[df["female"]==1]
l_female_ratings = list(df_female_ratings["avg_rating"])

def calculate_cohens_d_with_ci(data1, data2, confidence=0.95):
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) /
                        (n1 + n2 - 2))

    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    # Calculate standard error for Cohen's d
    se_d = np.sqrt((std1**2 / n1) + (std2**2 / n2))

    # Get z-value for confidence level
    z_value = norm.ppf(1 - (1 - confidence) / 2)

    # Calculate confidence interval
    margin_error = z_value * se_d
    ci_lower = cohens_d - margin_error
    ci_upper = cohens_d + margin_error

    return cohens_d, ci_lower, ci_upper

d, lower, upper = calculate_cohens_d_with_ci(l_male_ratings, l_female_ratings)
print(f"Cohen's d: {d:.3f}")
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")

def effect_size_spread_with_ci(data1, data2, confidence=0.95):
    # Sample sizes
    n1, n2 = len(data1), len(data2)
    
    # Calculate variances of both datasets
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    s1 = np.sqrt(var1)
    s2 = np.sqrt(var2)
    
    # Calculate pooled variance
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) /
                        (n1 + n2 - 2))
    
    # Calculate the effect size (spread difference) based on variance
    effect_size = (var1 - var2) / pooled_sd
    
    # Calculate standard error for effect size
    se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    
    # Calculate the z-value for the confidence interval
    z_value = norm.ppf(1 - (1 - confidence) / 2)
    
    # Calculate the margin of error
    margin_error = z_value * se
    
    # Calculate confidence interval
    ci_lower = effect_size - margin_error
    ci_upper = effect_size + margin_error
    
    return effect_size, ci_lower, ci_upper

# Example usage
effect_size_male_female, ci_lower, ci_upper = effect_size_spread_with_ci(l_female_ratings, l_male_ratings)
print(f"Effect size (spread difference): {effect_size_male_female:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Load data
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))

# Filter data
df_filtered = df.loc[(df["male"] != df["female"]) & (df["avg_rating"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"] >= 3]
print(df_filtered.shape)

# Extract ratings for males and females
df_male_ratings = df_filtered[["avg_rating", "male"]].loc[df["male"] == 1]
l_male_ratings = list(df_male_ratings["avg_rating"])
df_female_ratings = df_filtered[["avg_rating", "female"]].loc[df["female"] == 1]
l_female_ratings = list(df_female_ratings["avg_rating"])

# Cohen's d Calculation with Confidence Interval
def calculate_cohens_d_with_ci(data1, data2, confidence=0.95):
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) /
                        (n1 + n2 - 2))

    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    # Calculate standard error for Cohen's d
    se_d = np.sqrt((std1**2 / n1) + (std2**2 / n2))

    # Get z-value for confidence level
    z_value = norm.ppf(1 - (1 - confidence) / 2)

    # Calculate confidence interval
    margin_error = z_value * se_d
    ci_lower = cohens_d - margin_error
    ci_upper = cohens_d + margin_error

    return cohens_d, ci_lower, ci_upper

# Cohen's d and Confidence Interval
d, lower, upper = calculate_cohens_d_with_ci(l_male_ratings, l_female_ratings)
print(f"Cohen's d: {d:.3f}")
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")

# Effect size spread with CI
def effect_size_spread_with_ci(data1, data2, confidence=0.95):
    # Sample sizes
    n1, n2 = len(data1), len(data2)
    
    # Calculate variances of both datasets
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    s1 = np.sqrt(var1)
    s2 = np.sqrt(var2)
    
    # Calculate pooled variance
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) /
                        (n1 + n2 - 2))
    
    # Calculate the effect size (spread difference) based on variance
    effect_size = (var1 - var2) / pooled_sd
    
    # Calculate standard error for effect size
    se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    
    # Calculate the z-value for the confidence interval
    z_value = norm.ppf(1 - (1 - confidence) / 2)
    
    # Calculate the margin of error
    margin_error = z_value * se
    
    # Calculate confidence interval
    ci_lower = effect_size - margin_error
    ci_upper = effect_size + margin_error
    
    return effect_size, ci_lower, ci_upper

# Calculate Effect Size and CI
effect_size_male_female, ci_lower, ci_upper = effect_size_spread_with_ci(l_female_ratings, l_male_ratings)
print(f"Effect size (spread difference): {effect_size_male_female:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Plotting
plt.figure(figsize=(14, 6))

# Plot: Histogram + KDE (for rating distributions)
ax1 = plt.subplot(1, 2, 1)
sns.histplot(l_male_ratings, kde=True, color='blue', label='Male Ratings', stat='density', bins=20, ax=ax1)
sns.histplot(l_female_ratings, kde=True, color='red', label='Female Ratings', stat='density', bins=20, ax=ax1)
ax1.legend()
ax1.set_title("Distribution of Ratings (Male vs Female)")
ax1.set_xlabel("Average Rating")
ax1.set_ylabel("Density")

# Add Cohen's d and Confidence Interval annotation inside the plot
# Position the annotation at a location based on the data range
x_pos = 0.4  # Place horizontally near the right side
y_pos = 0.4  # Place vertically near the middle of the density
ax1.annotate(f"Cohen's d: {d:.3f}\n95% CI: [{lower:.3f}, {upper:.3f}]", 
             xy=(x_pos, y_pos), xycoords='axes fraction',
             fontsize=12, color='black', ha='center', va='center')

# Plot: Boxplot (for spread and medians)
ax2 = plt.subplot(1, 2, 2)
sns.boxplot(data=[l_male_ratings, l_female_ratings], palette="Set2", ax=ax2)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Male', 'Female'])
ax2.set_title("Boxplot of Average Ratings (Male vs Female)")

# Show the plots
plt.tight_layout()
plt.show()

print("solution 4 -----------><------------")

df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_male_ratings = df["male"]
df_female_ratings = df["female"]
result = pd.concat([df_male_ratings, df_female_ratings], axis=1)
df_tags = pd.read_csv('rmpCapstoneTags.csv')
df_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]
result = pd.concat([result, df_tags], axis=1)
df_filtered = result.loc[(result["male"]!=result["female"])]
df_male = df_filtered.loc[df_filtered["male"]==1]
df_female = df_filtered.loc[df_filtered["female"]==1]
col = df_male.shape[1]
count = 0
p_values1 = {}
p_values2 = {}
for i in range(2,col):
  male_tags = df_male.iloc[:,i]
  female_tags = df_female.iloc[:,i]
  t_stat, p_value = stats.ttest_ind(male_tags, female_tags, equal_var=False)
  col_name = df_male.columns[i]
  if(p_value <= 0.005):
    count += 1
    p_values1[col_name] = p_value
  else:
    p_values2[col_name] = p_value
sorted_p_values1 = dict(sorted(p_values1.items(), key=lambda item: item[1]))
sorted_p_values2 = dict(sorted(p_values2.items(), key=lambda item: item[1]))
sorted_tags1 = list(sorted_p_values1.keys())
sorted_tags2 = list(sorted_p_values2.keys())
print("Most gendered Tags are: ")
for i in range(3):
  print(sorted_tags1[i])
print()
print()
print("Least gendered Tags are: ")
for i in range(3):
  print(sorted_tags2[len(sorted_tags2) - i - 1])
print("total significant tags: " + str(count))

# Load the data
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_male_ratings = df["male"]
df_female_ratings = df["female"]
result = pd.concat([df_male_ratings, df_female_ratings], axis=1)

df_tags = pd.read_csv('rmpCapstoneTags.csv')
df_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]
result = pd.concat([result, df_tags], axis=1)

# Filter the data
df_filtered = result.loc[(result["male"] != result["female"])]
df_male = df_filtered.loc[df_filtered["male"] == 1]
df_female = df_filtered.loc[df_filtered["female"] == 1]

# Initialize variables
col = df_male.shape[1]
count = 0
p_values1 = {}  # for significant tags (p <= 0.005)
p_values2 = {}  # for non-significant tags (p > 0.005)

# T-test calculation
for i in range(2, col):
    male_tags = df_male.iloc[:, i]
    female_tags = df_female.iloc[:, i]
    t_stat, p_value = stats.ttest_ind(male_tags, female_tags, equal_var=False)
    col_name = df_male.columns[i]
    
    if p_value <= 0.005:
        count += 1
        p_values1[col_name] = p_value
    else:
        p_values2[col_name] = p_value

# Sort tags by p-value
sorted_p_values1 = dict(sorted(p_values1.items(), key=lambda item: item[1]))  # most gendered
sorted_p_values2 = dict(sorted(p_values2.items(), key=lambda item: item[1]))  # least gendered

# Get the sorted tags
sorted_tags1 = list(sorted_p_values1.keys())  # most gendered tags
sorted_tags2 = list(sorted_p_values2.keys())  # least gendered tags

# Print the findings
print("Most gendered Tags are: ")
for i in range(3):
    print(sorted_tags1[i])

print("\nLeast gendered Tags are: ")
for i in range(3):
    print(sorted_tags2[len(sorted_tags2) - i - 1])

print("Total significant tags: " + str(count))

# Create a bar plot to illustrate the findings
plt.figure(figsize=(14, 6))

# Plot for the Most Gendered Tags
most_gendered = sorted_tags1  # all most gendered tags
p_values_most = [sorted_p_values1[tag] for tag in most_gendered]
plt.bar(most_gendered, p_values_most, color='mediumorchid', label='Significant Gendered Tags')

# Plot for the Least Gendered Tags
least_gendered = sorted_tags2  # all least gendered tags
p_values_least = [sorted_p_values2[tag] for tag in least_gendered]
plt.bar(least_gendered, p_values_least, color='lightseagreen', label='Not Significant Gendered Tags')

# Annotating p-values on the bars
for i, p_val in enumerate(p_values_most):
    plt.text(i, p_val + 0.0005, f'{p_val:.4f}', ha='center', va='bottom', fontsize=10, color='black')

for i, p_val in enumerate(p_values_least):
    plt.text(i + len(most_gendered), p_val + 0.0005, f'{p_val:.4f}', ha='center', va='bottom', fontsize=10, color='black')

# Title and labels
plt.title("Most and Least Gendered Tags Based on P-values (T-test)", fontsize=14)
plt.xlabel("Tags", fontsize=12)
plt.ylabel("P-value", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add a legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

print("solution 5 -----------><------------")
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.loc[(df["male"]!=df["female"]) & (df["avg_diff"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"]>=3]
print(df_filtered.shape)
df_male_ratings = df_filtered[["avg_diff","male"]].loc[df["male"]==1]
l_male_ratings = list(df_male_ratings["avg_diff"])
df_female_ratings = df_filtered[["avg_diff","female"]].loc[df["female"]==1]
l_female_ratings = list(df_female_ratings["avg_diff"])
stat, p_value = mannwhitneyu(np.array(l_male_ratings), np.array(l_female_ratings), alternative='two-sided')
print("statistic: ",stat)
print(f"P-value: {p_value:.100f}")
if(p_value <= 0.005):
    print("Result is significant")
else:
    print("Result is not significant")

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(np.array(l_male_ratings), np.array(l_female_ratings), alternative='two-sided')

# Set up the plot
plt.figure(figsize=(10, 6))

# Create the box plot
sns.boxplot(data=[l_male_ratings, l_female_ratings])

# Customize the plot
plt.title("Gender Differences in Instructor Difficulty Ratings", fontsize=16)
plt.ylabel("Average Difficulty Rating", fontsize=12)
plt.xticks([0, 1], ['Male Instructors', 'Female Instructors'], fontsize=12)
plt.ylim(0, 5)  # Assuming ratings are on a 0-5 scale

# Add statistical annotation
significance = "Result is significant" if p_value <= 0.005 else "Result is not significant"
stats_text = f"Mann-Whitney U statistic: {stat:.2f}\nP-value: {p_value:.2e}\n{significance}"
plt.text(0.5, -0.15, stats_text, 
         horizontalalignment='center', 
         verticalalignment='center', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         fontsize=10)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

print("solution 6 -----------><------------")
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.loc[(df["male"]!=df["female"]) & (df["avg_diff"].notna())]
df_filtered = df_filtered.loc[df_filtered["num_ratings"]>=3]
print(df_filtered.shape)
df_male_ratings = df_filtered[["avg_diff","male"]].loc[df["male"]==1]
l_male_ratings = list(df_male_ratings["avg_diff"])
df_female_ratings = df_filtered[["avg_diff","female"]].loc[df["female"]==1]
l_female_ratings = list(df_female_ratings["avg_diff"])

def calculate_cohens_d_with_ci(data1, data2, confidence=0.95):
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) /
                        (n1 + n2 - 2))

    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    # Calculate standard error for Cohen's d
    se = np.sqrt((std1**2 / n1) + (std2**2 / n2))

    # Get z-value for confidence level
    z_value = norm.ppf(1 - (1 - confidence) / 2)

    # Calculate confidence interval
    margin_error = z_value * se
    ci_lower = cohens_d - margin_error
    ci_upper = cohens_d + margin_error

    return cohens_d, ci_lower, ci_upper

d, lower, upper = calculate_cohens_d_with_ci(l_male_ratings, l_female_ratings)
print(f"Cohen's d: {d:.3f}")
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")

# Calculate Cohen's d and confidence interval
d, lower, upper = calculate_cohens_d_with_ci(l_male_ratings, l_female_ratings)

# Set up the plot
plt.figure(figsize=(10, 6))

# Create the violin plot
sns.violinplot(data=[l_male_ratings, l_female_ratings], 
               inner="box", 
               cut=0)

# Customize the plot
plt.title("Gender Differences in Instructor Difficulty Ratings", fontsize=16)
plt.ylabel("Average Difficulty Rating", fontsize=12)
plt.xticks([0, 1], ['Male Instructors', 'Female Instructors'], fontsize=12)
plt.ylim(0, 5)  # Assuming ratings are on a 0-5 scale

# Add effect size annotation
effect_size_text = f"Cohen's d: {d:.3f}\n95% CI: [{lower:.3f}, {upper:.3f}]"
plt.text(0.5, -0.15, effect_size_text, 
         horizontalalignment='center', 
         verticalalignment='center', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         fontsize=10)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

print("solution 7 -----------><------------")

import numpy as np
import random
#The N-Number of Rishabh Patil
N_Number=16150234
random_state= random.seed(N_Number)
random_state = random.randint(0,100)
print(f"Random number generator seeded with N-number: {random_state}")

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from math import sqrt

df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.drop(columns=["female"])
df_filtered.dropna(inplace=True)
df_filtered_scaled = df_filtered.drop(columns=['male','pepper'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered_scaled)
df_scaled = pd.DataFrame(scaled_data)
df_scaled.columns = ["avg_rating", "avg_diff", "num_ratings","class_again_prop","num_ratings_online"]
df_scaled.reset_index(drop=True, inplace=True)
df_filtered.reset_index(drop=True, inplace=True)
df_scaled['pepper'] = df_filtered['pepper']
df_scaled['male'] = df_filtered['male']
X_train, X_test, y_train, y_test = train_test_split(np.array(df_scaled.iloc[:,1:]), np.array(df_filtered.iloc[:,0]), test_size=0.2, random_state=N_Number)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(y_pred)
# Model evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Root Mean Squared Error:",sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared:", r2_score(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
l = list((df.columns))
df_filtered = df.drop(columns=["female"])
df_filtered = df_filtered[df_filtered["num_ratings"] >= 10]
df_filtered.dropna(inplace=True)
correlation_matrix = df_filtered.iloc[:,1:].corr()

# Plot a heatmap
plt.figure(figsize=(10, 8))  # Adjust the size of the figure
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title("Correlation Matrix")
plt.show()

# Identify the most predictive variable
coefficients = pd.DataFrame({
    'Feature': df_scaled.iloc[:,1:].columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)


top_3_coefficients = coefficients.head(3)

print("\nTop 3 Most Predictive Factors (Linear Regression Coefficients):")
print(top_3_coefficients)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()

print("solution 8 -----------><------------")

import numpy as np
import random
#The N-Number of Rishabh Patil
N_Number=16150234
random_state= random.seed(N_Number)
random_state = random.randint(0,100)
print(f"Random number generator seeded with N-number: {random_state}")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_num = pd.read_csv('rmpCapstoneNum.csv')
df_tags = pd.read_csv('rmpCapstoneTags.csv')
df_num.columns = ["avg_rating", "avg_diff", "num_rating","pepper","class_again_prop","num_ratings_online","male","female"]
df_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]
df_tags["avg_rating"] = df_num["avg_rating"]
df_tags["num_rating"] = df_num["num_rating"]
df_tags = df_tags.loc[df_tags["avg_rating"].notna()]
df_tags= df_tags[df_tags["num_rating"] >= 3]
print(df_tags.shape)

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df_tags.iloc[:,:-2].corr()
# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title("Correlation Matrix")
plt.show()

X = df_tags.drop(columns=["avg_rating","num_rating"])

tags = X
# Calculate the correlation matrix
correlation_matrix = tags.corr()

# Set a correlation threshold (e.g., 0.7 or -0.7)
threshold = 0.7

# Create an upper triangle matrix (we don’t need to check for correlation twice)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find columns with correlation greater than the threshold
to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]

#Collinear Columns
to_drop
df_tags=df_tags.drop(columns=to_drop)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt
from scipy.stats import boxcox, zscore

# Apply Box-Cox or log transformations to all columns
def apply_transformation(column):
    if (df_tags[column] > 0).all():  # Check if Box-Cox is applicable
        df_tags[column], _ = boxcox(df_tags[column] + 1e-5)  # Add small constant to avoid log(0)
    else:
        df_tags[column] = np.log1p(df_tags[column])  # Apply log1p for non-positive values


for col in df_tags.columns:
    if df_tags[col].dtype != 'object' and col != "avg_rating":
        apply_transformation(col)

# Target variable
y = df_tags["avg_rating"]

# Feature selection (or use all columns except the target)
X = df_tags.drop(columns=["avg_rating","num_rating"])

# Add polynomial features for better representation
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)

# Train-test split using polynomial features
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=N_Number)

# Train the linear regression model on polynomial features
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred= linear.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Model:")
    print(f"  Root Mean Squared Error: {rmse}")
    print(f"  R-squared: {r2}")
    print("-" * 40)
# Evaluate the model
evaluate_model("Linear Regression", y_test, y_pred)

# Identify the most significant features
coefficients = linear.coef_
abs_coefficients = np.abs(coefficients)

# Sort features by importance (absolute coefficient value)
important_indices = np.argsort(-abs_coefficients)
important_features = [(feature_names[i], coefficients[i]) for i in important_indices]

# Display the top features
top_n = 3
print(f"Top {top_n} important features:")
for i, (feature, coeff) in enumerate(important_features[:top_n], 1):
    print(f"{i}. {feature}: Coefficient = {coeff:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = X.iloc[:,:-1].corr()

# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title("Correlation Matrix")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()

print("solution 9 -----------><------------")

import numpy as np
import random
#The N-Number of Rishabh Patil
N_Number=16150234
random_state= random.seed(N_Number)
random_state = random.randint(0,100)
print(f"Random number generator seeded with N-number: {random_state}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_num = pd.read_csv('rmpCapstoneNum.csv')
df_tags = pd.read_csv('rmpCapstoneTags.csv')
df_num.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
df_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]
df_tags["avg_diff"] = df_num["avg_diff"]
df_tags["num_ratings"] = df_num["num_ratings"]
df_tags = df_tags.loc[df_tags["avg_diff"].notna()]
df_tags= df_tags[df_tags["num_ratings"] >= 3]
print(df_tags.shape)

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df_tags.iloc[:,:-2].corr()
# Plot a heatmap
plt.figure(figsize=(10, 8))  # Adjust the size of the figure
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title("Correlation Matrix")
plt.show()

X = df_tags.drop(columns=["avg_diff","num_ratings"])

tags = X
# Calculate the correlation matrix
correlation_matrix = tags.corr()

# Set a correlation threshold (e.g., 0.7 or -0.7)
threshold = 0.7

# Create an upper triangle matrix (we don’t need to check for correlation twice)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find columns with correlation greater than the threshold
to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]

#Collinear Columns
to_drop

df_tags=df_tags.drop(columns=to_drop)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt
from scipy.stats import boxcox, zscore

# Apply Box-Cox or log transformations to all columns
def apply_transformation(column):
    if (df_tags[column] > 0).all():  # Check if Box-Cox is applicable
        df_tags[column], _ = boxcox(df_tags[column] + 1e-5)  # Add small constant to avoid log(0)
    else:
        df_tags[column] = np.log1p(df_tags[column])  # Apply log1p for non-positive values

for col in df_tags.columns:
    if df_tags[col].dtype != 'object' and col != "avg_diff":
        apply_transformation(col)

# Target variable
y = df_tags["avg_diff"]

# Feature selection (or use all columns except the target)
X = df_tags.drop(columns=["avg_diff",'num_ratings'])

# Add polynomial features for better representation
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=N_Number)

linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Model:")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")
    print("-" * 40)

evaluate_model("Linear Regression", y_test, y_pred)

# Identify the most significant features
coefficients = linear.coef_
abs_coefficients = np.abs(coefficients)

# Sort features by importance (absolute coefficient value)
important_indices = np.argsort(-abs_coefficients)  # Sort in descending order
important_features = [(feature_names[i], coefficients[i]) for i in important_indices]

# Display the top features
top_n = 3
print(f"Top {top_n} important features:")
for i, (feature, coeff) in enumerate(important_features[:top_n], 1):
    print(f"{i}. {feature}: Coefficient = {coeff:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = X.iloc[:,:-1].corr()

# Plot a heatmap
plt.figure(figsize=(10, 8))  # Adjust the size of the figure
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title("Correlation Matrix")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Difficulty")
plt.ylabel("Predicted Difficulty")
plt.title("Actual vs Predicted Difficulty")
plt.show()

print("solution 10 -----------><------------")

import numpy as np
import random
#The N-Number of Rishabh Patil
N_Number=16150234
random_state= random.seed(N_Number)
random_state = random.randint(0,100)
print(f"Random number generator seeded with N-number: {random_state}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_num = pd.read_csv('rmpCapstoneNum.csv')
df_tags = pd.read_csv('rmpCapstoneTags.csv')
df_num.columns = ["avg_rating", "avg_diff", "num_ratings","pepper","class_again_prop","num_ratings_online","male","female"]
df_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]

df_com = pd.concat([df_num, df_tags], axis=1)
df_com = df_com.drop(columns = ['female','class_again_prop'])
df_com= df_com[df_com["num_ratings"] >= 3]
df_com = df_com.dropna()
df_com = df_com.reset_index(drop=True)
y = df_com['pepper']
male = df_com['male']
df=df_com
df_com = df_com.drop(columns = ["male","pepper"])
print(df_com.shape)

import seaborn as sns
sns.countplot(data=df, x='pepper')

imbalance_ratio = df['pepper'].value_counts().max() / df['pepper'].value_counts().min()
print(f"Imbalance Ratio: {imbalance_ratio}")

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_com)
scaled_df = pd.DataFrame(scaled_data, columns=df_com.columns)
combined_columns = df_com.columns.tolist()
scaled_df.columns = combined_columns
scaled_df['male'] = male
scaled_df = scaled_df.reset_index(drop=True)
X = scaled_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=N_Number)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Predict probabilities instead of labels for ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Compute AUC-ROC
auc_score = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc_score:.2f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print("solution 11 extra credit -----------><------------")

# Load and preprocess data
df_qual = pd.read_csv('rmpCapstoneQual.csv')
df_num = pd.read_csv('rmpCapstoneNum.csv')
df_qual.columns = ["subject", "university", "state"]
df_num.columns = ["avg_rating", "avg_diff", "num_ratings", "pepper", "class_again_prop", "num_ratings_online", "male", "female"]

# Combine relevant columns into a new DataFrame
df_sub_ratings = pd.DataFrame({
    "subject": df_qual["subject"],
    "avg_rating": df_num["avg_rating"]
})
df_sub_ratings = df_sub_ratings.dropna()

# Filter subjects with more than 1000 occurrences
subjects = list(set(df_sub_ratings["subject"]))
filtered_subjects = [subject for subject in subjects if (df_sub_ratings["subject"] == subject).sum() > 1000]

# Collect ratings by subject
subject_ratings = []
for subject in filtered_subjects:
    subject_ratings.append(list(df_sub_ratings.loc[df_sub_ratings["subject"] == subject]["avg_rating"]))

# Perform Kruskal-Wallis test
stat, p_value = stats.kruskal(*subject_ratings)
print("Subjects:", filtered_subjects)
print("Statistic:", stat)
print("p-value:", p_value)

# Compute ranks and mean ranks for each subject
all_ratings = []
group_labels = []

for subject in filtered_subjects:
    ratings = df_sub_ratings.loc[df_sub_ratings["subject"] == subject]["avg_rating"]
    all_ratings.extend(ratings)
    group_labels.extend([subject] * len(ratings))

ranks = stats.rankdata(all_ratings)
df_ranks = pd.DataFrame({"subject": group_labels, "rating": all_ratings, "rank": ranks})

# Compute mean ranks for each subject
mean_ranks = df_ranks.groupby("subject")["rank"].mean()

# Sort mean ranks in descending order
sorted_mean_ranks = mean_ranks.sort_values(ascending=False)

# Get the top 3 subjects with the highest mean ranks
top_3_subjects = sorted_mean_ranks.head(3)

print("\nTop 3 Subjects with the Highest Effect:")
print(top_3_subjects)

# Optional: Individual breakdown
print("\nAll Mean Ranks:")
print(sorted_mean_ranks)

# Load and preprocess data
df_qual = pd.read_csv('rmpCapstoneQual.csv')
df_num = pd.read_csv('rmpCapstoneNum.csv')
df_qual.columns = ["subject", "university", "state"]
df_num.columns = ["avg_rating", "avg_diff", "num_ratings", "pepper", "class_again_prop", "num_ratings_online", "male", "female"]

# Combine relevant columns into a new DataFrame
df_sub_ratings = pd.DataFrame({
    "subject": df_qual["subject"],
    "avg_rating": df_num["avg_rating"]
})
df_sub_ratings = df_sub_ratings.dropna()

# Filter subjects with more than 1000 occurrences
subjects = list(set(df_sub_ratings["subject"]))
filtered_subjects = [subject for subject in subjects if (df_sub_ratings["subject"] == subject).sum() > 1000]

# Collect ratings by subject
subject_ratings = []
for subject in filtered_subjects:
    subject_ratings.append(list(df_sub_ratings.loc[df_sub_ratings["subject"] == subject]["avg_rating"]))

# Perform Kruskal-Wallis test
stat, p_value = stats.kruskal(*subject_ratings)
print("Subjects:", filtered_subjects)
print("Statistic:", stat)
print("p-value:", p_value)

# Compute ranks and mean ranks for each subject
all_ratings = []
group_labels = []

for subject in filtered_subjects:
    ratings = df_sub_ratings.loc[df_sub_ratings["subject"] == subject]["avg_rating"]
    all_ratings.extend(ratings)
    group_labels.extend([subject] * len(ratings))

ranks = stats.rankdata(all_ratings)
df_ranks = pd.DataFrame({"subject": group_labels, "rating": all_ratings, "rank": ranks})

# Compute mean ranks for each subject
mean_ranks = df_ranks.groupby("subject")["rank"].mean()

# Sort mean ranks in descending order
sorted_mean_ranks = mean_ranks.sort_values(ascending=False)

# Get the top 3 subjects with the highest mean ranks
top_3_subjects = sorted_mean_ranks.head(3)

print("\nTop 3 Subjects with the Highest Effect:")
print(top_3_subjects)

# Plot the mean ranks for each subject
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_mean_ranks.index, y=sorted_mean_ranks.values, palette="viridis")

# Add title and labels
plt.title("Mean Ranks of Subjects (Kruskal-Wallis Test)", fontsize=16)
plt.xlabel("Subject", fontsize=12)
plt.ylabel("Mean Rank", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)

# Add a box with the top 3 subjects' details
top_3_text = '\n'.join([f"{subject}: {rank:.2f}" for subject, rank in top_3_subjects.items()])
plt.gca().text(16.05, 0.5, top_3_text, fontsize=12, ha="left", va="center", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

plt.tight_layout()
plt.show()
