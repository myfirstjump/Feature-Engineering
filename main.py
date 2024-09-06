from sklearn import datasets, svm, preprocessing, pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

df_X = pd.DataFrame(X)
df_X.columns = ['Time', 'RPM', 'TEMP', 'ROT']
print("X")
print(df_X)

stats = df_X.describe()
print(stats)



# ### box-plot
# sns.boxplot(data=df_X)
# plt.show()
# # Outliers --- Q3 + 1.5*IQR --- Q3 (75%) --- Median --- Q1 (25%) --- Q1 - 1.5*IQR --- Outliers


# ### Histogram
# df_X.hist(bins=50)

# ### Time Series
# plt.plot(df_X['Time'], df_X['RPM'])

### Corr
correlation_matrix = df_X.corr()
sns.heatmap(correlation_matrix, annot=True)


plt.tight_layout()
plt.show()