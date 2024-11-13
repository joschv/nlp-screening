import pandas as pd

df = pd.read_csv('./out/out.csv')
df = pd.DataFrame(df)

rows = len(df)
print('Total rows found:', rows)
print(df.columns)

# print((df == True).sum())

df_both = df[(df['is_about_epidemiology_or_virology'] == True) & (df['uses_deep_learning'] == True)]
df_neither = df[(df['is_about_epidemiology_or_virology'] == False) & (df['uses_deep_learning'] == False)]

passing_studies = {
    'on-topic': (df['is_about_epidemiology_or_virology'] == True).sum(),
    'uses deep learning': (df['uses_deep_learning'] == True).sum(),
    'on topic and uses deep learning': len(df_both),
    'neither': len(df_neither)
}
print(passing_studies)

method_counts = {
    'computer vision': (df_both['deep_learning_method'] == 'computer vision').sum(),
    'text mining': (df_both['deep_learning_method'] == 'text mining').sum(),
    'both': (df_both['deep_learning_method'] == 'both').sum(),
    'other': (df_both['deep_learning_method'] == 'other').sum()
}
print(method_counts)