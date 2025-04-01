import pandas as pd

# Предполагается, что в локальном окружении уже есть переменная df,
# содержащая исходный DataFrame.

# 1) Преобразуем столбец 'Date' к datetime (формат 'M/D/YYYY')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# 2) Группируем одновременно по дате и категории (Product line)
df = df.groupby(['Date', 'Product line'])['Quantity'].sum().reset_index()