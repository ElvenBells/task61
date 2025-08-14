import pandas as pd

print("Download...")
df = pd.read_parquet('data/transaction_fraud_data.parquet')

print("### TASK 50 \n")

total_transactions = len(df)
fraud_transactions = df['is_fraud'].sum()

fraud_ratio = fraud_transactions / total_transactions

print(f"total: {total_transactions:}")
print(f"fraud: {fraud_transactions:}")
print(f"fraud/total: {fraud_ratio:}")


print("### TASK 51 \n")

fraud_df = df[df['is_fraud'] == 1]

top_countries = fraud_df['country'].value_counts().head(5)

print(top_countries.to_string())


print("### TASK 52 \n")

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['hour_bin'] = df['timestamp'].dt.to_period('h')

transactions_per_hour_per_customer = (
    df.groupby(['customer_id', 'hour_bin'])
    .size()
    .reset_index(name='transaction_count')
)

avg_transactions = transactions_per_hour_per_customer['transaction_count'].mean()

print(f"avg_transactions: {avg_transactions}")


print("### TASK 53 \n")

df['is_high_risk_vendor'] = df['is_high_risk_vendor'].astype(bool)
df['is_fraud'] = df['is_fraud'].astype(int)

high_risk_df = df[df['is_high_risk_vendor'] == True]

if len(high_risk_df) == 0:
    print("empty")
else:
    total_high_risk = len(high_risk_df)
    fraud_high_risk = high_risk_df['is_fraud'].sum()
    fraud_ratio = fraud_high_risk / total_high_risk

    print(f"total: {total_high_risk:,}")
    print(f"fraud: {fraud_high_risk:,}")
    print(f"fraud/total: {fraud_ratio:.4f}")


print("\n### TASK 54 \n")

df_exchange = pd.read_parquet('data/historical_currency_exchange.parquet')
df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.date

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

currencies = [col for col in df_exchange.columns if col != 'date']
exchange_stacked = df_exchange.melt(id_vars='date', value_vars=currencies,
                                    var_name='currency', value_name='exchange_rate_to_usd')

usd_row = pd.DataFrame({'date': df_exchange['date'].unique(),
                        'currency': 'USD',
                        'exchange_rate_to_usd': 1.0})
exchange_stacked = pd.concat([exchange_stacked, usd_row], ignore_index=True)

df = df.merge(exchange_stacked, on=['date', 'currency'], how='left')

if df['exchange_rate_to_usd'].isnull().any():
    df['exchange_rate_to_usd'].fillna(df['exchange_rate_to_usd'].median(), inplace=True)

df['amount_usd'] = df['amount'] * df['exchange_rate_to_usd']

df = df[df['amount_usd'] > 0]

city_avg = df.groupby('city')['amount_usd'].mean().sort_values(ascending=False)

top_city = city_avg.index[0]
top_avg = city_avg.iloc[0]

print(f"city: {top_city}")
print(f"avg: ${top_avg:}")


print("\n### TASK 55 \n")

category_col = 'vendor_category'

if category_col is None:
    raise ValueError("Не найден столбец с категорией продавца (например, merchant_category)")

print(f"Используется столбец категории: {category_col}")

df[category_col] = df[category_col].astype(str).str.lower()
df['city'] = df['city'].astype(str)

fast_food_keywords = ['fast_food', 'fast food', 'food']

mask = df[category_col].apply(lambda x: any(kw in x for kw in fast_food_keywords))
df_fast_food = df[mask]

if len(df_fast_food) == 0:
    print("Не найдено транзакций, связанных с fast_food.")
else:
    print(f"total {len(df_fast_food):,}")

    df_exchange = pd.read_parquet('data/historical_currency_exchange.parquet')
    df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.date

    df_fast_food['timestamp'] = pd.to_datetime(df_fast_food['timestamp'])
    df_fast_food['date'] = df_fast_food['timestamp'].dt.date

    currencies = [col for col in df_exchange.columns if col != 'date']
    exchange_stacked = df_exchange.melt(id_vars='date', value_vars=currencies,
                                        var_name='currency', value_name='exchange_rate_to_usd')

    usd_row = pd.DataFrame({'date': df_exchange['date'].unique(),
                            'currency': 'USD',
                            'exchange_rate_to_usd': 1.0})
    exchange_stacked = pd.concat([exchange_stacked, usd_row], ignore_index=True)

    df_fast_food = df_fast_food.merge(exchange_stacked, on=['date', 'currency'], how='left')

    if df_fast_food['exchange_rate_to_usd'].isnull().any():
        df_fast_food['exchange_rate_to_usd'].fillna(df_fast_food['exchange_rate_to_usd'].median(), inplace=True)

    df_fast_food['amount_usd'] = df_fast_food['amount'] * df_fast_food['exchange_rate_to_usd']

    df_fast_food = df_fast_food[df_fast_food['amount_usd'] > 0]

    city_avg = df_fast_food.groupby('city')['amount_usd'].mean().sort_values(ascending=False)

    if len(city_avg) == 0:
        print("empty")
    else:
        top_city = city_avg.index[0]
        top_avg = city_avg.iloc[0]

        print(f"\nГород с самым высоким средним чеком в fast_food: **{top_city}**")
        print(f"Средний чек: **${top_avg:.2f}**")


# print("\n### TASK 56 \n")

# df_exchange = pd.read_parquet('data/historical_currency_exchange.parquet')

# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df['date'] = df['timestamp'].dt.date

# df_legit = df[df['is_fraud'] == 0]

# if len(df_legit) == 0:
#     print("empty")
#     exit()

# print(f"Найдено {len(df_legit):,} легитимных транзакций.")

# df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.date
# currencies = [col for col in df_exchange.columns if col != 'date']

# exchange_stacked = df_exchange.melt(
#     id_vars='date',
#     value_vars=currencies,
#     var_name='currency',
#     value_name='exchange_rate_to_usd'
# )

# usd_row = pd.DataFrame({
#     'date': df_exchange['date'].unique(),
#     'currency': 'USD',
#     'exchange_rate_to_usd': 1.0
# })
# exchange_stacked = pd.concat([exchange_stacked, usd_row], ignore_index=True)

# df_legit = df_legit.merge(exchange_stacked, on=['date', 'currency'], how='left')

# if df_legit['exchange_rate_to_usd'].isnull().any():
#     median_rate = df_legit['exchange_rate_to_usd'].median()
#     df_legit['exchange_rate_to_usd'].fillna(median_rate, inplace=True)

# df_legit['amount_usd'] = df_legit['amount'] * df_legit['exchange_rate_to_usd']

# df_legit = df_legit[df_legit['amount_usd'] > 0]

# avg_amount_usd = df_legit['amount_usd'].mean()

# print(f"\nСредняя сумма немошеннической транзакции (в USD): ${avg_amount_usd:,.2f}")


print("\n### TASK 57 \n")

df_exchange = pd.read_parquet('data/historical_currency_exchange.parquet')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

df_legit = df[df['is_fraud'] == 0]

if len(df_legit) == 0:
    print("Нет легитимных транзакций.")
    exit()

print(f"Количество легитимных транзакций: {len(df_legit):,}")

df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.date
currencies = [col for col in df_exchange.columns if col != 'date']

exchange_stacked = df_exchange.melt(
    id_vars='date',
    value_vars=currencies,
    var_name='currency',
    value_name='exchange_rate_to_usd'
)

usd_row = pd.DataFrame({
    'date': df_exchange['date'].unique(),
    'currency': 'USD',
    'exchange_rate_to_usd': 1.0
})
exchange_stacked = pd.concat([exchange_stacked, usd_row], ignore_index=True)

df_legit = df_legit.merge(exchange_stacked, on=['date', 'currency'], how='left')

if df_legit['exchange_rate_to_usd'].isnull().any():
    print("Пропущенные курсы валют. Заполняем медианой...")
    median_rate = df_legit['exchange_rate_to_usd'].median()
    df_legit['exchange_rate_to_usd'].fillna(median_rate, inplace=True)

df_legit['amount_usd'] = df_legit['amount'] * df_legit['exchange_rate_to_usd']

df_legit = df_legit[df_legit['amount_usd'] > 0]

std_amount_usd = df_legit['amount_usd'].std()
mean_amount_usd = df_legit['amount_usd'].mean()

print(f"Среднее:     ${mean_amount_usd:,.2f}")
print(f"Стандартное отклонение: ${std_amount_usd:,.2f}")