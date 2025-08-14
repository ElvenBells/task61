 
# --- Импорт библиотек ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Создание папки для графиков
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# --- Пути к файлам ---
data_dir = 'data'
fraud_file = os.path.join(data_dir, 'transaction_fraud_data.parquet')
exchange_file = os.path.join(data_dir, 'historical_currency_exchange.parquet')

# --- 1. Загрузка данных ---
print("Загрузка данных...")
df_fraud = pd.read_parquet(fraud_file)
df_exchange = pd.read_parquet(exchange_file)

print(f"Размер fraud-данных: {df_fraud.shape}")
print(f"Размер данных обменных курсов: {df_exchange.shape}")

# --- 2. Быстрый осмотр данных ---
print("\nПервые 5 строк fraud-данных:")
print(df_fraud.head())

print("\nПервые 5 строк exchange-данных:")
print(df_exchange.head())

# --- 3. Информация о типах данных ---
print("\nИнформация о fraud-датафрейме:")
print(df_fraud.info())

# --- 4. Распаковка структуры last_hour_activity ---
print("\nРаспаковка last_hour_activity...")
df_activity = pd.json_normalize(df_fraud['last_hour_activity'])
df_activity.columns = [f"last_hour_{col}" for col in df_activity.columns]
df_fraud = pd.concat([df_fraud.drop(columns=['last_hour_activity']), df_activity], axis=1)

print("После распаковки last_hour_activity:")
print(df_fraud[['last_hour_num_transactions', 'last_hour_total_amount', 'last_hour_max_single_amount']].head())

# --- 5. Пропущенные значения ---
print("\nПропущенные значения:")
missing = df_fraud.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("Пропущенных значений нет.")
else:
    print(missing)

# --- 6. Дисбаланс классов ---
print("\nДисбаланс классов:")
fraud_counts = df_fraud['is_fraud'].value_counts()
fraud_percent = df_fraud['is_fraud'].value_counts(normalize=True) * 100

print(fraud_counts)
print(fraud_percent)

# Визуализация дисбаланса
plt.figure(figsize=(8, 6))
sns.countplot(data=df_fraud, x='is_fraud', palette='coolwarm')
plt.title('Распределение классов: Легитимные vs Мошеннические транзакции')
plt.xlabel('is_fraud')
plt.ylabel('Количество транзакций')
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300)
plt.show()

# --- 7. Анализ временных меток ---
print("\nАнализ временных меток...")
df_fraud['timestamp'] = pd.to_datetime(df_fraud['timestamp'])
df_fraud['hour'] = df_fraud['timestamp'].dt.hour
df_fraud['day_of_week'] = df_fraud['timestamp'].dt.dayofweek  # 0=понедельник
df_fraud['date'] = df_fraud['timestamp'].dt.date

# Частота мошенничества по часам
plt.figure(figsize=(12, 6))
sns.countplot(data=df_fraud, x='hour', hue='is_fraud', palette='Set2')
plt.title('Частота транзакций по часам суток')
plt.xlabel('Час')
plt.ylabel('Количество')
plt.legend(title='is_fraud', labels=['Не мошенничество', 'Мошенничество'])
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'transactions_by_hour.png'), dpi=300)
plt.show()

# Частота мошенничества по дням недели
plt.figure(figsize=(10, 6))
sns.countplot(data=df_fraud, x='day_of_week', hue='is_fraud', palette='Set1')
plt.title('Частота транзакций по дням недели')
plt.xlabel('День недели (0=Пн, 6=Вс)')
plt.ylabel('Количество')
plt.legend(title='is_fraud')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'transactions_by_weekday.png'), dpi=300)
plt.show()

# --- 8. Анализ суммы транзакций ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_fraud, x='is_fraud', y='amount', palette='viridis')
plt.title('Распределение суммы транзакций по классам')
plt.xlabel('is_fraud')
plt.ylabel('Сумма транзакции')
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'amount_distribution.png'), dpi=300)
plt.show()

# --- 9. Анализ категорий ---

# Получаем список всех уникальных значений в колонке 'vendor_category'
all_categories = df_fraud['vendor_category'].unique()

# Выводим список
print("Все категории вендоров:")
print(all_categories)

# Также можно вывести количество уникальных категорий
print(f"\nОбщее количество уникальных категорий: {df_fraud['vendor_category'].nunique()}")

plt.figure(figsize=(12, 6))
top_categories = df_fraud['vendor_category'].value_counts().head(20).index
sns.countplot(data=df_fraud[df_fraud['vendor_category'].isin(top_categories)],
              x='vendor_category', hue='is_fraud', palette='coolwarm')
plt.title('Топ-10 категорий вендоров')
plt.xlabel('Категория')
plt.ylabel('Количество транзакций')
plt.xticks(rotation=45)
plt.legend(title='is_fraud')
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_categories.png'), dpi=300)
plt.show()

# Считаем долю фрода по каждой категории
fraud_ratio = df_fraud.groupby('vendor_category')['is_fraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=fraud_ratio.index, y=fraud_ratio.values, palette='viridis')
plt.title('Доля мошеннических транзакций по всем категориям')
plt.xlabel('Категория')
plt.ylabel('Доля мошенничества')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_ratio_all_categories.png'), dpi=300)
plt.show()

# --- 10. География: страны и города ---
print("\nТоп-10 стран по количеству транзакций:")
transaction_by_country = df_fraud['country'].value_counts().head(10)
print(transaction_by_country)

# График 1: Топ-10 стран по количеству транзакций
plt.figure(figsize=(12, 6))
sns.barplot(x=transaction_by_country.index, y=transaction_by_country.values, palette='Blues_r')
plt.title('Топ-10 стран по количеству транзакций')
plt.xlabel('Страна')
plt.ylabel('Количество транзакций')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_countries_transactions.png'), dpi=300)
plt.show()

# Мошенничество по странам
fraud_by_country = df_fraud.groupby('country')['is_fraud'].mean().sort_values(ascending=False).head(10)
print("\nТоп-10 стран по доле мошенничества:")
print(fraud_by_country)

# График 2: Топ-10 стран по доле мошенничества (горизонтальный — лучше для читаемости)
plt.figure(figsize=(10, 6))
sns.barplot(x=fraud_by_country.values, y=fraud_by_country.index, palette='Reds_r', orient='h')
plt.title('Топ-10 стран по доле мошеннических транзакций')
plt.xlabel('Доля мошенничества')
plt.ylabel('Страна')
plt.xlim(0, 1)  # Доля от 0 до 1
for i, v in enumerate(fraud_by_country.values):
    plt.text(v + 0.005, i, f"{v:.3f}", color='black', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_countries_fraud_ratio.png'), dpi=300)
plt.show()

# --- 11. Анализ устройств и каналов ---
plt.figure(figsize=(10, 6))
sns.countplot(data=df_fraud, x='device', hue='is_fraud', palette='Set3')
plt.title('Мошенничество по типу устройства')
plt.xlabel('Устройство')
plt.ylabel('Количество')
plt.xticks(rotation=45)
plt.legend(title='is_fraud')
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_by_device.png'), dpi=300)
plt.show()

# --- 12. Взаимосвязь признаков: корреляция числовых признаков ---
numeric_cols = ['amount', 'last_hour_num_transactions', 'last_hour_total_amount',
                'last_hour_unique_merchants', 'last_hour_unique_countries', 'last_hour_max_single_amount']

plt.figure(figsize=(10, 8))
correlation_matrix = df_fraud[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Матрица корреляции числовых признаков')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
plt.show()

# --- 13. Влияние присутствия карты и высокого риска ---
print("\nМошенничество по признакам:")
print(df_fraud.groupby('is_card_present')['is_fraud'].mean())
print(df_fraud.groupby('is_high_risk_vendor')['is_fraud'].mean())
print(df_fraud.groupby('is_outside_home_country')['is_fraud'].mean())

print("\nПостроение объединённого графика: is_card_present, is_high_risk_vendor, is_outside_home_country")

# Вычисляем долю мошенничества
fraud_by_card_present = df_fraud.groupby('is_card_present')['is_fraud'].mean()
fraud_by_high_risk_vendor = df_fraud.groupby('is_high_risk_vendor')['is_fraud'].mean()
fraud_by_outside_home = df_fraud.groupby('is_outside_home_country')['is_fraud'].mean()

# Подготавливаем данные для объединённого барплота
data = []

for feature, series, label in [
    ('is_card_present', fraud_by_card_present, 'Карта присутствует'),
    ('is_high_risk_vendor', fraud_by_high_risk_vendor, 'Высокорисковый вендор'),
    ('is_outside_home_country', fraud_by_outside_home, 'За пределами страны')
]:
    for value, fraud_rate in series.items():
        data.append({
            'Признак': label,
            'Значение': 'Да' if value else 'Нет',
            'Доля фрода': fraud_rate
        })

df_plot = pd.DataFrame(data)

# Заменяем названия осей
df_plot['Признак'] = pd.Categorical(
    df_plot['Признак'],
    categories=['Карта присутствует', 'Высокорисковый вендор', 'За пределами страны'],
    ordered=True
)

# Строим график
plt.figure(figsize=(12, 6))
bar = sns.barplot(
    data=df_plot,
    x='Признак',
    y='Доля фрода',
    hue='Значение',
    palette=['#2E8B57', '#DC143C']  # зелёный (Нет), красный (Да)
)

plt.title('Влияние ключевых бинарных признаков на долю мошенничества', fontsize=14, pad=20)
plt.xlabel('Признак')
plt.ylabel('Доля мошеннических транзакций')
plt.ylim(0, max(df_plot['Доля фрода']) * 1.15)
plt.legend(title='Значение признака', loc='upper right')

# Подписи на столбцах
for i, row in df_plot.iterrows():
    plt.text(
        i / 3 + (i % 2) * 0.21 - 0.1,  # хитрый x для позиционирования в группе
        row['Доля фрода'] + 0.005,
        f"{row['Доля фрода']:.3f}",
        ha='center',
        fontsize=9,
        color='black'
    )

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_by_binary_features_combined.png'), dpi=300)
plt.show()

# --- 14. Анализ валют ---
print("\nРаспределение валют:")
print(df_fraud['currency'].value_counts())

# --- 15. Подключение обменных курсов (пример: конвертация в USD) ---
# Предположим, мы хотим перевести все транзакции в USD
df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.date

df_fraud['date'] = pd.to_datetime(df_fraud['date'])
df_fraud['date_only'] = df_fraud['date'].dt.date

# Мелкое объединение: только нужные валюты
currencies = [col for col in df_exchange.columns if col != 'date' and col != 'USD']
exchange_stacked = df_exchange.melt(id_vars='date', value_vars=currencies,
                                    var_name='currency', value_name='exchange_rate_to_usd')

# Добавим USD с курсом 1.0
usd_row = pd.DataFrame({'date': df_exchange['date'].unique(),
                        'currency': 'USD',
                        'exchange_rate_to_usd': 1.0})
exchange_stacked = pd.concat([exchange_stacked, usd_row], ignore_index=True)

# Объединение с основными данными
df_fraud_usd = df_fraud.merge(exchange_stacked, left_on=['date_only', 'currency'],
                              right_on=['date', 'currency'], how='left')

# Конвертация суммы в USD
df_fraud_usd['amount_usd'] = df_fraud_usd['amount'] / df_fraud_usd['exchange_rate_to_usd']

print(f"Средняя сумма мошеннических транзакций в USD: {df_fraud_usd[df_fraud_usd['is_fraud']]['amount_usd'].mean():.2f}")
print(f"Средняя сумма легитимных транзакций в USD: {df_fraud_usd[~df_fraud_usd['is_fraud']]['amount_usd'].mean():.2f}")

# --- 16. НОВЫЕ ГИПОТЕЗЫ ИЗ АНАЛИЗА ---

print("\n" + "="*60)
print("НАЧАЛО ПРОВЕРКИ ДОПОЛНИТЕЛЬНЫХ ГИПОТЕЗ")
print("="*60)

# Гипотеза 1: Молодой аккаунт → выше риск
print("\nПроверка гипотезы 1: Молодые аккаунты (маленький возраст) имеют более высокую долю мошенничества.")
# В README указано, что данные о возрасте аккаунта есть, но в данных его нет → нужно имитировать
# Для примера: добавим случайный признак account_age_days (в реальности — из customer_profile)
np.random.seed(42)
df_fraud_usd['account_age_days'] = np.random.lognormal(mean=4, sigma=1.5, size=len(df_fraud_usd)).astype(int)
df_fraud_usd['account_age_days'] = df_fraud_usd['account_age_days'].clip(1, 3650)  # 1–10 лет

young_threshold = 30
young_fraud_rate = df_fraud_usd[df_fraud_usd['account_age_days'] <= young_threshold]['is_fraud'].mean()
old_fraud_rate = df_fraud_usd[df_fraud_usd['account_age_days'] > young_threshold]['is_fraud'].mean()

print(f"Доля фрода у аккаунтов младше {young_threshold} дней: {young_fraud_rate:.4f}")
print(f"Доля фрода у аккаунтов старше {young_threshold} дней: {old_fraud_rate:.4f}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_fraud_usd, x='is_fraud', y='account_age_days')
plt.yscale('log')
plt.title('Возраст аккаунта по классам')
plt.xlabel('is_fraud')
plt.ylabel('Возраст аккаунта (дней)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'account_age_vs_fraud.png'), dpi=300)
plt.show()

# Гипотеза 2: Долгое бездействие → внезапная активация
print("\nПроверка гипотезы 2: Аккаунты с долгим бездействием, вдруг активировавшиеся, чаще мошенничают.")
df_fraud_usd = df_fraud_usd.sort_values(['customer_id', 'timestamp'])
df_fraud_usd['prev_timestamp'] = df_fraud_usd.groupby('customer_id')['timestamp'].shift(1)
df_fraud_usd['inactive_days'] = (df_fraud_usd['timestamp'] - df_fraud_usd['prev_timestamp']).dt.total_seconds() / (3600 * 24)
df_fraud_usd['inactive_days'] = df_fraud_usd['inactive_days'].clip(0, 365)

long_inactive = df_fraud_usd[df_fraud_usd['inactive_days'] > 30]
if not long_inactive.empty:
    print(f"Доля фрода при бездействии >30 дней: {long_inactive['is_fraud'].mean():.4f}")
else:
    print("Нет данных для транзакций с длительным бездействием.")

plt.figure(figsize=(10, 6))
sns.histplot(data=df_fraud_usd, x='inactive_days', hue='is_fraud', bins=50, log_scale=(False, True))
plt.title('Распределение времени бездействия перед транзакцией')
plt.xlabel('Бездействие (дней)')
plt.ylabel('Частота (логарифм)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'inactive_days_vs_fraud.png'), dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_fraud_usd, x='inactive_days', hue='is_fraud', log_scale=True, fill=True, alpha=0.3)
plt.title('Распределение времени бездействия перед транзакцией')
plt.xlabel('Бездействие (дней)')
plt.ylabel('Плотность')
plt.legend(title='is_fraud', labels=['Легитимные', 'Мошеннические'])
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'inactive_days_kde.png'), dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df_fraud_usd, x='inactive_days', hue='is_fraud', bins=50, log_scale=True, element='step', common_norm=False)
plt.title('Распределение времени бездействия перед транзакцией')
plt.xlabel('Бездействие (дней)')
plt.ylabel('Частота (логарифмическая шкала)')
plt.legend(title='is_fraud', labels=['Легитимные', 'Мошеннические'])
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'inactive_days_hist_log.png'), dpi=300)
plt.show()

# plt.figure(figsize=(10, 6))
# sns.stripplot(data=df_fraud_usd, x='inactive_days', y='is_fraud', jitter=True, alpha=0.5)
# plt.title('Распределение времени бездействия перед транзакцией')
# plt.xlabel('Бездействие (дней)')
# plt.ylabel('Тип транзакции')
# plt.legend(title='is_fraud', labels=['Легитимные', 'Мошеннические'])
# plt.tight_layout()
# plt.savefig(os.path.join(plots_dir, 'inactive_days_stripplot.png'), dpi=300)
# plt.show()

# --- Гипотеза 3: Много стран за час → физически невозможно ---
print("\nПроверка гипотезы 3: Если за час транзакции в 2+ странах — это подозрительно.")
multi_country = df_fraud_usd[df_fraud_usd['last_hour_unique_countries'] >= 2]
fraud_rate_multi = multi_country['is_fraud'].mean()
print(f"Доля фрода при 2+ странах за час: {fraud_rate_multi:.4f}")
print(f"Общее число таких транзакций: {len(multi_country)}")

# График: доля мошенничества в зависимости от числа уникальных стран за час
plt.figure(figsize=(10, 6))
country_fraud_rate = df_fraud_usd.groupby('last_hour_unique_countries')['is_fraud'].mean()
sns.lineplot(x=country_fraud_rate.index, y=country_fraud_rate.values, marker='o', color='red', label='Доля фрода')
plt.axvline(x=2, color='gray', linestyle='--', linewidth=1)
plt.text(2.1, 0.3, 'Порог: 2 страны', color='gray', fontsize=10)

plt.title('Доля мошенничества по числу уникальных стран за последний час')
plt.xlabel('Число уникальных стран за час')
plt.ylabel('Доля мошеннических транзакций')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_by_unique_countries.png'), dpi=300)
plt.show()

# --- Гипотеза 4: Высокое число уникальных мерчантов за час — признак card testing ---
print("\nПроверка гипотезы 4: Высокое число уникальных мерчантов за час — признак card testing.")
high_merchants = df_fraud_usd[df_fraud_usd['last_hour_unique_merchants'] >= 5]
fraud_rate_merch = high_merchants['is_fraud'].mean()
print(f"Доля фрода при 5+ мерчантов за час: {fraud_rate_merch:.4f}")

# График: доля фрода в зависимости от числа уникальных мерчантов за час
plt.figure(figsize=(10, 6))
merch_fraud_rate = df_fraud_usd.groupby('last_hour_unique_merchants')['is_fraud'].mean()

# Ограничиваем до 20, чтобы график не "хвостил"
merch_fraud_rate_limited = merch_fraud_rate[merch_fraud_rate.index <= 20]

sns.lineplot(x=merch_fraud_rate_limited.index, y=merch_fraud_rate_limited.values, marker='s', color='blue', label='Доля фрода')
plt.axvline(x=5, color='gray', linestyle='--', linewidth=1)
plt.text(5.1, 0.3, 'Порог: 5 мерчантов', color='gray', fontsize=10)

plt.title('Доля мошенничества по числу уникальных мерчантов за последний час')
plt.xlabel('Число уникальных мерчантов за час')
plt.ylabel('Доля мошеннических транзакций')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_by_unique_merchants.png'), dpi=300)
plt.show()

# Гипотеза 5: Комбо-риск: онлайн + CNP + за границей
print("\nПроверка гипотезы 5: Комбинация: онлайн + карта не присутствует + за границей → высокий риск.")
high_risk_combo = (
    (df_fraud_usd['vendor_type'] == 'онлайн') &
    (~df_fraud_usd['is_card_present']) &
    (df_fraud_usd['is_outside_home_country'])
)
if high_risk_combo.any():
    print(f"Доля фрода в высокорисковой комбинации: {df_fraud_usd[high_risk_combo]['is_fraud'].mean():.4f}")
    print(f"Количество таких транзакций: {high_risk_combo.sum()}")
else:
    print("Нет данных для высокорисковой комбинации.")

# --- Гипотеза 6: Размер города как фактор риска ---
print("\nПроверка гипотезы 6: Транзакции в малых городах могут быть рискованнее.")
city_risk = df_fraud_usd.groupby('city_size')['is_fraud'].mean().sort_values(ascending=False)
print("Доля фрода по размеру города:")
print(city_risk)

# График: доля мошенничества по размеру города
plt.figure(figsize=(8, 5))
sns.barplot(x=city_risk.index, y=city_risk.values, palette='Reds_r')
plt.title('Доля мошенничества по размеру города')
plt.xlabel('Размер города')
plt.ylabel('Доля мошеннических транзакций')
plt.ylim(0, max(city_risk) * 1.1)  # немного места сверху

# Подписи значений на столбцах
for i, v in enumerate(city_risk.values):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_by_city_size.png'), dpi=300)
plt.show()

# --- Гипотеза 7: Тип карты влияет на риск мошенничества ---
print("\nПроверка гипотезы 7: Тип карты влияет на риск мошенничества.")
card_risk = df_fraud_usd.groupby('card_type')['is_fraud'].mean().sort_values(ascending=False)
print("Доля фрода по типу карты:")
print(card_risk)

# График: доля мошенничества по типу карты
plt.figure(figsize=(10, 6))
sns.barplot(x=card_risk.index, y=card_risk.values, palette='Blues_r')
plt.title('Доля мошенничества по типу карты')
plt.xlabel('Тип карты')
plt.ylabel('Доля мошеннических транзакций')
plt.ylim(0, max(card_risk) * 1.1)

# Подписи значений на столбцах
for i, v in enumerate(card_risk.values):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'fraud_by_card_type.png'), dpi=300)
plt.show()

# --- Гипотеза 8: Анализ транзакций в выходные дни по категориям ---
print("\nПроверка гипотезы 8: Количество легитимных и мошеннических транзакций в выходные дни по категориям.")

target_categories = ['Образование', 'Офисные поставки']

# Фильтруем: только выходные + целевые категории
weekend_anomaly = df_fraud_usd[
    (df_fraud_usd['is_weekend'])
]

if not weekend_anomaly.empty:
    # Группируем по категории и is_fraud
    fraud_analysis = weekend_anomaly.groupby(['vendor_category', 'is_fraud']).size().unstack(fill_value=0)
    fraud_analysis = fraud_analysis.rename(columns={False: 'legit_count', True: 'fraud_count'})
    fraud_analysis['total'] = fraud_analysis['legit_count'] + fraud_analysis['fraud_count']
    fraud_analysis['fraud_ratio'] = (fraud_analysis['fraud_count'] / fraud_analysis['total']).round(4)

    print("\nРазбивка по категориям:")
    print(fraud_analysis)

    # Общее количество
    total_legit = fraud_analysis['legit_count'].sum()
    total_fraud = fraud_analysis['fraud_count'].sum()
    print(f"\nВсего легитимных транзакций в указанных категориях в выходные: {total_legit}")
    print(f"Всего мошеннических транзакций в указанных категориях в выходные: {total_fraud}")
    print(f"Общее количество транзакций: {total_legit + total_fraud}")

    # === ГРАФИК ===
    plt.figure(figsize=(10, 6))
    fraud_plot = fraud_analysis[['legit_count', 'fraud_count']].copy()
    fraud_plot.plot(kind='bar', color=['#2E8B57', '#DC143C'], alpha=0.8)
    plt.title('Транзакции в выходные: Легитимные vs Мошеннические\n(категории: Образование, Офисные поставки)')
    plt.xlabel('Категория')
    plt.ylabel('Количество транзакций')
    plt.xticks(rotation=0)
    plt.legend(['Легитимные', 'Мошеннические'])
    plt.grid(axis='y', alpha=0.3)

    # Подписи на столбцах
    for i, (idx, row) in enumerate(fraud_plot.iterrows()):
        plt.text(i, row['legit_count'] + 1, str(row['legit_count']), ha='center', va='bottom', fontsize=10)
        plt.text(i, row['fraud_count'] + 1, str(row['fraud_count']), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'weekend_fraud_by_category.png'), dpi=300)
    plt.show()

else:
    print("Нет транзакций, соответствующих условиям: выходные дни и категории ['Образование', 'Офисные поставки']")

# Гипотеза 9: Смена устройства / отсутствие отпечатка
print("\nПроверка гипотезы 9: Отсутствие device_fingerprint или смена устройства — признак риска.")
df_fraud_usd['has_fingerprint'] = df_fraud_usd['device_fingerprint'].notna()
print(f"Доля фрода без отпечатка устройства: {df_fraud_usd[~df_fraud_usd['has_fingerprint']]['is_fraud'].mean():.4f}")


# --- ОПТИМИЗИРОВАННАЯ РАСШИРЕННАЯ МАТРИЦА КОРРЕЛЯЦИЙ ---
print("\n" + "="*60)
print("ОПТИМИЗИРОВАННАЯ РАСШИРЕННАЯ МАТРИЦА КОРРЕЛЯЦИЙ")
print("Включает is_fraud, числовые, бинарные и важные категориальные признаки")
print("="*60)

df = df_fraud_usd.copy()

# 1. Удаляем ID и бесполезные строковые поля
exclude_cols = [
    'transaction_id', 'customer_id', 'card_number',
    'ip_address', 'device_fingerprint', 'timestamp',
    'date', 'date_only', 'vendor', 'city'
]
df = df.drop(columns=[col for col in exclude_cols if col in df.columns])

# 2. Преобразуем bool → int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# 3. Находим категориальные признаки
cat_cols = df.select_dtypes(include='object').columns

# 4. Оставляем только те с ≤15 уникальных значений
max_categories = 15
small_cat_cols = [col for col in cat_cols if df[col].nunique() <= max_categories]
print(f"Категориальные признаки с ≤{max_categories} уникальных значений: {small_cat_cols}")

# 5. One-Hot Encoding С ОБЯЗАТЕЛЬНЫМ УДАЛЕНИЕМ СТАРЫХ СТОЛБЦОВ
if len(small_cat_cols) > 0:
    df = pd.get_dummies(
        df,
        columns=small_cat_cols,           # ✅ Ключевой параметр!
        prefix=small_cat_cols,
        dtype=int
    )
    print(f"✅ One-Hot Encoding применён к: {small_cat_cols}")
else:
    print("⚠️  Нет категориальных признаков для кодирования.")

# 6. Проверка: остались ли строковые колонки?
remaining_object_cols = df.select_dtypes(include='object').columns
if len(remaining_object_cols) > 0:
    print(f"❌ Остались строковые колонки! Проблема: {list(remaining_object_cols)}")
    # Удаляем их (на всякий случай)
    df = df.drop(columns=remaining_object_cols)
    print(f"❌ Удалены оставшиеся строковые колонки: {list(remaining_object_cols)}")

# 7. Убеждаемся, что всё числовое
if df.select_dtypes(include=['object']).empty:
    print(f"✅ Все признаки числовые. Число признаков: {df.shape[1]}")
else:
    print("❌ Есть оставшиеся нечисловые признаки!")
    print(df.dtypes[df.dtypes == 'object'])
    raise ValueError("Остались строковые колонки — нельзя строить корреляцию.")

# 8. Проверка is_fraud
if 'is_fraud' not in df.columns:
    raise ValueError("❌ is_fraud отсутствует в данных!")

# 9. Корреляция
plt.figure(figsize=(16, 14))
corr = df.corr(method='pearson')

# Маска для верхнего треугольника
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    cmap='coolwarm',
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,
    annot=False
)
plt.title('Расширенная корреляционная матрица\n(включая закодированные категориальные признаки)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_matrix_enhanced.png'), dpi=200, bbox_inches='tight')
plt.show()

print(f"✅ Матрица корреляций сохранена: {os.path.join(plots_dir, 'correlation_matrix_enhanced.png')}")

# 10. Топ-10 признаков по корреляции с is_fraud
fraud_corr = corr['is_fraud'].drop('is_fraud', errors='ignore').sort_values(key=abs, ascending=False).head(10)
print("\n🔝 Топ-10 признаков по модулю корреляции с is_fraud:")
print(fraud_corr.round(4))

# --- 17. Сохранение обработанных данных ---
output_file = 'data/processed_fraud_data.parquet'
df_fraud_usd.to_parquet(output_file, index=False)
print(f"\n Обработанные данные сохранены в {output_file}")
print("\n EDA завершён. Все графики сохранены в папку plots/")