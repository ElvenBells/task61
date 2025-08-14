# modeling_updated.py
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Регистрация tqdm для pandas
tqdm.pandas()

print("🚀 Загрузка данных...")
df_fraud = pd.read_parquet('data/transaction_fraud_data.parquet')
df_exchange = pd.read_parquet('data/historical_currency_exchange.parquet')

# --- 1. Базовая предобработка ---
print("🔧 Преобразование timestamp и извлечение признаков...")
df_fraud['timestamp'] = pd.to_datetime(df_fraud['timestamp'])
df_fraud['hour'] = df_fraud['timestamp'].dt.hour
df_fraud['is_weekend'] = df_fraud['timestamp'].dt.dayofweek >= 5  # Сб=5, Вс=6
df_fraud['date'] = df_fraud['timestamp'].dt.date

# Распаковка last_hour_activity
df_activity = pd.json_normalize(df_fraud['last_hour_activity'])
df_activity.columns = [f"last_hour_{col}" for col in df_activity.columns]
df_fraud = pd.concat([df_fraud.drop(columns=['last_hour_activity']), df_activity], axis=1)

# --- 2. Конвертация в USD ---
print("💱 Конвертация суммы в USD...")
df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.date
currencies = [col for col in df_exchange.columns if col != 'date']
exchange_stacked = df_exchange.melt(id_vars='date', value_vars=currencies,
                                    var_name='currency', value_name='exchange_rate_to_usd')
usd_row = pd.DataFrame({'date': df_exchange['date'].unique(),
                        'currency': 'USD', 'exchange_rate_to_usd': 1.0})
exchange_stacked = pd.concat([exchange_stacked, usd_row], ignore_index=True)

df_fraud = df_fraud.merge(exchange_stacked, on=['date', 'currency'], how='left')
df_fraud['amount_usd'] = df_fraud['amount'] * df_fraud['exchange_rate_to_usd']

print(f"✅ Данные загружены. Размер: {df_fraud.shape}")

# --- 3. Сортировка для временных признаков ---
df_fraud = df_fraud.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)

# --- 4. Генерация новых признаков на основе EDA ---

print("⚙️ Генерация новых признаков...")

# 4.1. Время с предыдущей транзакции (в часах)
df_fraud['time_since_last_transaction'] = (
    df_fraud.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 3600
)
df_fraud['time_since_last_transaction'].fillna(0, inplace=True)

# 4.2. Возраст аккаунта (в днях) — имитация, в реальности из профиля
np.random.seed(42)
df_fraud['account_age_days'] = np.random.lognormal(mean=4, sigma=1.5, size=len(df_fraud)).astype(int)
df_fraud['account_age_days'] = df_fraud['account_age_days'].clip(1, 3650)

# 4.3. Долгое бездействие (>30 дней)
df_fraud['prev_timestamp'] = df_fraud.groupby('customer_id')['timestamp'].shift(1)
df_fraud['inactive_days'] = (df_fraud['timestamp'] - df_fraud['prev_timestamp']).dt.total_seconds() / (3600 * 24)
df_fraud['inactive_days'] = df_fraud['inactive_days'].clip(0, 365)

# 4.4. Много стран за час → подозрительно
df_fraud['is_multi_country_hour'] = (df_fraud['last_hour_unique_countries'] >= 2).astype(int)

# 4.5. Высокий всплеск мерчантов → card testing
df_fraud['is_high_merchant_burst'] = (df_fraud['last_hour_unique_merchants'] >= 5).astype(int)

# 4.6. Высокорисковая комбинация: онлайн + CNP + за границей
df_fraud['is_high_risk_combo'] = (
    (df_fraud['vendor_type'] == 'онлайн') &
    (~df_fraud['is_card_present']) &
    (df_fraud['is_outside_home_country'])
).astype(int)

# 4.7. Отсутствие отпечатка устройства
df_fraud['has_device_fingerprint'] = df_fraud['device_fingerprint'].notna().astype(int)

# 4.8. Среднее и стандартное отклонение расходов (с прогресс-баром)
print("📊 Вычисление past_avg_amount и past_std_amount...")
df_fraud['past_avg_amount'] = df_fraud.groupby('customer_id')['amount_usd'].progress_apply(
    lambda x: x.shift(1).expanding().mean()
).reset_index(level=0, drop=True)
df_fraud['past_std_amount'] = df_fraud.groupby('customer_id')['amount_usd'].progress_apply(
    lambda x: x.shift(1).expanding().std()
).reset_index(level=0, drop=True)
df_fraud['past_avg_amount'].fillna(df_fraud['amount_usd'].median(), inplace=True)
df_fraud['past_std_amount'].fillna(df_fraud['amount_usd'].std(), inplace=True)

df_fraud['spending_deviation_score'] = (
    (df_fraud['amount_usd'] - df_fraud['past_avg_amount']) / (df_fraud['past_std_amount'] + 1e-6)
)

# 4.9. Velocity: сумма за последние 24 часа
def calc_velocity(group):
    times = group['timestamp'].values
    amounts = group['amount_usd'].values
    velocity = np.zeros(len(group))
    for i in range(1, len(group)):
        window_start = times[i] - timedelta(hours=24)
        recent_mask = (times[:i] >= window_start) & (times[:i] < times[i])
        velocity[i] = amounts[:i][recent_mask].sum() if recent_mask.any() else 0
    return pd.Series(velocity, index=group.index)

print("⏳ Вычисление velocity_score (сумма за 24 часа)...")
df_fraud['velocity_score'] = df_fraud.groupby('customer_id', group_keys=False).progress_apply(calc_velocity)

# 4.10. isNewDevice: новое устройство для клиента
def is_new_device(group):
    seen = set()
    result = []
    for dev in group['device']:
        result.append(1 if dev not in seen else 0)
        seen.add(dev)
    return result

df_fraud['isNewDevice'] = df_fraud.groupby('customer_id', group_keys=False)['device'].progress_apply(is_new_device).reset_index(level=0, drop=True)

print("✅ Все признаки сгенерированы.")

# --- 5. Выбор финальных признаков ---
print("📋 Формирование признакового пространства...")

features = [
    # Базовые
    'amount_usd',
    'last_hour_num_transactions',
    'last_hour_total_amount',
    'last_hour_unique_merchants',
    'last_hour_unique_countries',
    'last_hour_max_single_amount',
    # Новые временные
    'time_since_last_transaction',
    'inactive_days',
    'account_age_days',
    # Поведенческие
    'spending_deviation_score',
    'velocity_score',
    # Бинарные риски
    'is_card_present',
    'is_outside_home_country',
    'is_high_risk_vendor',
    'is_weekend',
    'has_device_fingerprint',
    'isNewDevice',
    'is_multi_country_hour',
    'is_high_merchant_burst',
    'is_high_risk_combo',
    # Категориальные
    'hour',
    'city_size',
    'card_type',
    'vendor_category'
]

# Фильтр: только существующие
features = [f for f in features if f in df_fraud.columns]
X = df_fraud[features].copy()
y = df_fraud['is_fraud'].astype(int)

# --- 6. Кодирование категориальных признаков ---
from sklearn.preprocessing import LabelEncoder

cat_cols = ['city_size', 'card_type', 'vendor_category']
le_dict = {}

for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le  # сохраняем энкодеры

# Bool → int
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

# Заполнение пропусков
X = X.fillna(0)

print(f"✅ X.shape: {X.shape}, y.shape: {y.shape}")

# --- 7. Обучение модели ---
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Вес класса
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"⚖️ Вес положительного класса: {scale_pos_weight:.2f}")

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False,
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

print("🏋️ Обучение модели...")
model.fit(X_train, y_train)

# --- 8. Оценка ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")
# График PR-кривой
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 9. Важность признаков ---
print("\n🔝 Топ-10 важных признаков:")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print(importance_df)

# --- 10. Сохранение ---
print("💾 Сохранение модели и компонентов...")
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(features, 'feature_names.pkl')

print("✅ Модель и компоненты успешно сохранены.")