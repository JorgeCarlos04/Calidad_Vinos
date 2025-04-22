
import pandas as pd  # Manipulación y análisis de datos
import seaborn as sns  # Visualización de datos
import matplotlib.pyplot as plt  # Gráficos básicos
from sklearn.model_selection import train_test_split, GridSearchCV  # División de datos y ajuste de hiperparámetros
from sklearn.preprocessing import StandardScaler  # Estandarización de características
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Árboles de decisión
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.metrics import accuracy_score, classification_report  # Métricas de evaluación

df = pd.read_csv('winequality-red.csv', delimiter=';')

print("Primeras filas:\n", df.head())
print("\nEstadísticas descriptivas:\n", df.describe())
print("\nDistribución de calidades:\n", df['quality'].value_counts())


plt.figure(figsize=(8, 4))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribución de Calidades del Vino', fontsize=14)
plt.xlabel('Calidad (0-10)', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=df, palette='coolwarm')
plt.title('Relación Alcohol vs Calidad', fontsize=14)
plt.xlabel('Calidad', fontsize=12)
plt.ylabel('Alcohol (% vol)', fontsize=12)
plt.show()


X = df.drop('quality', axis=1)  # Todas las columnas excepto quality
y = df['quality']  # Variable objetivo


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,  # Semilla para reproducibilidad
    stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajuste y transformación del train
X_test_scaled = scaler.transform(X_test)  # Solo transformación del test


dt_model = DecisionTreeClassifier(
    random_state=42,  # Semilla para reproducibilidad
    max_depth=5  # Limitar profundidad para evitar overfitting
)


dt_model.fit(X_train_scaled, y_train)


y_pred_dt = dt_model.predict(X_test_scaled)
print("\n--- Evaluación Árbol de Decisión ---")
print(f"Precisión: {accuracy_score(y_test, y_pred_dt):.2f}")
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_dt))


plt.figure(figsize=(20, 10))
plot_tree(
    dt_model, 
    feature_names=X.columns, 
    class_names=sorted(y.astype(str).unique()), 
    filled=True, 
    rounded=True, 
    max_depth=2  # Mostrar solo primeros 2 niveles
)
plt.title("Árbol de Decisión (Primeros 2 niveles)", fontsize=14)
plt.show()


rf_model = RandomForestClassifier(
    n_estimators=100,  # Número de árboles en el bosque
    random_state=42,
    n_jobs=-1  # Usar todos los núcleos del CPU
)


param_grid = {
    'n_estimators': [100, 200],  # Número de árboles a probar
    'max_depth': [None, 10, 20],  # Profundidad máxima
    'min_samples_split': [2, 5]  # Mínimo muestras para dividir nodo
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # Validación cruzada de 5 folds
    scoring='accuracy'  # Métrica de evaluación
)


grid_search.fit(X_train_scaled, y_train)


best_rf = grid_search.best_estimator_
print("\nMejores hiperparámetros:", grid_search.best_params_)


y_pred_rf = best_rf.predict(X_test_scaled)
print("\n--- Evaluación Random Forest ---")
print(f"Precisión: {accuracy_score(y_test, y_pred_rf):.2f}")
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_rf))


importancias = best_rf.feature_importances_
features = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importancia', 
    y='Característica', 
    data=features, 
    palette='plasma'
)
plt.title('Importancia de Características - Random Forest', fontsize=14)
plt.xlabel('Importancia', fontsize=12)
plt.ylabel('')
plt.show()
