# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR


# 데이터 생성

df = pd.read_csv('data.csv')

# 데이터 준비
X = df[['*금융 문해력']]
y = df['*적금 관심도']

# 데이터 필터링
mask = (X['*금융 문해력'] >= 2.5) & (X['*금융 문해력'] <= 5.0)
X_filtered = X[mask]
y_filtered = y[mask]


# 학습용과 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, 
    y_filtered, 
    test_size=0.2, 
    random_state=42,
    shuffle=True,
)

# 2. 교차 검증
def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f'\n교차 검증 RMSE 점수: {rmse_scores}')
    print(f'평균 RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})')

# 3. 여러 블랙박스 모델 정의
models = {
    'linear': LinearRegression(),
    'rf': RandomForestRegressor(random_state=42),
    'gbm': GradientBoostingRegressor(random_state=42),
    'svr': SVR(kernel='rbf')
}

# 4. 하이퍼파라미터 탐색
def perform_grid_search(X, y):
    # RandomForest 하이퍼파라미터 그리드
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, 
        rf_params, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    print("\n최적 하이퍼파라미터:", grid_search.best_params_)
    print("최적 RMSE:", np.sqrt(-grid_search.best_score_))
    return grid_search.best_estimator_

# 5. 앙상블 모델 생성
def create_ensemble(models_dict, X, y):
    estimators = []
    for name, model in models_dict.items():
        model.fit(X, y)
        estimators.append((name, model))
    
    ensemble = VotingRegressor(estimators=estimators)
    return ensemble



# 모델 실행


# 선형 회귀 모델 평가
def print_metrics(y_true, y_pred, model_name, X):
    # MSE와 RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # R2 score
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R2
    n = len(y_true)  # 샘플 수
    p = X.shape[1]   # 특성 수
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # 상관계수 (Pearson correlation)
    correlation, p_value = stats.pearsonr(y_true, y_pred)
    
    print(f"\n{model_name} 모델 평가 지표:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"상관계수 (Pearson): {correlation:.4f}")
    print(f"상관계수 P-value: {p_value:.4f}")
    print("-" * 60)

# 모델 평가 및 실행
def evaluate_all_models(X_train, X_test, y_train, y_test):
    results = {}
    
    # 각 모델 평가
    for name, model in models.items():
        print(f"\n{name} 모델 평가 중...")
        model.fit(X_train, y_train)
        perform_cross_validation(model, X_train, y_train)
        y_pred = model.predict(X_test)
        print_metrics(y_test, y_pred, name, X_test)
        results[name] = model
    
    # 그리드 서치로 최적화된 RandomForest
    print("\n하이퍼파라미터 최적화 중...")
    best_rf = perform_grid_search(X_train, y_train)
    
    # 앙상블 모델 생성 및 평가
    print("\n앙상블 모델 생성 중...")
    ensemble = create_ensemble(results, X_train, y_train)
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    print_metrics(y_test, y_pred_ensemble, "Ensemble", X_test)
    
    return ensemble, best_rf

ensemble_model, best_rf = evaluate_all_models(X_train, X_test, y_train, y_test)

# 시각화 업데이트
plt.figure(figsize=(12, 8))
plt.scatter(X_filtered, y_filtered, color='blue', label='실제 데이터', alpha=0.5)
plt.plot(X_filtered, ensemble_model.predict(X_filtered), color='red', label='앙상블 모델')
plt.plot(X_filtered, best_rf.predict(X_filtered), color='green', label='최적화된 RF')
plt.xlabel('금융 문해력')
plt.ylabel('적금 관심도')
plt.title('금융 문해력과 적금 관심도의 관계 (다양한 모델 비교)')
plt.legend()
plt.show()
