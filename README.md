# AIML_Project - 김건

| 모델 | 하이퍼파라미터 | 설정값 | 설명 |
|------|----------------|--------|------|
| LinearRegression | - | - | - |
| RandomForestRegressor | random_state | 42 | 재현성을 위한 랜덤 시드 값 |
| GradientBoostingRegressor | random_state | 42 | 재현성을 위한 랜덤 시드 값 |
| SVR | kernel | 'rbf' | RBF(Radial Basis Function) 커널 사용 |
### RandomForest 그리드서치 하이퍼파라미터
| 하이퍼파라미터 | 탐색 범위 | 설명 |
|----------------|------------|------|
| n_estimators | [100, 200] | 의사결정 트리의 개수 |
| max_depth | [None, 10, 20] | 트리의 최대 깊이 (None은 제한 없음) |
| min_samples_split | [2, 5] | 노드를 분할하기 위한 최소 샘플 수 |
| min_samples_leaf | [1, 2] | 리프 노드가 가져야 할 최소 샘플 수 |
### 교차 검증 설정
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| cv | 5 | 5-fold 교차 검증 수행 |
| scoring | neg_mean_squared_error | 평가 지표로 MSE의 음수값 사용 |
| n_jobs | -1 | 모든 가용 CPU 코어 사용 |
### 데이터 분할
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| test_size | 0.2 | 테스트 세트 비율 20% |
| random_state | 42 | 재현성을 위한 랜덤 시드 값 |
| shuffle | True | 데이터 섞기 활성화 |

