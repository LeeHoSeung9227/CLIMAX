"""
시계열 검증: 2021년 훈련 → 2022년 검증
실제 시나리오와 동일한 검증 방법
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class TimeSeriesValidator:
    """2021→2023 시계열 검증"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
        
    def load_data(self, train_years=['21'], test_year='23', sample_size=5000000):
        """복수 학습 데이터(21,22년)와 테스트 데이터(23년) 로드"""
        print(f"🚀 시계열 검증 데이터 로드")
        print(f"  📚 훈련: {', '.join('20'+y for y in train_years)}년 데이터")
        print(f"  🎯 검증: 20{test_year}년 데이터")
        print("="*60)

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(base_dir, '..', '데이터'))
            
            # 여러 학습 데이터 불러와서 concat
            train_dfs = []
            for y in train_years:
                train_file = os.path.join(data_dir, f'train_subway{y}.csv')
                print(f"📊 20{y}년 훈련 데이터 로드 중...")
                df = pd.read_csv(train_file, encoding='cp949', nrows=sample_size)
                df = self._preprocess_data(df, y)
                train_dfs.append(df)
            self.train_data = pd.concat(train_dfs).reset_index(drop=True)
            
            # 테스트 데이터 로드
            test_file = os.path.join(data_dir, f'train_subway{test_year}.csv')
            print(f"📊 20{test_year}년 검증 데이터 로드 중...")
            test_df = pd.read_csv(test_file, encoding='cp949', nrows=sample_size)
            self.test_data = self._preprocess_data(test_df, test_year)

            print(f"✅ 데이터 로드 완료!")
            print(f"  📚 훈련 데이터: {len(self.train_data):,}개 ({', '.join('20'+y for y in train_years)}년)")
            print(f"  🎯 검증 데이터: {len(self.test_data):,}개 (20{test_year}년)")
            return True
        
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def _preprocess_data(self, df, year):
        """데이터 전처리"""
        # 컬럼명 정리
        df.columns = [col.replace(f'train_subway{year}.', '') for col in df.columns]
        
        # 시간 특성 생성
        df['datetime'] = pd.to_datetime(df['tm'], format='%Y%m%d%H')
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # 계절 특성
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # 겨울
            3: 1, 4: 1, 5: 1,   # 봄
            6: 2, 7: 2, 8: 2,   # 여름
            9: 3, 10: 3, 11: 3  # 가을
        })
        
        # 출퇴근 시간 특성
        df['is_rush_hour'] = df['hour'].apply(
            lambda x: 1 if x in [7, 8, 9, 18, 19, 20] else 0
        )
        
        # 순환적 시간 특성
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 기상 데이터 전처리
        weather_cols = ['ta', 'ws', 'rn_hr1', 'hm']
        for col in weather_cols:
            if col in df.columns:
                # 이상치 처리
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        
        # 결측값 처리
        df = df.fillna(df.median(numeric_only=True))
        
        return df
    
    def prepare_features(self):
        """특성 준비 (훈련/검증 데이터 일관성 보장)"""
        print("\n📊 특성 준비 중...")
        
        # 기본 특성 선택
        base_feature_cols = [
            'hour', 'dayofweek', 'month', 'day', 'season',
            'is_weekend', 'is_rush_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos',
            'ta', 'ws', 'rn_hr1', 'hm'
        ]
        
        # Lag features 생성
        print("🕒 Lag features 생성 중...")
        lag_periods = [1, 2, 3, 6, 12, 24, 168]  # 1시간, 2시간, 3시간, 6시간, 12시간, 1일, 1주일 전
        
        for dataset_name, dataset in [('train', self.train_data), ('test', self.test_data)]:
            print(f"  📈 {dataset_name} 데이터 Lag features 생성...")
            
            # 시간순 정렬
            dataset = dataset.sort_values(['station_name', 'datetime']).reset_index(drop=True)
            
            # 각 역별로 lag features 생성
            for lag in lag_periods:
                lag_col_name = f'congestion_lag_{lag}'
                dataset[lag_col_name] = np.nan
                
                for station in dataset['station_name'].unique():
                    station_mask = dataset['station_name'] == station
                    station_data = dataset[station_mask].copy()
                    
                    # lag feature 생성 (역별로)
                    lagged_values = station_data['congestion'].shift(lag)
                    dataset.loc[station_mask, lag_col_name] = lagged_values
            
            # 롤링 윈도우 통계 특성 추가
            rolling_windows = [3, 6, 12, 24]  # 3시간, 6시간, 12시간, 24시간 롤링
            for window in rolling_windows:
                for stat in ['mean', 'std', 'min', 'max']:
                    col_name = f'congestion_rolling_{window}h_{stat}'
                    dataset[col_name] = np.nan
                    
                    for station in dataset['station_name'].unique():
                        station_mask = dataset['station_name'] == station
                        station_data = dataset[station_mask].copy()
                        
                        if stat == 'mean':
                            rolling_values = station_data['congestion'].rolling(window=window, min_periods=1).mean().shift(1)
                        elif stat == 'std':
                            rolling_values = station_data['congestion'].rolling(window=window, min_periods=1).std().shift(1)
                        elif stat == 'min':
                            rolling_values = station_data['congestion'].rolling(window=window, min_periods=1).min().shift(1)
                        elif stat == 'max':
                            rolling_values = station_data['congestion'].rolling(window=window, min_periods=1).max().shift(1)
                        
                        dataset.loc[station_mask, col_name] = rolling_values
            
            # 시간대별 평균 혼잡도 특성 (과거 데이터 기반)
            hourly_avg_col = 'hourly_avg_congestion'
            dataset[hourly_avg_col] = np.nan
            
            for station in dataset['station_name'].unique():
                station_mask = dataset['station_name'] == station
                station_data = dataset[station_mask].copy()
                
                # 각 시점의 이전 데이터만 사용하여 시간대별 평균 계산
                for idx in station_data.index:
                    current_hour = station_data.loc[idx, 'hour']
                    current_datetime = station_data.loc[idx, 'datetime']
                    
                    # 현재 시점 이전의 같은 시간대 데이터
                    historical_mask = (station_data['hour'] == current_hour) & (station_data['datetime'] < current_datetime)
                    if historical_mask.sum() > 0:
                        avg_congestion = station_data.loc[historical_mask, 'congestion'].mean()
                        dataset.loc[idx, hourly_avg_col] = avg_congestion
            
            # 업데이트
            if dataset_name == 'train':
                self.train_data = dataset
            else:
                self.test_data = dataset
        
        # 전체 특성 리스트 업데이트
        lag_feature_cols = [f'congestion_lag_{lag}' for lag in lag_periods]
        rolling_feature_cols = [f'congestion_rolling_{window}h_{stat}' 
                               for window in rolling_windows 
                               for stat in ['mean', 'std', 'min', 'max']]
        time_avg_cols = ['hourly_avg_congestion']
        
        feature_cols = base_feature_cols + lag_feature_cols + rolling_feature_cols + time_avg_cols
        
        # 공통 특성만 선택
        available_features = [col for col in feature_cols 
                            if col in self.train_data.columns and col in self.test_data.columns]
        
        # 역 인코딩 (훈련 데이터 기준)
        le_station = LabelEncoder()
        train_stations = self.train_data['station_name'].unique()
        test_stations = self.test_data['station_name'].unique()
        
        # 공통 역만 사용
        common_stations = set(train_stations) & set(test_stations)
        print(f"📍 공통 역: {len(common_stations)}개")
        
        # 공통 역만 필터링
        self.train_data = self.train_data[self.train_data['station_name'].isin(common_stations)]
        self.test_data = self.test_data[self.test_data['station_name'].isin(common_stations)]
        
        # 역 인코딩
        le_station.fit(sorted(common_stations))
        self.train_data['station_encoded'] = le_station.transform(self.train_data['station_name'])
        self.test_data['station_encoded'] = le_station.transform(self.test_data['station_name'])
        
        available_features.append('station_encoded')
        
        # 특성 및 타겟 준비
        X_train = self.train_data[available_features]
        y_train = self.train_data['congestion']
        X_test = self.test_data[available_features]
        y_test = self.test_data['congestion']
        
        # Lag features 때문에 생긴 결측값 제거
        print("🧹 결측값 처리 중...")
        
        # 결측값이 너무 많은 초기 데이터 제거 (lag features 때문에)
        max_lag = max(lag_periods)
        
        # 각 역별로 처음 max_lag개 시간의 데이터 제거
        train_valid_indices = []
        test_valid_indices = []
        
        for station in common_stations:
            # 훈련 데이터
            station_train_mask = self.train_data['station_name'] == station
            station_train_data = self.train_data[station_train_mask].sort_values('datetime')
            if len(station_train_data) > max_lag:
                valid_train_indices = station_train_data.index[max_lag:]
                train_valid_indices.extend(valid_train_indices)
            
            # 테스트 데이터
            station_test_mask = self.test_data['station_name'] == station
            station_test_data = self.test_data[station_test_mask].sort_values('datetime')
            if len(station_test_data) > max_lag:
                valid_test_indices = station_test_data.index[max_lag:]
                test_valid_indices.extend(valid_test_indices)
        
        # 유효한 인덱스만 선택
        X_train = X_train.loc[train_valid_indices]
        y_train = y_train.loc[train_valid_indices]
        X_test = X_test.loc[test_valid_indices]
        y_test = y_test.loc[test_valid_indices]
        
        # 나머지 결측값 처리 (interpolation)
        print("🔧 나머지 결측값 보간 처리...")
        X_train = X_train.ffill().bfill()
        X_test = X_test.ffill().bfill()
        
        # 최종 결측값 제거
        final_train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        final_test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train = X_train[final_train_mask]
        y_train = y_train[final_train_mask]
        X_test = X_test[final_test_mask]
        y_test = y_test[final_test_mask]
        
        print(f"✅ 특성 준비 완료:")
        print(f"  - 기본 특성: {len(base_feature_cols)}개")
        print(f"  - Lag 특성: {len(lag_feature_cols)}개")
        print(f"  - 롤링 통계: {len(rolling_feature_cols)}개")
        print(f"  - 시간대별 평균: {len(time_avg_cols)}개")
        print(f"  - 총 특성 수: {len(available_features)}개")
        print(f"  - 훈련 샘플: {len(X_train):,}개")
        print(f"  - 검증 샘플: {len(X_test):,}개")
        
        self.feature_names = available_features
        return X_train, y_train, X_test, y_test
    
    def train_and_validate(self, X_train, y_train, X_test, y_test):
        """모델 훈련 및 검증"""
        print("\n🚀 시계열 검증 시작 (2021→2022)")
        print("-" * 60)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 정의
        models = {
            #'Linear Regression': LinearRegression(),
            #'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method='gpu_hist')

        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  🔄 {name} 훈련 중...")
            
            try:
                # 모델 훈련 (2021 데이터)
                if name == 'Linear Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # 2022 데이터로 검증
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                print(f"    ✅ {name} 완료 - MAE: {mae:.3f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"    ❌ {name} 실패: {str(e)}")
        
        self.models = models
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def analyze_temporal_generalization(self):
        """시간적 일반화 성능 분석"""
        print("\n" + "="*60)
        print("📈 시간적 일반화 성능 분석")
        print("="*60)
        
        # 연도별 데이터 특성 비교
        print(f"\n📊 연도별 데이터 특성 비교:")
        
        train_stats = {
            '평균 혼잡도': self.train_data['congestion'].mean(),
            '혼잡도 표준편차': self.train_data['congestion'].std(),
            '평균 온도': self.train_data['ta'].mean(),
            '평균 습도': self.train_data['hm'].mean(),
        }
        
        test_stats = {
            '평균 혼잡도': self.test_data['congestion'].mean(),
            '혼잡도 표준편차': self.test_data['congestion'].std(),
            '평균 온도': self.test_data['ta'].mean(),
            '평균 습도': self.test_data['hm'].mean(),
        }
        
        print(f"{'특성':<15} {'2021년':<10} {'2022년':<10} {'차이':<10}")
        print("-" * 50)
        for key in train_stats:
            diff = test_stats[key] - train_stats[key]
            print(f"{key:<15} {train_stats[key]:<10.2f} {test_stats[key]:<10.2f} {diff:+7.2f}")
        
        # 모델 성능 비교
        print(f"\n🏆 모델별 시간적 일반화 성능:")
        print(f"{'모델명':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 55)
        
        for name, result in self.results.items():
            print(f"{name:<20} {result['mae']:<10.3f} {result['rmse']:<10.3f} {result['r2']:<10.3f}")
        
        # 최고 성능 모델
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_result = self.results[best_model_name]
        best_model = best_result['model']
        
        print(f"\n🥇 최고 성능 모델: {best_model_name}")
        print(f"  - MAE: {best_result['mae']:.3f}")
        print(f"  - R²: {best_result['r2']:.3f}")
        
        # 특성 중요도 분석 (Random Forest인 경우)
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n🔍 특성 중요도 분석 (상위 15개):")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"{'특성명':<25} {'중요도':<10} {'유형':<15}")
            print("-" * 50)
            
            for idx, row in feature_importance.head(15).iterrows():
                feature_name = row['feature']
                importance = row['importance']
                
                # 특성 유형 분류
                if 'lag' in feature_name:
                    feature_type = 'Lag Feature'
                elif 'rolling' in feature_name:
                    feature_type = 'Rolling Stats'
                elif 'hourly_avg' in feature_name:
                    feature_type = 'Time Average'
                elif feature_name in ['hour', 'dayofweek', 'month', 'is_weekend', 'is_rush_hour']:
                    feature_type = 'Time Feature'
                elif feature_name in ['ta', 'ws', 'rn_hr1', 'hm']:
                    feature_type = 'Weather'
                else:
                    feature_type = 'Other'
                
                print(f"{feature_name:<25} {importance:<10.4f} {feature_type:<15}")
            
            # Lag features 중요도 요약
            lag_features = feature_importance[feature_importance['feature'].str.contains('lag')]
            if len(lag_features) > 0:
                print(f"\n📊 Lag Features 중요도 요약:")
                print(f"  - 총 Lag features: {len(lag_features)}개")
                print(f"  - 평균 중요도: {lag_features['importance'].mean():.4f}")
                print(f"  - 최고 중요도 Lag: {lag_features.iloc[0]['feature']} ({lag_features.iloc[0]['importance']:.4f})")
                
                # 상위 5개 lag features
                print(f"  - 상위 5개 Lag features:")
                for idx, row in lag_features.head(5).iterrows():
                    lag_period = row['feature'].split('_')[-1]
                    print(f"    • {lag_period}시간 전: {row['importance']:.4f}")
        
        return best_model_name, best_result
    
    def plot_validation_results(self):
        """검증 결과 시각화"""
        print("\n📊 검증 결과 시각화 중...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # 1. 연도별 혼잡도 분포 비교
        axes[0, 0].hist(self.train_data['congestion'], bins=50, alpha=0.7, 
                       label='2021년 (훈련)', density=True)
        axes[0, 0].hist(self.test_data['congestion'], bins=50, alpha=0.7, 
                       label='2022년 (검증)', density=True)
        axes[0, 0].set_title('연도별 혼잡도 분포 비교')
        axes[0, 0].set_xlabel('혼잡도')
        axes[0, 0].set_ylabel('밀도')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 모델 성능 비교
        model_names = list(self.results.keys())
        mae_scores = [self.results[name]['mae'] for name in model_names]
        r2_scores = [self.results[name]['r2'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        axes[0, 1].bar(x_pos, mae_scores, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('모델별 MAE (2022년 검증)')
        axes[0, 1].set_xlabel('모델')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        
        # 값 표시
        for i, v in enumerate(mae_scores):
            axes[0, 1].text(i, v + 0.05, f'{v:.3f}', ha='center')
        
        # 3. R² 점수
        axes[0, 2].bar(x_pos, r2_scores, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('모델별 R² (2022년 검증)')
        axes[0, 2].set_xlabel('모델')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(model_names, rotation=45)
        
        for i, v in enumerate(r2_scores):
            axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 4. 최고 성능 모델의 예측 vs 실제
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_predictions = self.results[best_model_name]['predictions']
        best_model = self.results[best_model_name]['model']
        
        # 샘플링 (너무 많으면 시각화가 어려움)
        sample_size = min(5000, len(self.y_test))
        sample_indices = np.random.choice(len(self.y_test), sample_size, replace=False)
        
        axes[1, 0].scatter(self.y_test.iloc[sample_indices], best_predictions[sample_indices], 
                          alpha=0.5, s=1)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'{best_model_name} - 예측 vs 실제')
        axes[1, 0].set_xlabel('실제값 (2022)')
        axes[1, 0].set_ylabel('예측값')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 잔차 분석
        residuals = self.y_test.iloc[sample_indices] - best_predictions[sample_indices]
        axes[1, 1].scatter(best_predictions[sample_indices], residuals, alpha=0.5, s=1)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('잔차 분석')
        axes[1, 1].set_xlabel('예측값')
        axes[1, 1].set_ylabel('잔차')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 시간별 성능 분석
        # 2022년 데이터에 예측값 추가
        test_with_pred = self.test_data.copy()
        test_with_pred['predictions'] = best_predictions
        test_with_pred['residuals'] = abs(test_with_pred['congestion'] - test_with_pred['predictions'])
        
        hourly_mae = test_with_pred.groupby('hour')['residuals'].mean()
        axes[1, 2].plot(hourly_mae.index, hourly_mae.values, marker='o', linewidth=2)
        axes[1, 2].set_title('시간대별 예측 오차 (MAE)')
        axes[1, 2].set_xlabel('시간')
        axes[1, 2].set_ylabel('평균 절대 오차')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xticks(range(0, 24, 2))
        
        # 7. 특성 중요도 (상위 10개)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            y_pos = np.arange(len(feature_importance))
            axes[2, 0].barh(y_pos, feature_importance['importance'], alpha=0.7)
            axes[2, 0].set_yticks(y_pos)
            axes[2, 0].set_yticklabels(feature_importance['feature'], fontsize=8)
            axes[2, 0].set_xlabel('중요도')
            axes[2, 0].set_title('특성 중요도 (상위 10개)')
            axes[2, 0].grid(True, alpha=0.3)
            
            # 8. Lag features별 중요도
            lag_features = feature_importance[feature_importance['feature'].str.contains('lag')]
            if len(lag_features) > 0:
                lag_periods = [int(feat.split('_')[-1]) for feat in lag_features['feature']]
                axes[2, 1].bar(range(len(lag_periods)), lag_features['importance'], alpha=0.7, color='orange')
                axes[2, 1].set_title('Lag Features 중요도')
                axes[2, 1].set_xlabel('Lag Period (시간)')
                axes[2, 1].set_ylabel('중요도')
                axes[2, 1].set_xticks(range(len(lag_periods)))
                axes[2, 1].set_xticklabels([f'{p}h' for p in lag_periods], rotation=45)
                axes[2, 1].grid(True, alpha=0.3)
            else:
                axes[2, 1].text(0.5, 0.5, 'Lag Features 없음', ha='center', va='center', transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Lag Features 중요도')
        
        # 9. 특성 유형별 중요도 합계
        if hasattr(best_model, 'feature_importances_'):
            feature_importance_full = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            })
            
            # 특성 유형별 분류
            feature_types = []
            for feature in feature_importance_full['feature']:
                if 'lag' in feature:
                    feature_types.append('Lag Features')
                elif 'rolling' in feature:
                    feature_types.append('Rolling Stats')
                elif 'hourly_avg' in feature:
                    feature_types.append('Time Average')
                elif feature in ['hour', 'dayofweek', 'month', 'is_weekend', 'is_rush_hour', 'season']:
                    feature_types.append('Time Features')
                elif feature in ['ta', 'ws', 'rn_hr1', 'hm']:
                    feature_types.append('Weather')
                elif 'sin' in feature or 'cos' in feature:
                    feature_types.append('Cyclic Features')
                else:
                    feature_types.append('Other')
            
            feature_importance_full['type'] = feature_types
            type_importance = feature_importance_full.groupby('type')['importance'].sum().sort_values(ascending=False)
            
            axes[2, 2].pie(type_importance.values, labels=type_importance.index, autopct='%1.1f%%', startangle=90)
            axes[2, 2].set_title('특성 유형별 중요도 비율')
        
        plt.tight_layout()
        plt.savefig('../result/time_series_validation_with_lag.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self):
        """시계열 검증 인사이트"""
        print("\n" + "="*60)
        print("🔍 시계열 검증 인사이트")
        print("="*60)
        
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_result = self.results[best_model_name]
        best_model = best_result['model']
        
        print(f"\n🎯 시간적 일반화 성능:")
        print(f"  - 최고 모델: {best_model_name}")
        print(f"  - 2022년 예측 MAE: {best_result['mae']:.3f}")
        print(f"  - 2022년 예측 R²: {best_result['r2']:.3f}")
        
        # 성능 평가
        if best_result['r2'] > 0.8:
            performance = "우수"
        elif best_result['r2'] > 0.6:
            performance = "양호"
        else:
            performance = "개선 필요"
        
        print(f"\n📊 성능 평가: {performance}")
        
        # Lag features 분석
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            })
            
            # Lag features 중요도 분석
            lag_features = feature_importance[feature_importance['feature'].str.contains('lag')]
            rolling_features = feature_importance[feature_importance['feature'].str.contains('rolling')]
            time_avg_features = feature_importance[feature_importance['feature'].str.contains('hourly_avg')]
            
            total_temporal_importance = (lag_features['importance'].sum() + 
                                       rolling_features['importance'].sum() + 
                                       time_avg_features['importance'].sum())
            
            print(f"\n🕒 시간적 특성 분석:")
            print(f"  - Lag Features 개수: {len(lag_features)}개")
            print(f"  - Lag Features 중요도 합계: {lag_features['importance'].sum():.3f}")
            print(f"  - Rolling Stats 중요도 합계: {rolling_features['importance'].sum():.3f}")
            print(f"  - 시간대별 평균 중요도: {time_avg_features['importance'].sum():.3f}")
            print(f"  - 전체 시간적 특성 중요도: {total_temporal_importance:.3f}")
            
            if len(lag_features) > 0:
                # 가장 중요한 lag period 분석
                top_lag = lag_features.iloc[0]
                lag_period = int(top_lag['feature'].split('_')[-1])
                
                if lag_period == 1:
                    lag_desc = "1시간 전 (직전 시간)"
                elif lag_period == 24:
                    lag_desc = "24시간 전 (전일 동시간)"
                elif lag_period == 168:
                    lag_desc = "168시간 전 (전주 동시간)"
                else:
                    lag_desc = f"{lag_period}시간 전"
                
                print(f"  - 최고 중요도 Lag: {lag_desc} (중요도: {top_lag['importance']:.4f})")
                
                # 단기 vs 장기 lag 비교
                short_term_lags = lag_features[lag_features['feature'].str.contains(r'lag_[1-6]$', regex=True)]
                long_term_lags = lag_features[lag_features['feature'].str.contains(r'lag_(24|168)$', regex=True)]
                
                if len(short_term_lags) > 0 and len(long_term_lags) > 0:
                    short_importance = short_term_lags['importance'].sum()
                    long_importance = long_term_lags['importance'].sum()
                    
                    print(f"  - 단기 Lag (1-6시간): {short_importance:.4f}")
                    print(f"  - 장기 Lag (24, 168시간): {long_importance:.4f}")
                    
                    if short_importance > long_importance:
                        temporal_pattern = "단기 패턴 중심"
                    else:
                        temporal_pattern = "장기 패턴 중심"
                    print(f"  - 주요 시간적 패턴: {temporal_pattern}")
        
        print(f"\n💡 시계열 모델링 권장사항:")
        if best_result['r2'] > 0.7:
            print(f"  ✅ 모델이 시간적 패턴을 잘 학습했습니다")
            print(f"  ✅ 2022년 데이터에 대한 일반화 성능 우수")
            
            if hasattr(best_model, 'feature_importances_'):
                if total_temporal_importance > 0.3:
                    print(f"  🕒 Lag features가 효과적으로 작동하고 있습니다")
                    print(f"  📈 추가 개선: 더 많은 lag periods, 계절성 lag 추가")
                else:
                    print(f"  ⚠️ Lag features의 활용도가 낮습니다")
                    print(f"  🔧 개선 방향: lag period 조정, 더 긴 시계열 데이터 활용")
            
            print(f"  🚀 추가 최적화: 하이퍼파라미터 튜닝, 앙상블")
        else:
            print(f"  ⚠️ 시간적 일반화 성능 개선 필요")
            print(f"  📈 추천 방법:")
            print(f"    - 더 다양한 lag periods 실험")
            print(f"    - 계절성 decomposition 활용")
            print(f"    - 외부 데이터 (공휴일, 이벤트) 추가")
            print(f"    - LSTM/GRU 등 순환 신경망 모델 시도")
            
            if hasattr(best_model, 'feature_importances_') and len(lag_features) > 0:
                print(f"    - Lag features 최적화 ({len(lag_features)}개 현재 사용 중)")
        
        print(f"\n🔮 모델 활용 방향:")
        print(f"  - 실시간 혼잡도 예측 시스템 구축")
        print(f"  - 시간대별 혼잡도 패턴 분석")
        print(f"  - 교통 정책 수립 지원 도구")
        if best_result['r2'] > 0.7:
            print(f"  - 운영 시스템 도입 준비 완료")
        else:
            print(f"  - 추가 개선 후 운영 시스템 도입 권장")

def main():
    """메인 실행 함수"""
    print("🚀 시계열 검증 시작: 2021년 훈련 → 2022년 검증")
    print("=" * 60)
    
    # 검증기 초기화
    validator = TimeSeriesValidator()
    
    # 1. 데이터 로드
    if not validator.load_data(train_years=['21'], test_year='23', sample_size=5000000):
        return None
    
    # 2. 특성 준비
    X_train, y_train, X_test, y_test = validator.prepare_features()
    
    # 3. 모델 훈련 및 검증
    results = validator.train_and_validate(X_train, y_train, X_test, y_test)
    
    # 4. 시간적 일반화 성능 분석
    validator.analyze_temporal_generalization()
    
    # 5. 결과 시각화
    validator.plot_validation_results()
    
    # 6. 인사이트 생성
    validator.generate_insights()
    
    print(f"\n🎉 시계열 검증 완료!")
    print(f"결과 이미지: time_series_validation_with_lag.png")
    
    return validator

if __name__ == "__main__":
    validator = main() 
