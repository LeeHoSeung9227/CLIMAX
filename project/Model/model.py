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
        
    def load_data(self, train_years=['21', '22'], test_year='23', sample_size=5000000):
        """복수 학습 데이터(21,22년)와 테스트 데이터(23년) 로드"""
        print(f"🚀 시계열 검증 데이터 로드")
        print(f"  훈련: {', '.join('20'+y for y in train_years)}년 데이터")
        print(f"  검증: 20{test_year}년 데이터")
        print("="*60)

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(base_dir, '..', '데이터'))
            
            # 여러 학습 데이터 불러와서 concat
            train_dfs = []
            for y in train_years:
                train_file = os.path.join(data_dir, f'train_subway{y}.csv')
                print(f"20{y}년 훈련 데이터 로드 중...")
                df = pd.read_csv(train_file, encoding='cp949', nrows=sample_size)
                df = self._preprocess_data(df, y)
                train_dfs.append(df)
            self.train_data = pd.concat(train_dfs).reset_index(drop=True)
            
            # 테스트 데이터 로드
            test_file = os.path.join(data_dir, f'train_subway{test_year}.csv')
            print(f"📊 20{test_year}년 검증 데이터 로드 중...")
            test_df = pd.read_csv(test_file, encoding='cp949', nrows=sample_size)
            self.test_data = self._preprocess_data(test_df, test_year)

            print(f"데이터 로드 완료!")
            print(f"  훈련 데이터: {len(self.train_data):,}개 ({', '.join('20'+y for y in train_years)}년)")
            print(f"  검증 데이터: {len(self.test_data):,}개 (20{test_year}년)")
            return True
        
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
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
        print("\n특성 준비 중...")
        
        # 특성 선택
        feature_cols = [
            'hour', 'dayofweek', 'month', 'day', 'season',
            'is_weekend', 'is_rush_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos',
            'ta', 'ws', 'rn_hr1', 'hm'
        ]
        
        # 공통 특성만 선택
        available_features = [col for col in feature_cols 
                            if col in self.train_data.columns and col in self.test_data.columns]
        
        # 역 인코딩 (훈련 데이터 기준)
        le_station = LabelEncoder()
        train_stations = self.train_data['station_name'].unique()
        test_stations = self.test_data['station_name'].unique()
        
        # 공통 역만 사용
        common_stations = set(train_stations) & set(test_stations)
        print(f"공통 역: {len(common_stations)}개 - {', '.join(sorted(common_stations))}")
        
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
        
        # 결측값 제거
        train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        print(f"특성 준비 완료:")
        print(f"  - 특성 수: {len(available_features)}")
        print(f"  - 훈련 샘플: {len(X_train):,}개")
        print(f"  - 검증 샘플: {len(X_test):,}개")
        
        self.feature_names = available_features
        return X_train, y_train, X_test, y_test
    
    def train_and_validate(self, X_train, y_train, X_test, y_test):
        """모델 훈련 및 검증"""
        print("\n시계열 검증 시작 (2021→2022)")
        print("-" * 60)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 정의
        models = {
            #'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            #'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method='gpu_hist')

        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  {name} 훈련 중...")
            
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
                print(f"   {name} 실패: {str(e)}")
        
        self.models = models
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def analyze_temporal_generalization(self):
        """시간적 일반화 성능 분석"""
        print("\n" + "="*60)
        print("시간적 일반화 성능 분석")
        print("="*60)
        
        # 연도별 데이터 특성 비교
        print(f"\n연도별 데이터 특성 비교:")
        
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
        print(f"\n모델별 시간적 일반화 성능:")
        print(f"{'모델명':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 55)
        
        for name, result in self.results.items():
            print(f"{name:<20} {result['mae']:<10.3f} {result['rmse']:<10.3f} {result['r2']:<10.3f}")
        
        # 최고 성능 모델
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_result = self.results[best_model_name]
        
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"  - MAE: {best_result['mae']:.3f}")
        print(f"  - R²: {best_result['r2']:.3f}")
        
        return best_model_name, best_result
    
    def plot_validation_results(self):
        """검증 결과 시각화"""
        print("\n📊 검증 결과 시각화 중...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        plt.tight_layout()
        plt.savefig('../result/time_series_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self):
        """시계열 검증 인사이트"""
        print("\n" + "="*60)
        print("시계열 검증 인사이트")
        print("="*60)
        
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_result = self.results[best_model_name]
        
        print(f"\n시간적 일반화 성능:")
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
        
        print(f"\n성능 평가: {performance}")
        
        print(f"\n시계열 모델링 권장사항:")
        if best_result['r2'] > 0.7:
            print(f"  모델이 시간적 패턴을 잘 학습했습니다")
            print(f"  2022년 데이터에 대한 일반화 성능 우수")
            print(f"  추가 개선: 하이퍼파라미터 튜닝, 앙상블")
        else:
            print(f"  시간적 일반화 성능 개선 필요")
            print(f"  추천 방법: Lag features, 계절성 강화, 외부 데이터 추가")

def main():
    """메인 실행 함수"""
    print("시계열 검증 시작: 2021년 훈련 → 2022년 검증")
    print("=" * 60)
    
    # 검증기 초기화
    validator = TimeSeriesValidator()
    
    # 1. 데이터 로드
    if not validator.load_data(train_years=['21', '22'], test_year='23', sample_size=5000000):
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
    
    print(f"\n시계열 검증 완려")
    print(f"결과 이미지: time_series_validation.png")
    
    return validator

if __name__ == "__main__":
    validator = main() 
