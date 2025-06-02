#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Subway Congestion Prediction Model
고도화된 지하철 혼잡도 예측 모델

개선사항:
1. 하이퍼파라미터 튜닝 (Optuna)
2. 극한 기상 상태 이진 변수
3. 세분화된 시간대/계절 범주형 변수
4. 상호작용 항 추가
5. 특성 중요도 기반 선택
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class EnhancedFeatureEngineering:
    """고도화된 특성 공학"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.selected_features = None
        
    def create_extreme_weather_features(self, df):
        """극한 기상 상태 이진 변수 생성"""
        print("🌪️ 극한 기상 상태 변수 생성 중...")
        
        # 온도 기반 극한 상태
        if 'ta' in df.columns:
            temp_q05 = df['ta'].quantile(0.05)  # 한파
            temp_q95 = df['ta'].quantile(0.95)  # 폭염
            
            df['extreme_cold'] = (df['ta'] <= temp_q05).astype(int)
            df['extreme_heat'] = (df['ta'] >= temp_q95).astype(int)
            df['moderate_temp'] = ((df['ta'] > temp_q05) & (df['ta'] < temp_q95)).astype(int)
            
            print(f"  한파 기준: ≤{temp_q05:.1f}°C ({df['extreme_cold'].sum():,}개)")
            print(f"  폭염 기준: ≥{temp_q95:.1f}°C ({df['extreme_heat'].sum():,}개)")
        
        # 강수 기반 극한 상태
        if 'rn_hr1' in df.columns:
            df['no_rain'] = (df['rn_hr1'] == 0).astype(int)
            df['light_rain'] = ((df['rn_hr1'] > 0) & (df['rn_hr1'] <= 2)).astype(int)
            df['heavy_rain'] = (df['rn_hr1'] > 10).astype(int)
            df['extreme_rain'] = (df['rn_hr1'] > 30).astype(int)
            
            print(f"  폭우 기준: >10mm ({df['heavy_rain'].sum():,}개)")
            print(f"  극한 강수: >30mm ({df['extreme_rain'].sum():,}개)")
        
        # 풍속 기반 극한 상태
        if 'ws' in df.columns:
            wind_q90 = df['ws'].quantile(0.90)
            wind_q95 = df['ws'].quantile(0.95)
            
            df['calm_wind'] = (df['ws'] <= 1.0).astype(int)
            df['strong_wind'] = (df['ws'] >= wind_q90).astype(int)
            df['extreme_wind'] = (df['ws'] >= wind_q95).astype(int)
            
            print(f"  강풍 기준: ≥{wind_q90:.1f}m/s ({df['strong_wind'].sum():,}개)")
            print(f"  극한 풍속: ≥{wind_q95:.1f}m/s ({df['extreme_wind'].sum():,}개)")
        
        # 습도 기반 극한 상태
        if 'hm' in df.columns:
            humidity_q05 = df['hm'].quantile(0.05)
            humidity_q95 = df['hm'].quantile(0.95)
            
            df['extreme_dry'] = (df['hm'] <= humidity_q05).astype(int)
            df['extreme_humid'] = (df['hm'] >= humidity_q95).astype(int)
            
            print(f"  극건조: ≤{humidity_q05:.1f}% ({df['extreme_dry'].sum():,}개)")
            print(f"  극습함: ≥{humidity_q95:.1f}% ({df['extreme_humid'].sum():,}개)")
        
        # 복합 극한 상태
        df['extreme_weather_any'] = (
            df.get('extreme_cold', 0) | df.get('extreme_heat', 0) |
            df.get('heavy_rain', 0) | df.get('extreme_wind', 0) |
            df.get('extreme_dry', 0) | df.get('extreme_humid', 0)
        ).astype(int)
        
        print(f"  전체 극한 기상: {df['extreme_weather_any'].sum():,}개 ({df['extreme_weather_any'].mean()*100:.1f}%)")
        
        return df
    
    def create_detailed_time_features(self, df):
        """세분화된 시간대/계절 범주형 변수"""
        print("⏰ 세분화된 시간 특성 생성 중...")
        
        # 시간 관련 기본 특성
        df['datetime'] = pd.to_datetime(df['tm'], format='%Y%m%d%H')
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        
        # 세분화된 출퇴근 시간대
        def get_detailed_time_period(hour):
            if hour in [6, 7]:
                return 'early_morning_rush'
            elif hour in [8, 9]:
                return 'morning_rush_peak'
            elif hour == 10:
                return 'morning_rush_end'
            elif hour in [11, 12, 13, 14]:
                return 'daytime'
            elif hour in [15, 16]:
                return 'afternoon_start'
            elif hour in [17, 18]:
                return 'evening_rush_start'
            elif hour in [19, 20]:
                return 'evening_rush_peak'
            elif hour == 21:
                return 'evening_rush_end'
            elif hour in [22, 23]:
                return 'night'
            else:  # 0-5시
                return 'late_night'
        
        df['time_period'] = df['hour'].apply(get_detailed_time_period)
        
        # 세분화된 계절
        def get_detailed_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4]:
                return 'spring_early'
            elif month == 5:
                return 'spring_late'
            elif month in [6, 7]:
                return 'summer_early'
            elif month == 8:
                return 'summer_peak'
            elif month in [9, 10]:
                return 'autumn_early'
            else:  # 11월
                return 'autumn_late'
        
        df['detailed_season'] = df['month'].apply(get_detailed_season)
        
        # 주말/평일 세분화
        df['day_type'] = df['dayofweek'].apply(
            lambda x: 'weekend' if x >= 5 else 'weekday'
        )
        
        # 월요일/금요일 효과
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        
        # 순환적 시간 특성 (기존 + 추가)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        print(f"  세분화된 시간대: {df['time_period'].nunique()}개 카테고리")
        print(f"  세분화된 계절: {df['detailed_season'].nunique()}개 카테고리")
        
        return df
    
    def create_interaction_features(self, df):
        """상호작용 항 생성"""
        print("🔗 상호작용 특성 생성 중...")
        
        # 기상 변수 간 상호작용
        if 'ta' in df.columns and 'hm' in df.columns:
            # 체감온도 (온도 × 습도)
            df['apparent_temp'] = df['ta'] * (1 + df['hm'] / 100)
            df['temp_humidity_interaction'] = df['ta'] * df['hm']
            print("  온도 × 습도 상호작용 생성")
        
        if 'ta' in df.columns and 'ws' in df.columns:
            # 풍속에 의한 체감온도
            df['wind_chill'] = df['ta'] - df['ws'] * 2
            df['temp_wind_interaction'] = df['ta'] * df['ws']
            print("  온도 × 풍속 상호작용 생성")
        
        if 'rn_hr1' in df.columns and 'ws' in df.columns:
            # 비바람 효과
            df['rain_wind_interaction'] = df['rn_hr1'] * df['ws']
            print("  강수 × 풍속 상호작용 생성")
        
        if 'rn_hr1' in df.columns and 'hm' in df.columns:
            # 습도-강수 상호작용
            df['rain_humidity_interaction'] = df['rn_hr1'] * df['hm']
            print("  강수 × 습도 상호작용 생성")
        
        # 시간-기상 상호작용
        if 'ta' in df.columns:
            df['temp_hour_interaction'] = df['ta'] * df['hour']
            df['temp_season_interaction'] = df['ta'] * df['month']
            print("  온도 × 시간 상호작용 생성")
        
        # 극한 기상과 시간 상호작용
        if 'extreme_weather_any' in df.columns:
            df['extreme_weather_rush'] = df['extreme_weather_any'] * (
                df['time_period'].isin(['morning_rush_peak', 'evening_rush_peak']).astype(int)
            )
            print("  극한기상 × 출퇴근시간 상호작용 생성")
        
        return df
    
    def create_lag_features(self, df, target_col='congestion'):
        """시차 특성 생성"""
        print("📈 기상 변수 시차 특성 생성 중 (혼잡도 시차 제외)...")
        
        # 시간 정렬
        df = df.sort_values(['station_name', 'tm']).reset_index(drop=True)
        
        # 기상 변수 시차 특성만 생성 (1일 전)
        weather_vars = ['ta', 'ws', 'rn_hr1', 'hm']
        for var in weather_vars:
            if var in df.columns:
                df[f'{var}_lag_24'] = df[var].shift(24)
        
        # 기상 변수 이동평균 특성 (3,6,12시간)
        for var in weather_vars:
            if var in df.columns:
                for window in [3, 6, 12]:
                    df[f'{var}_ma_{window}'] = df[var].rolling(
                        window=window, min_periods=1
                    ).mean()
        
        print(f"  기상 시차 특성: {len(weather_vars)}개")
        print(f"  기상 이동평균 특성: {len(weather_vars) * 3}개")
        print("  ✅ 혼잡도 과거 데이터는 제외됨")
        
        return df

class EnhancedSubwayModel:
    """고도화된 지하철 혼잡도 예측 모델"""
    
    def __init__(self, use_optuna=True, n_trials=100):
        self.feature_engineer = EnhancedFeatureEngineering()
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.results = {}
        
    def load_and_preprocess_data(self, train_years=['21'], test_year='23', sample_size=5000000):
        """데이터 로드 및 전처리"""
        print("🚀 고도화된 데이터 전처리 시작")
        print("=" * 60)
        
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(base_dir, '..', '데이터'))
            
            # 훈련 데이터 로드
            train_dfs = []
            for year in train_years:
                file_path = os.path.join(data_dir, f'train_subway{year}.csv')
                print(f"20{year}년 데이터 로드 중...")
                df = pd.read_csv(file_path, encoding='cp949', nrows=sample_size)
                df.columns = [col.replace(f'train_subway{year}.', '') for col in df.columns]
                train_dfs.append(df)
            
            self.train_data = pd.concat(train_dfs).reset_index(drop=True)
            
            # 테스트 데이터 로드
            test_file = os.path.join(data_dir, f'train_subway{test_year}.csv')
            print(f"20{test_year}년 검증 데이터 로드 중...")
            self.test_data = pd.read_csv(test_file, encoding='cp949', nrows=sample_size)
            self.test_data.columns = [col.replace(f'train_subway{test_year}.', '') for col in self.test_data.columns]
            
            # 특성 공학 적용
            print("\n🔧 고도화된 특성 공학 적용 중...")
            self.train_data = self._apply_feature_engineering(self.train_data)
            self.test_data = self._apply_feature_engineering(self.test_data)
            
            print(f"\n✅ 데이터 준비 완료!")
            print(f"  훈련 데이터: {len(self.train_data):,}개")
            print(f"  검증 데이터: {len(self.test_data):,}개")
            print(f"  특성 수: {self.train_data.shape[1]}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def _apply_feature_engineering(self, df):
        """특성 공학 파이프라인 적용"""
        print("📊 결측치 처리 전 상태:")
        print(f"  전체 결측치: {df.isnull().sum().sum():,}개")
        
        # 0. 기상 데이터 특수 결측치 값 처리
        print("\n🔧 기상 데이터 특수 결측치 값 처리...")
        weather_vars = ['ta', 'ws', 'rn_hr1', 'hm','si']
        special_missing_values = [-99, -9999, 999, -999, 9999, -88, -77]
        
        for var in weather_vars:
            if var in df.columns:
                original_missing = df[var].isnull().sum()
                
                # 특수 결측치 값들을 NaN으로 변환
                for missing_val in special_missing_values:
                    special_count = (df[var] == missing_val).sum()
                    if special_count > 0:
                        print(f"  {var}: {special_count}개의 {missing_val} 값을 NaN으로 변환")
                        df[var] = df[var].replace(missing_val, np.nan)
                
                new_missing = df[var].isnull().sum()
                if new_missing != original_missing:
                    print(f"  {var}: 결측치 {original_missing} → {new_missing}개")
        
        # 비상식적인 값들도 체크 (온도가 -50도 이하나 60도 이상 등)
        if 'ta' in df.columns:
            extreme_temp = ((df['ta'] < -50) | (df['ta'] > 60)) & df['ta'].notna()
            if extreme_temp.sum() > 0:
                print(f"  ta: {extreme_temp.sum()}개의 극한 온도값을 NaN으로 변환")
                df.loc[extreme_temp, 'ta'] = np.nan
        
        if 'hm' in df.columns:
            extreme_hum = ((df['hm'] < 0) | (df['hm'] > 100)) & df['hm'].notna()
            if extreme_hum.sum() > 0:
                print(f"  hm: {extreme_hum.sum()}개의 극한 습도값을 NaN으로 변환")
                df.loc[extreme_hum, 'hm'] = np.nan
        
        if 'ws' in df.columns:
            extreme_wind = (df['ws'] < 0) & df['ws'].notna()
            if extreme_wind.sum() > 0:
                print(f"  ws: {extreme_wind.sum()}개의 음수 풍속값을 NaN으로 변환")
                df.loc[extreme_wind, 'ws'] = np.nan
        
        if 'rn_hr1' in df.columns:
            extreme_rain = (df['rn_hr1'] < 0) & df['rn_hr1'].notna()
            if extreme_rain.sum() > 0:
                print(f"  rn_hr1: {extreme_rain.sum()}개의 음수 강수량을 NaN으로 변환")
                df.loc[extreme_rain, 'rn_hr1'] = np.nan
        
        print(f"특수값 처리 후 총 결측치: {df.isnull().sum().sum():,}개")
        
        # 1. 극한 기상 특성
        df = self.feature_engineer.create_extreme_weather_features(df)
        
        # 2. 세분화된 시간 특성
        df = self.feature_engineer.create_detailed_time_features(df)
        
        # 3. 상호작용 특성
        df = self.feature_engineer.create_interaction_features(df)
        
        # 4. 시차 특성 (기상 변수만, 혼잡도 시차는 제외)
        if 'congestion' in df.columns:
            print("📈 기상 변수 시차 특성 생성 중 (혼잡도 시차 제외)...")
            
            # 시간 정렬
            df = df.sort_values(['station_name', 'tm']).reset_index(drop=True)
            
            # 기상 변수 시차 특성만 생성 (1일 전)
            weather_vars = ['ta', 'ws', 'rn_hr1', 'hm']
            for var in weather_vars:
                if var in df.columns:
                    df[f'{var}_lag_24'] = df[var].shift(24)
            
            # 기상 변수 이동평균 특성 (3,6,12시간)
            for var in weather_vars:
                if var in df.columns:
                    for window in [3, 6, 12]:
                        df[f'{var}_ma_{window}'] = df[var].rolling(
                            window=window, min_periods=1
                        ).mean()
            
            print(f"  기상 시차 특성: {len(weather_vars)}개")
            print(f"  기상 이동평균 특성: {len(weather_vars) * 3}개")
            print("  ✅ 혼잡도 과거 데이터는 제외됨")
        
        # 5. 체계적인 결측치 처리
        print("\n🔧 체계적인 결측치 처리 시작...")
        
        # 5-1. 기상 변수 결측치 처리 (시계열 특성 고려)
        if 'datetime' in df.columns:
            df = df.sort_values(['station_name', 'datetime']).reset_index(drop=True)
            
            for var in weather_vars:
                if var in df.columns:
                    missing_before = df[var].isnull().sum()
                    if missing_before > 0:
                        # Forward fill -> Backward fill -> Median
                        df[var] = df.groupby('station_name')[var].fillna(method='ffill').fillna(method='bfill')
                        df[var] = df[var].fillna(df[var].median())
                        missing_after = df[var].isnull().sum()
                        print(f"  {var}: {missing_before} → {missing_after} 결측치 처리")
        
        # 5-2. 시차 특성 결측치 처리 (시계열 특성 특별 처리)
        lag_cols = [col for col in df.columns if 'lag_' in col or '_ma_' in col]
        for col in lag_cols:
            missing_before = df[col].isnull().sum()
            if missing_before > 0:
                # 시차 특성은 0으로 채우거나 기본값 사용
                if 'congestion' in col:
                    # 혼잡도 시차는 해당 역의 평균값으로
                    df[col] = df.groupby('station_name')[col].transform(
                        lambda x: x.fillna(x.mean()) if x.notna().any() else x.fillna(50)
                    )
                else:
                    # 기상 시차는 원본 변수 값으로
                    base_var = col.split('_lag_')[0] if '_lag_' in col else col.split('_ma_')[0]
                    if base_var in df.columns:
                        df[col] = df[col].fillna(df[base_var])
                    else:
                        df[col] = df[col].fillna(0)
                
                missing_after = df[col].isnull().sum()
                if missing_before > 0:
                    print(f"  {col}: {missing_before} → {missing_after} 시차 결측치 처리")
        
        # 5-3. 범주형 변수 결측치 처리
        categorical_vars = ['time_period', 'detailed_season', 'day_type']
        for var in categorical_vars:
            if var in df.columns:
                missing_before = df[var].isnull().sum()
                if missing_before > 0:
                    # 최빈값으로 채우기
                    mode_value = df[var].mode()
                    if len(mode_value) > 0:
                        df[var] = df[var].fillna(mode_value[0])
                    missing_after = df[var].isnull().sum()
                    print(f"  {var}: {missing_before} → {missing_after} 범주형 결측치 처리")
        
        # 5-4. 이진 변수 결측치 처리 (극한 기상 등)
        binary_vars = [col for col in df.columns if col.startswith(('extreme_', 'is_', 'no_', 'light_', 'heavy_', 'strong_', 'calm_'))]
        for var in binary_vars:
            missing_before = df[var].isnull().sum()
            if missing_before > 0:
                df[var] = df[var].fillna(0)  # 이진 변수는 0으로
                missing_after = df[var].isnull().sum()
                if missing_before > 0:
                    print(f"  {var}: {missing_before} → {missing_after} 이진 결측치 처리")
        
        # 5-5. 상호작용 특성 결측치 처리
        interaction_vars = [col for col in df.columns if 'interaction' in col or 'apparent_temp' in col or 'wind_chill' in col]
        for var in interaction_vars:
            missing_before = df[var].isnull().sum()
            if missing_before > 0:
                df[var] = df[var].fillna(df[var].median())
                missing_after = df[var].isnull().sum()
                if missing_before > 0:
                    print(f"  {var}: {missing_before} → {missing_after} 상호작용 결측치 처리")
        
        # 5-6. 숫자형 변수 최종 처리 (median)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                missing_before = df[col].isnull().sum()
                df[col] = df[col].fillna(df[col].median())
                missing_after = df[col].isnull().sum()
                if missing_before > 0:
                    print(f"  {col}: {missing_before} → {missing_after} 기타 숫자형 결측치 처리")
        
        # 5-7. 최종 결측치 확인
        final_missing = df.isnull().sum().sum()
        print(f"\n✅ 결측치 처리 완료: {final_missing}개 남음")
        
        if final_missing > 0:
            print("⚠️ 남은 결측치가 있는 컬럼:")
            missing_cols = df.columns[df.isnull().any()].tolist()
            for col in missing_cols:
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df) * 100
                print(f"  {col}: {missing_count}개 ({missing_pct:.1f}%)")
        
        return df
    
    def prepare_features(self):
        """최종 특성 준비"""
        print("\n🎯 최종 특성 준비 중...")
        
        # 공통 역 필터링
        train_stations = set(self.train_data['station_name'].unique())
        test_stations = set(self.test_data['station_name'].unique())
        common_stations = train_stations & test_stations
        
        print(f"공통 역: {len(common_stations)}개")
        
        self.train_data = self.train_data[self.train_data['station_name'].isin(common_stations)]
        self.test_data = self.test_data[self.test_data['station_name'].isin(common_stations)]
        
        # 역 인코딩
        le_station = LabelEncoder()
        le_station.fit(sorted(common_stations))
        self.train_data['station_encoded'] = le_station.transform(self.train_data['station_name'])
        self.test_data['station_encoded'] = le_station.transform(self.test_data['station_name'])
        
        # 범주형 변수 인코딩
        categorical_cols = ['time_period', 'detailed_season', 'day_type']
        for col in categorical_cols:
            if col in self.train_data.columns:
                le = LabelEncoder()
                # 훈련 데이터와 테스트 데이터의 모든 값으로 fit
                combined_values = pd.concat([self.train_data[col], self.test_data[col]]).unique()
                le.fit(combined_values)
                self.train_data[f'{col}_encoded'] = le.transform(self.train_data[col])
                self.test_data[f'{col}_encoded'] = le.transform(self.test_data[col])
        
        # 명시적으로 제외할 컬럼들 정의
        exclude_cols = [
            'tm', 'datetime', 'station_name', 'congestion',
            'time_period', 'detailed_season', 'day_type'  # 인코딩된 버전을 사용하므로 원본 제외
        ]
        
        # 숫자형 컬럼만 선택 (더 안전한 방법)
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 특성 컬럼 선택: 숫자형이면서 제외 목록에 없고, 테스트 데이터에도 있는 컬럼들
        feature_cols = [col for col in numeric_cols 
                       if col not in exclude_cols and col in self.test_data.columns]
        
        print(f"전체 숫자형 컬럼: {len(numeric_cols)}개")
        print(f"제외된 컬럼: {len([col for col in numeric_cols if col in exclude_cols])}개")
        
        # 결측치가 많은 특성 제거
        missing_threshold = 0.5
        features_to_remove = []
        for col in feature_cols.copy():
            train_missing = self.train_data[col].isnull().mean()
            test_missing = self.test_data[col].isnull().mean()
            if train_missing > missing_threshold or test_missing > missing_threshold:
                features_to_remove.append(col)
                feature_cols.remove(col)
                print(f"제거: {col} (훈련 결측치 {train_missing:.1%}, 테스트 결측치 {test_missing:.1%})")
        
        # 데이터 타입 확인 및 안전성 검증
        print(f"\n데이터 타입 검증:")
        for col in feature_cols[:5]:  # 처음 5개만 확인
            train_dtype = self.train_data[col].dtype
            test_dtype = self.test_data[col].dtype
            print(f"  {col}: 훈련={train_dtype}, 테스트={test_dtype}")
        
        # 최종 특성 데이터 생성
        X_train = self.train_data[feature_cols].copy()
        y_train = self.train_data['congestion'].copy()
        X_test = self.test_data[feature_cols].copy()
        y_test = self.test_data['congestion'].copy()
        
        # 문자열이 섞여있는지 최종 확인
        for col in feature_cols:
            if X_train[col].dtype == 'object':
                print(f"⚠️ 경고: {col}이 문자열 타입입니다. 샘플: {X_train[col].head().tolist()}")
                # 문자열 컬럼이면 제거
                feature_cols.remove(col)
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        
        # LightGBM 호환성을 위한 컬럼명 정리
        print(f"\n🔧 LightGBM 호환성을 위한 컬럼명 정리 중...")
        def clean_feature_name(name):
            """특수문자를 안전한 문자로 치환"""
            # 특수문자들을 안전한 문자로 치환
            replacements = {
                '%': 'pct',
                '(': '_',
                ')': '_',
                '[': '_',
                ']': '_',
                ':': '_',
                ' ': '_',
                '-': '_',
                '/': '_',
                '.': '_',
                ',': '_',
                '&': 'and',
                '+': 'plus',
                '*': 'mult',
                '=': 'eq',
                '<': 'lt',
                '>': 'gt',
                '!': 'not',
                '?': 'q',
                '@': 'at',
                '#': 'hash',
                '$': 'dollar'
            }
            
            cleaned_name = name
            for old_char, new_char in replacements.items():
                cleaned_name = cleaned_name.replace(old_char, new_char)
            
            # 연속된 언더스코어 정리
            while '__' in cleaned_name:
                cleaned_name = cleaned_name.replace('__', '_')
            
            # 시작과 끝의 언더스코어 제거
            cleaned_name = cleaned_name.strip('_')
            
            return cleaned_name
        
        # 컬럼명 정리 및 변경사항 추적
        original_feature_cols = feature_cols.copy()
        cleaned_feature_cols = [clean_feature_name(col) for col in feature_cols]
        
        # 변경된 컬럼명이 있는지 확인
        changes_made = False
        for original, cleaned in zip(original_feature_cols, cleaned_feature_cols):
            if original != cleaned:
                if not changes_made:
                    print("  컬럼명 변경 사항:")
                    changes_made = True
                print(f"    {original} → {cleaned}")
        
        if not changes_made:
            print("  ✅ 모든 컬럼명이 이미 안전함")
        
        # DataFrame 컬럼명 변경
        column_mapping = dict(zip(original_feature_cols, cleaned_feature_cols))
        X_train = X_train.rename(columns=column_mapping)
        X_test = X_test.rename(columns=column_mapping)
        feature_cols = cleaned_feature_cols
        
        print(f"\n최종 특성 수: {len(feature_cols)}개")
        print(f"특성 종류: 기본시간, 극한기상, 상호작용, 기상시차, 인코딩")
        print(f"훈련 데이터 형태: {X_train.shape}")
        print(f"테스트 데이터 형태: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test, feature_cols
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='xgboost'):
        """Optuna를 사용한 하이퍼파라미터 튜닝"""
        print(f"\n🔍 {model_type} 하이퍼파라미터 튜닝 시작...")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'objective': 'regression',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = lgb.LGBMRegressor(**params)
            
            # 시계열 교차검증
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
            return cv_scores.mean()
        
        # Optuna 스터디
        study = optuna.create_study(direction='maximize', 
                                  sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 CV 점수: {study.best_value:.4f}")
        
        return study.best_params
    
    def feature_selection(self, X_train, y_train, X_test, feature_cols, method='simple'):
        """빠른 특성 선택"""
        print(f"\n🎯 빠른 특성 선택 ({method}) 중...")
        
        if method == 'simple' or len(feature_cols) < 20:
            # 간단한 방법: 분산이 너무 낮은 특성만 제거
            from sklearn.feature_selection import VarianceThreshold
            
            # 분산 임계값으로 특성 선택 (매우 빠름)
            selector = VarianceThreshold(threshold=0.01)
            X_train_transformed = selector.fit_transform(X_train)
            X_test_transformed = selector.transform(X_test)
            
            selected_features = [feature_cols[i] for i, selected in enumerate(selector.get_support()) if selected]
            
            print(f"분산 기반 선택: {len(selected_features)}개 (전체 {len(feature_cols)}개 중)")
            
            X_train_selected = pd.DataFrame(X_train_transformed, columns=selected_features, index=X_train.index)
            X_test_selected = pd.DataFrame(X_test_transformed, columns=selected_features, index=X_test.index)
            
        else:
            # 기존 중요도 기반 방법 (더 느림)
            rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)  # 더 빠르게
            rf.fit(X_train, y_train)
            
            importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            threshold = 0.005  # 임계값 완화
            selected_features = feature_importance_df[
                feature_importance_df['importance'] >= threshold
            ]['feature'].tolist()
            
            print(f"중요도 기반 선택: {len(selected_features)}개")
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
        
        self.selected_features = selected_features
        return X_train_selected, X_test_selected, selected_features
    
    def train_enhanced_models(self, X_train, y_train, X_test, y_test):
        """고도화된 모델들 훈련 (XGBoost, LightGBM만)"""
        print("\n🚀 고도화된 모델 훈련 시작")
        print("=" * 50)
        
        models_to_train = {
            'Enhanced_XGBoost': 'xgboost',
            'Enhanced_LightGBM': 'lightgbm'
        }
        
        for model_name, model_type in models_to_train.items():
            print(f"\n🔧 {model_name} 훈련 중...")
            
            # 하이퍼파라미터 튜닝
            if self.use_optuna:
                best_params = self.hyperparameter_tuning(X_train, y_train, model_type)
            else:
                # 기본 파라미터 사용
                if model_type == 'xgboost':
                    best_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
                elif model_type == 'lightgbm':
                    best_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
            
            # 모델 생성 및 훈련
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(**best_params)
            elif model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**best_params)
            
            model.fit(X_train, y_train)
            
            # 예측 및 평가
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.models[model_name] = model
            self.results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'params': best_params
            }
            
            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R²: {r2:.3f}")
        
        # 최고 성능 모델 선택
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 최고 성능 모델: {best_model_name}")
        print(f"  MAE: {self.results[best_model_name]['mae']:.3f}")
        print(f"  R²: {self.results[best_model_name]['r2']:.3f}")
        
        return self.results
    
    def analyze_feature_importance(self):
        """특성 중요도 분석"""
        print("\n📊 특성 중요도 분석")
        print("=" * 40)
        
        if self.best_model is None:
            print("❌ 훈련된 모델이 없습니다.")
            return
        
        # 특성 중요도 추출
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        else:
            print("❌ 모델이 특성 중요도를 지원하지 않습니다.")
            return
        
        # 중요도 데이터프레임 생성
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # 상위 20개 특성 출력
        print("상위 20개 중요 특성:")
        for i, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def plot_results(self):
        """결과 시각화"""
        print("\n📊 결과 시각화 생성 중...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('고도화된 지하철 혼잡도 예측 모델 결과', fontsize=16, fontweight='bold')
        
        # 1) 모델 성능 비교
        model_names = list(self.results.keys())
        mae_scores = [self.results[name]['mae'] for name in model_names]
        r2_scores = [self.results[name]['r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, mae_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('모델별 MAE 비교')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, r2_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('모델별 R² 비교')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 2) 특성 중요도 (상위 15개)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            axes[0, 2].barh(top_features['feature'][::-1], top_features['importance'][::-1])
            axes[0, 2].set_title('특성 중요도 (상위 15개)')
            axes[0, 2].set_xlabel('중요도')
        
        # 3) 예측 vs 실제 (최고 모델)
        if self.best_model is not None:
            X_train, y_train, X_test, y_test, _ = self.prepare_features()
            if self.selected_features:
                X_test = X_test[self.selected_features]
            
            y_pred = self.best_model.predict(X_test)
            
            # 샘플링 (시각화 최적화)
            if len(y_test) > 10000:
                sample_idx = np.random.choice(len(y_test), 10000, replace=False)
                y_test_sample = y_test.iloc[sample_idx]
                y_pred_sample = y_pred[sample_idx]
            else:
                y_test_sample = y_test
                y_pred_sample = y_pred
            
            axes[1, 0].scatter(y_test_sample, y_pred_sample, alpha=0.5, s=1)
            axes[1, 0].plot([y_test_sample.min(), y_test_sample.max()], 
                           [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('실제 혼잡도')
            axes[1, 0].set_ylabel('예측 혼잡도')
            axes[1, 0].set_title('예측 vs 실제 (최고 모델)')
            
            # 4) 잔차 분포
            residuals = y_test_sample - y_pred_sample
            axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('잔차 (실제 - 예측)')
            axes[1, 1].set_ylabel('빈도')
            axes[1, 1].set_title('잔차 분포')
            axes[1, 1].axvline(0, color='red', linestyle='--')
            
            # 5) 시간대별 성능
            test_data_with_pred = self.test_data.copy()
            test_data_with_pred['predictions'] = self.best_model.predict(X_test)
            test_data_with_pred['residuals'] = abs(test_data_with_pred['congestion'] - test_data_with_pred['predictions'])
            
            hourly_mae = test_data_with_pred.groupby('hour')['residuals'].mean()
            axes[1, 2].plot(hourly_mae.index, hourly_mae.values, marker='o', linewidth=2)
            axes[1, 2].set_title('시간대별 예측 오차 (MAE)')
            axes[1, 2].set_xlabel('시간')
            axes[1, 2].set_ylabel('평균 절대 오차')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig('../result/enhanced_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='enhanced_subway_model.pkl'):
        """모델 저장"""
        model_data = {
            'best_model': self.best_model,
            'feature_engineer': self.feature_engineer,
            'selected_features': self.selected_features,
            'results': self.results,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ 모델 저장 완료: {filename}")
    
    def generate_insights(self):
        """개선된 모델 인사이트"""
        print("\n" + "=" * 60)
        print("🚀 고도화된 모델 인사이트")
        print("=" * 60)
        
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_result = self.results[best_model_name]
        
        print(f"\n🏆 최고 성능 모델: {best_model_name}")
        print(f"  📊 MAE: {best_result['mae']:.3f}")
        print(f"  📊 RMSE: {best_result['rmse']:.3f}")
        print(f"  📊 R²: {best_result['r2']:.3f}")
        
        # 성능 개선 분석
        print(f"\n📈 모델 개선 효과:")
        if best_result['r2'] > 0.8:
            print("  ✅ 우수한 예측 성능 달성")
        elif best_result['r2'] > 0.6:
            print("  ⭐ 양호한 예측 성능")
        else:
            print("  🔧 추가 개선 필요")
        
        # 특성 중요도 인사이트
        if self.feature_importance is not None:
            print(f"\n🎯 핵심 예측 요인 (상위 5개):")
            for i, row in self.feature_importance.head(5).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
            
            # 특성 유형별 분석
            importance_by_type = {}
            for _, row in self.feature_importance.iterrows():
                feature = row['feature']
                if any(x in feature for x in ['extreme_', 'heavy_', 'strong_']):
                    importance_by_type['극한기상'] = importance_by_type.get('극한기상', 0) + row['importance']
                elif any(x in feature for x in ['interaction', '_x_', 'temp_humidity']):
                    importance_by_type['상호작용'] = importance_by_type.get('상호작용', 0) + row['importance']
                elif any(x in feature for x in ['lag_', '_ma_', 'shift']):
                    importance_by_type['시차특성'] = importance_by_type.get('시차특성', 0) + row['importance']
                elif any(x in feature for x in ['hour', 'day', 'time_period', 'season']):
                    importance_by_type['시간특성'] = importance_by_type.get('시간특성', 0) + row['importance']
                else:
                    importance_by_type['기본특성'] = importance_by_type.get('기본특성', 0) + row['importance']
            
            print(f"\n📊 특성 유형별 중요도:")
            for feature_type, importance in sorted(importance_by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature_type}: {importance:.3f}")
        
        print(f"\n💡 실무 활용 제안:")
        print("  🎯 극한 기상 조건에서의 혼잡도 예측 정확도 향상")
        print("  🎯 세분화된 시간대별 맞춤형 운영 전략 수립")
        print("  🎯 기상-시간 상호작용을 고려한 동적 배차 계획")
        print("  🎯 특성 중요도 기반 핵심 요인 모니터링")

    def separate_time_effects(self, method='residual'):
        """시간 효과 분리"""
        print(f"\n🕐 시간 효과 분리 시작 ({method} 방법)")
        print("=" * 50)
        
        if method == 'residual':
            return self._residual_based_separation()
        elif method == 'decompose':
            return self._seasonal_decompose_separation()
        else:
            print("❌ 지원되지 않는 방법입니다. 'residual' 또는 'decompose'를 사용하세요.")
            return None
    
    def _residual_based_separation(self):
        """잔차 기반 시간 효과 분리"""
        print("📊 잔차 기반 시간 효과 분리 중...")
        
        # 시간 변수만으로 간단 모델 학습
        time_features = ['hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        available_time_features = [f for f in time_features if f in self.train_data.columns]
        
        print(f"시간 특성: {available_time_features}")
        
        # 시간 모델 학습
        from sklearn.ensemble import RandomForestRegressor
        time_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X_time_train = self.train_data[available_time_features]
        y_time_train = self.train_data['congestion']
        X_time_test = self.test_data[available_time_features]
        y_time_test = self.test_data['congestion']
        
        time_model.fit(X_time_train, y_time_train)
        
        # 시간 효과 예측
        time_pred_train = time_model.predict(X_time_train)
        time_pred_test = time_model.predict(X_time_test)
        
        # 잔차 계산 (시간 효과 제거)
        self.train_data['congestion_residual'] = y_time_train - time_pred_train
        self.test_data['congestion_residual'] = y_time_test - time_pred_test
        
        print(f"시간 모델 R²: {time_model.score(X_time_train, y_time_train):.3f}")
        print(f"원본 혼잡도 분산: {y_time_train.var():.3f}")
        print(f"잔차 분산: {self.train_data['congestion_residual'].var():.3f}")
        print(f"시간 효과 제거율: {(1 - self.train_data['congestion_residual'].var()/y_time_train.var())*100:.1f}%")
        
        return time_model
    
    def _seasonal_decompose_separation(self):
        """시계열 분해 기반 시간 효과 분리"""
        print("📈 시계열 분해 기반 시간 효과 분리 중...")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # 역별로 시계열 분해
            for data_name, data in [('train', self.train_data), ('test', self.test_data)]:
                print(f"{data_name} 데이터 분해 중...")
                
                residuals = []
                for station in data['station_name'].unique():
                    station_data = data[data['station_name'] == station].sort_values('tm')
                    
                    if len(station_data) >= 48:  # 최소 2일 데이터
                        try:
                            # 시계열 분해 (24시간 주기)
                            decomposition = seasonal_decompose(
                                station_data['congestion'], 
                                model='additive', 
                                period=24,
                                extrapolate_trend='freq'
                            )
                            
                            station_residuals = decomposition.resid.fillna(0)
                            residuals.extend(zip(station_data.index, station_residuals))
                            
                        except Exception as e:
                            print(f"⚠️ {station} 역 분해 실패: {str(e)}")
                            # 실패 시 원본값 사용
                            residuals.extend(zip(station_data.index, station_data['congestion']))
                    else:
                        # 데이터 부족 시 원본값 사용  
                        residuals.extend(zip(station_data.index, station_data['congestion']))
                
                # 잔차 설정
                residual_dict = dict(residuals)
                data['congestion_residual'] = data.index.map(residual_dict).fillna(data['congestion'])
                
                print(f"{data_name} 원본 분산: {data['congestion'].var():.3f}")
                print(f"{data_name} 잔차 분산: {data['congestion_residual'].var():.3f}")
                
        except ImportError:
            print("⚠️ statsmodels 미설치. pip install statsmodels")
            return self._residual_based_separation()  # 대안 사용
            
        return True

def main():
    """메인 실행 함수 - 시간 효과 분리 포함"""
    print("🚀 고도화된 지하철 혼잡도 예측 모델 (시간효과분리)")
    print("=" * 60)
    
    # 모델 초기화
    model = EnhancedSubwayModel(use_optuna=True, n_trials=10)
    
    # 1. 데이터 로드 및 전처리
    if not model.load_and_preprocess_data(train_years=['21'], test_year='23', sample_size=5000000):
        return None
    
    # 2. 시간 효과 분리 (옵션)
    use_time_separation = True  # 시간 효과 분리 사용 여부
    separation_method = 'residual'  # 'residual' 또는 'decompose'
    
    if use_time_separation:
        model.separate_time_effects(method=separation_method)
        target_col = 'congestion_residual'  # 잔차를 타겟으로 사용
        print(f"✅ 타겟 변수: {target_col} (시간 효과 제거됨)")
    else:
        target_col = 'congestion'  # 원본 사용
        print(f"✅ 타겟 변수: {target_col} (원본)")
    
    # 3. 특성 준비 (시간 효과 분리된 타겟 사용)
    original_congestion = model.train_data['congestion'].copy()
    model.train_data['congestion'] = model.train_data[target_col]
    model.test_data['congestion'] = model.test_data[target_col]
    
    X_train, y_train, X_test, y_test, feature_cols = model.prepare_features()
    
    # 4. 특성 선택
    X_train_selected, X_test_selected, selected_features = model.feature_selection(
        X_train, y_train, X_test, feature_cols, method='simple'
    )
    
    # 5. 모델 훈련
    results = model.train_enhanced_models(X_train_selected, y_train, X_test_selected, y_test)
    
    # 6. 분석 및 저장
    model.analyze_feature_importance()
    
    try:
        filename = f'enhanced_model_{"time_separated" if use_time_separation else "original"}.pkl'
        model.save_model(filename)
    except:
        model.save_model('enhanced_model_time_separated.pkl')
    
    model.generate_insights()
    
    # 시간 효과 분리 결과 요약
    if use_time_separation:
        print(f"\n🕐 시간 효과 분리 요약:")
        print(f"  방법: {separation_method}")
        print(f"  시간 효과 제거로 기상변수 영향 더 명확히 분석 가능")
        
    print(f"\n🎉 시간효과 분리 모델 완료!")
    return model

if __name__ == "__main__":
    enhanced_model = main() 