import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
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

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class EnhancedFeatureEngineering:
    def create_detailed_time_features(self, df):
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
        df['day_type'] = df['dayofweek'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')

        # 월요일/금요일 효과
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)

        # 토요일/일요일 효과
        df['is_saturday'] = (df['dayofweek'] == 5).astype(int)
        df['is_sunday'] = (df['dayofweek'] == 6).astype(int)

        # 평일/주말 구분 (이진)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # 출근/퇴근 시간대 (더 세밀하게)
        df['is_morning_commute'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
        df['is_evening_commute'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
        df['is_commute_time'] = df['is_morning_commute'] | df['is_evening_commute']

        # 피크 시간대 세분화 (8-9시, 18-19시)
        df['is_morning_peak'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 9 else 0)
        df['is_evening_peak'] = df['hour'].apply(lambda x: 1 if 18 <= x <= 19 else 0)
        df['is_peak_time'] = df['is_morning_peak'] | df['is_evening_peak']

        # 주말 피크 시간대 (주말은 8시 이후 피크)
        df['is_weekend_peak'] = ((df['is_weekend'] == 1) & (df['hour'] >= 8)).astype(int)
        df['is_weekend_morning'] = ((df['is_weekend'] == 1) & (df['hour'].between(8, 12))).astype(int)
        df['is_weekend_afternoon'] = ((df['is_weekend'] == 1) & (df['hour'].between(13, 17))).astype(int)
        df['is_weekend_evening'] = ((df['is_weekend'] == 1) & (df['hour'].between(18, 22))).astype(int)

        # 요일-시간대 조합 변수
        df['day_hour_combo'] = df['dayofweek'].astype(str) + '_' + df['hour'].astype(str)

        # 순환적 시간 특성 (기존 + 추가)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        return df

class EnhancedSubwayModel:
    def __init__(self, use_optuna=True, n_trials=10):
        self.feature_engineer = EnhancedFeatureEngineering()
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.results = {}

    def load_and_preprocess_data(self, train_years=['22'], test_year='23', sample_size=5000000):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(base_dir, '..', '데이터'))
        train_dfs = []
        for year in train_years:
            file_path = os.path.join(data_dir, f'train_subway{year}.csv')
            df = pd.read_csv(file_path, encoding='cp949', nrows=sample_size)
            df.columns = [col.replace(f'train_subway{year}.', '') for col in df.columns]
            df['datetime'] = pd.to_datetime(df['tm'], format='%Y%m%d%H')
            df = df[(df['datetime'].dt.month.isin([6,7,8])) & ~((df['datetime'].dt.month == 8) & (df['datetime'].dt.day > 31))]
            train_dfs.append(df)
        self.train_data = pd.concat(train_dfs).reset_index(drop=True)
        test_file = os.path.join(data_dir, f'train_subway{test_year}.csv')
        self.test_data = pd.read_csv(test_file, encoding='cp949', nrows=sample_size)
        self.test_data.columns = [col.replace(f'train_subway{test_year}.', '') for col in self.test_data.columns]
        self.test_data['datetime'] = pd.to_datetime(self.test_data['tm'], format='%Y%m%d%H')
        self.test_data = self.test_data[(self.test_data['datetime'].dt.month.isin([6,7,8])) & ~((self.test_data['datetime'].dt.month == 8) & (self.test_data['datetime'].dt.day > 31))]
        # 특정 역만 필터링
        station_to_use = '강남'  # 원하는 역 이름으로 변경 가능
        self.train_data = self.train_data[self.train_data['station_name'] == station_to_use]
        self.test_data = self.test_data[self.test_data['station_name'] == station_to_use]
        self.original_train_congestion = self.train_data['congestion'].copy()
        self.original_test_congestion = self.test_data['congestion'].copy()
        if 'congestion' in self.test_data.columns:
            self.test_data['congestion'] = pd.to_numeric(self.test_data['congestion'], errors='coerce')
        self.train_data = self.feature_engineer.create_detailed_time_features(self.train_data)
        self.test_data = self.feature_engineer.create_detailed_time_features(self.test_data)
        # 여름(6,7,8월)만 필터링
        self.train_data = self.train_data[self.train_data['month'].isin([6,7,8])]
        self.test_data = self.test_data[self.test_data['month'].isin([6,7,8])]
        self.train_data['congestion'] = self.original_train_congestion
        self.test_data['congestion'] = self.original_test_congestion
        # 날짜 컬럼 생성 (datetime에서 날짜만 추출)
        self.test_data['date'] = self.test_data['datetime'].dt.date
        # direction 인코딩 (train, test 모두)
        le = LabelEncoder()
        # train_data, test_data 모두에 direction_encoded 컬럼 추가
        self.train_data['direction_encoded'] = le.fit_transform(self.train_data['direction'])
        self.test_data['direction_encoded'] = le.transform(self.test_data['direction'])

        # 공휴일 리스트 정의
        holidays_2021 = {
            '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-02-14',
            '2021-03-01', '2021-05-01', '2021-05-05', '2021-05-19', '2021-06-06',
            '2021-08-15', '2021-08-16', '2021-09-18', '2021-09-19', '2021-09-20',
            '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-04', '2021-10-09',
            '2021-10-10', '2021-10-11', '2021-10-20', '2021-11-18', '2021-12-25'
        }
        holidays_2022 = {
            '2022-01-01', '2022-01-29', '2022-01-30', '2022-01-31', '2022-02-01', '2022-02-02',
            '2022-03-01', '2022-03-09', '2022-05-01', '2022-05-05', '2022-05-08', '2022-06-01',
            '2022-06-04', '2022-06-05', '2022-06-06', '2022-08-13', '2022-08-14', '2022-08-15',
            '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12', '2022-10-01', '2022-10-02',
            '2022-10-03', '2022-10-08', '2022-10-09', '2022-10-10', '2022-11-17', '2022-11-30', '2022-12-25'
        }
        holidays_2023 = {
            '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
            '2023-03-01', '2023-05-05', '2023-05-06', '2023-05-07', '2023-05-27', '2023-05-28', '2023-05-29',
            '2023-06-06', '2023-08-15', '2023-09-14', '2023-09-28', '2023-09-29', '2023-09-30',
            '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-07', '2023-10-08', '2023-10-09',
            '2023-11-16', '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-30', '2023-12-31'
        }
        all_holidays = set()
        all_holidays.update(holidays_2021)
        all_holidays.update(holidays_2022)
        all_holidays.update(holidays_2023)

        # 공휴일 feature 생성 (train, test 모두)
        def make_is_holiday_col(df):
            date_str_col = df['datetime'].dt.strftime('%Y-%m-%d')
            return date_str_col.isin(all_holidays).astype(int)
        self.train_data['is_holiday'] = make_is_holiday_col(self.train_data)
        self.test_data['is_holiday'] = make_is_holiday_col(self.test_data)

        # 출근/퇴근 시간대 feature (train, test 모두)
        self.train_data['is_morning_commute'] = self.train_data['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
        self.train_data['is_evening_commute'] = self.train_data['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
        self.test_data['is_morning_commute'] = self.test_data['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
        self.test_data['is_evening_commute'] = self.test_data['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)

        # 피크 시간대 세분화 (8-9시, 18-19시)
        self.train_data['is_morning_peak'] = self.train_data['hour'].apply(lambda x: 1 if 8 <= x <= 9 else 0)
        self.train_data['is_evening_peak'] = self.train_data['hour'].apply(lambda x: 1 if 18 <= x <= 19 else 0)
        self.test_data['is_morning_peak'] = self.test_data['hour'].apply(lambda x: 1 if 8 <= x <= 9 else 0)
        self.test_data['is_evening_peak'] = self.test_data['hour'].apply(lambda x: 1 if 18 <= x <= 19 else 0)


        # 주말 vs 평일 시간대별 상호작용 (8시 이후 집중)
        self.train_data['weekend_hour_8_12'] = ((self.train_data['is_weekend'] == 1) & (self.train_data['hour'].between(8, 12))).astype(int)
        self.train_data['weekend_hour_13_17'] = ((self.train_data['is_weekend'] == 1) & (self.train_data['hour'].between(13, 17))).astype(int)
        self.train_data['weekend_hour_18_22'] = ((self.train_data['is_weekend'] == 1) & (self.train_data['hour'].between(18, 22))).astype(int)
        
        self.test_data['weekend_hour_8_12'] = ((self.test_data['is_weekend'] == 1) & (self.test_data['hour'].between(8, 12))).astype(int)
        self.test_data['weekend_hour_13_17'] = ((self.test_data['is_weekend'] == 1) & (self.test_data['hour'].between(13, 17))).astype(int)
        self.test_data['weekend_hour_18_22'] = ((self.test_data['is_weekend'] == 1) & (self.test_data['hour'].between(18, 22))).astype(int)

        # 공휴일+출근/퇴근 상호작용 feature (train, test 모두)
        self.train_data['is_holiday_and_morning'] = ((self.train_data['is_holiday'] == 1) & (self.train_data['is_morning_commute'] == 1)).astype(int)
        self.train_data['is_holiday_and_evening'] = ((self.train_data['is_holiday'] == 1) & (self.train_data['is_evening_commute'] == 1)).astype(int)
        self.test_data['is_holiday_and_morning'] = ((self.test_data['is_holiday'] == 1) & (self.test_data['is_morning_commute'] == 1)).astype(int)
        self.test_data['is_holiday_and_evening'] = ((self.test_data['is_holiday'] == 1) & (self.test_data['is_evening_commute'] == 1)).astype(int)

        return True

    def prepare_features(self):
        feature_cols = [
            'hour', 'dayofweek', 'is_weekend', 'month', 
            'season','is_winter', 'is_spring', 'is_summer', 'is_fall',  # 계절 변수
            'direction_encoded',  # 상선 / 하선
            'is_holiday',  # 공휴일 feature # 일단 하드코딩으로 때려박음 나중에 csv불러오든가 해야할듯
            'is_morning_commute', 'is_evening_commute',  # 출근/퇴근 시간대
            'is_morning_peak', 'is_evening_peak',  # 피크 시간대 세분화
            'is_holiday_and_morning', 'is_holiday_and_evening',  # 공휴일+출근/퇴근 상호작용
        ]
        feature_cols = [col for col in feature_cols if col in self.train_data.columns]
        X_train = self.train_data[feature_cols].copy()
        y_train = self.train_data['congestion'].copy()
        X_test = self.test_data[feature_cols].copy()
        y_test = self.test_data['congestion'].copy()
        train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        return X_train, y_train, X_test, y_test, feature_cols

    def train_enhanced_models(self, X_train, y_train, X_test, y_test):
        # 피크 시간대에 더 높은 가중치 부여
        sample_weights = np.ones(len(X_train))
        
        # 주말 전체에 높은 가중치 (기존 2.0에서 3.0으로 증가)
        weekend_mask = X_train['is_weekend'] == 1
        sample_weights[weekend_mask] = 3.0
        
     
        # 평일 피크 시간대 (8-9시, 18-19시)에 높은 가중치
        morning_peak_mask = X_train['is_morning_peak'] == 1
        evening_peak_mask = X_train['is_evening_peak'] == 1
        sample_weights[morning_peak_mask] = 2.0
        sample_weights[evening_peak_mask] = 2.0
        
        # 극값(높은 혼잡도)에 더 높은 가중치
        high_congestion_mask = y_train > y_train.quantile(0.8)
        sample_weights[high_congestion_mask] *= 1.8
        
        # 주말에서 높은 혼잡도는 더 높은 가중치
        weekend_high_congestion = (weekend_mask) & (high_congestion_mask)
        sample_weights[weekend_high_congestion] *= 2.0
        
      
        
        print(f"가중치 통계:")
        print(f"- 전체 데이터: {len(sample_weights)}")
        print(f"- 주말 데이터: {weekend_mask.sum()} (가중치: {sample_weights[weekend_mask].mean():.2f})")
        print(f"- 평일 데이터: {(~weekend_mask).sum()} (가중치: {sample_weights[~weekend_mask].mean():.2f})")
        # XGBoost 모델 하이퍼파라미터 조정
        model = xgb.XGBRegressor(
            n_estimators=2000,  # 더 많은 트리
            max_depth=15,       # 더 깊은 트리
            learning_rate=0.03, # 더 낮은 학습률로 세밀하게 학습
            subsample=0.8,      # 과적합 방지
            colsample_bytree=0.8,
            reg_alpha=0.1,      # L1 정규화
            reg_lambda=1.0,     # L2 정규화
            random_state=42
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        self.best_model = model
        # 특성 중요도 저장
        self.feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
        print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    model = EnhancedSubwayModel()
    model.load_and_preprocess_data()
    X_train, y_train, X_test, y_test, feature_cols = model.prepare_features()
    model.train_enhanced_models(X_train, y_train, X_test, y_test)

    # 예측 및 저장
    y_pred_full = model.best_model.predict(X_test)
    actual_congestion = model.original_test_congestion

    # 날짜(datetime) 컬럼을 'YYYYMMDDHHMM' 형식으로 변환
    datetime_col = model.test_data['datetime'].dt.strftime('%Y%m%d%H%M').reset_index(drop=True)
    # direction 컬럼도 포함
    direction_col = model.test_data['direction'].reset_index(drop=True)
    direction_encoded_col = model.test_data['direction_encoded'].reset_index(drop=True)
    # 이미 생성된 공휴일 feature 사용
    is_holiday_col = model.test_data['is_holiday'].reset_index(drop=True)
    # 출근/퇴근 상호작용 feature 생성
    is_morning_commute = model.test_data['hour'].reset_index(drop=True).apply(lambda x: 1 if 7 <= x <= 9 else 0)
    is_evening_commute = model.test_data['hour'].reset_index(drop=True).apply(lambda x: 1 if 17 <= x <= 19 else 0)
    is_morning_outer = ((is_morning_commute == 1) & (direction_col == '상선')).astype(int)
    is_evening_inner = ((is_evening_commute == 1) & (direction_col == '하선')).astype(int)
    pred_df = pd.DataFrame({'congestion2': y_pred_full}, index=X_test.index)
    result_df = pd.DataFrame({
        'train_subway21.tm': datetime_col,
        'congestion1': actual_congestion.reset_index(drop=True),
        'congestion2': pred_df['congestion2'].reset_index(drop=True),
        'direction': direction_col,
        'direction_encoded': direction_encoded_col,
        'is_holiday': is_holiday_col,
        'is_morning_outer': is_morning_outer,
        'is_evening_inner': is_evening_inner
    })
    result_df.to_csv('pred_result_hour_only.csv', index=False, encoding='utf-8-sig')
    print('✅ pred_result_hour_only.csv 저장 완료!')

    # train, test 데이터의 hour별 분포 확인
    print('\n[train 데이터 hour별 분포]')
    print(model.train_data['hour'].value_counts().sort_index())
    print('\n[test 데이터 hour별 분포]')
    print(model.test_data['hour'].value_counts().sort_index())

    # 예측값, 실제값 시간별 평균 비교
    result_df['hour'] = result_df['train_subway21.tm'].str[8:10].astype(int)
    result_df['dayofweek'] = pd.to_datetime(result_df['train_subway21.tm'], format='%Y%m%d%H%M').dt.dayofweek
    result_df['is_weekend'] = (result_df['dayofweek'] >= 5).astype(int)
    
    print('\n[시간별 실제/예측 혼잡도 평균]')
    print(result_df.groupby('hour')[['congestion1', 'congestion2']].mean())
    
    print('\n[주말 vs 평일 혼잡도 비교]')
    weekend_stats = result_df[result_df['is_weekend'] == 1][['congestion1', 'congestion2']].describe()
    weekday_stats = result_df[result_df['is_weekend'] == 0][['congestion1', 'congestion2']].describe()
    print('\n[주말 통계]')
    print(weekend_stats)
    print('\n[평일 통계]')
    print(weekday_stats)
    
    # 주말 시간별 혼잡도 분석
    weekend_df = result_df[result_df['is_weekend'] == 1]
    if len(weekend_df) > 0:
        weekend_hourly = weekend_df.groupby('hour')[['congestion1', 'congestion2']].mean()
        print('\n[주말 시간별 혼잡도]')
        print(weekend_hourly)
        
        # 주말 시간별 예측 오차 분석
        weekend_df['xgb_error'] = abs(weekend_df['congestion1'] - weekend_df['congestion2'])
        weekend_error_by_hour = weekend_df.groupby('hour')[['xgb_error']].mean()
        print('\n[주말 시간별 예측 오차]')
        print(weekend_error_by_hour)
        
        # 주말 시간별 혼잡도 시각화
        plt.figure(figsize=(15, 5))
        
        # 주말 vs 평일 비교
        plt.subplot(1, 3, 1)
        plt.plot(weekend_hourly.index, weekend_hourly['congestion1'], marker='o', label='실제 혼잡도', linewidth=2)
        plt.plot(weekend_hourly.index, weekend_hourly['congestion2'], marker='s', label='XGBoost 예측', linewidth=2)

        plt.xlabel('시간 (Hour)')
        plt.ylabel('혼잡도')
        plt.title('주말 시간별 혼잡도 비교')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 주말 시간별 예측 오차
        plt.subplot(1, 3, 2)
        plt.plot(weekend_error_by_hour.index, weekend_error_by_hour['xgb_error'], marker='o', label='XGBoost 오차', linewidth=2)
        plt.xlabel('시간 (Hour)')
        plt.ylabel('평균 절대 오차')
        plt.title('주말 시간별 예측 오차')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 주말 시간대별 과소/과대 평가 분석
        plt.subplot(1, 3, 3)
        weekend_df['prediction_bias'] = weekend_df['congestion2'] - weekend_df['congestion1']
        bias_by_hour = weekend_df.groupby('hour')['prediction_bias'].mean()
        plt.bar(bias_by_hour.index, bias_by_hour.values, alpha=0.7, color='red')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('시간 (Hour)')
        plt.ylabel('예측 편향 (예측값 - 실제값)')
        plt.title('주말 시간별 예측 편향\n(음수=과소평가, 양수=과대평가)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 주말 8시 이후 과소평가 문제 상세 분석
        print('\n[주말 8시 이후 과소평가 문제 분석]')
        weekend_8plus = weekend_df[weekend_df['hour'] >= 8]
        if len(weekend_8plus) > 0:
            print(f"주말 8시 이후 데이터: {len(weekend_8plus)}개")
            print(f"평균 실제 혼잡도: {weekend_8plus['congestion1'].mean():.2f}")
            print(f"평균 XGBoost 예측: {weekend_8plus['congestion2'].mean():.2f}")
            print(f"XGBoost 과소평가 정도: {weekend_8plus['prediction_bias'].mean():.2f}")
            
            # 8시 이후 시간대별 과소평가 분석
            bias_8plus = weekend_8plus.groupby('hour')['prediction_bias'].mean()
            print('\n[주말 8시 이후 시간대별 과소평가]')
            print(bias_8plus)
            
            # 8시 이후 과소평가 시각화
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            actual_8plus = weekend_8plus.groupby('hour')['congestion1'].mean()
            pred_8plus = weekend_8plus.groupby('hour')['congestion2'].mean()
            plt.plot(actual_8plus.index, actual_8plus.values, marker='o', label='실제 혼잡도', linewidth=2)
            plt.plot(pred_8plus.index, pred_8plus.values, marker='s', label='XGBoost 예측', linewidth=2)
            plt.xlabel('시간 (Hour)')
            plt.ylabel('혼잡도')
            plt.title('주말 8시 이후 실제 vs 예측')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.bar(bias_8plus.index, bias_8plus.values, alpha=0.7, color='red')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('시간 (Hour)')
            plt.ylabel('과소평가 정도')
            plt.title('주말 8시 이후 과소평가 정도\n(음수=과소평가)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # 평일 시간별 혼잡도 시각화
        weekday_df = result_df[result_df['is_weekend'] == 0]
        weekday_hourly = weekday_df.groupby('hour')[['congestion1', 'congestion2']].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(weekday_hourly.index, weekday_hourly['congestion1'], marker='o', label='실제 혼잡도', linewidth=2)
        plt.plot(weekday_hourly.index, weekday_hourly['congestion2'], marker='s', label='XGBoost 예측', linewidth=2)
        plt.xlabel('시간 (Hour)')
        plt.ylabel('혼잡도')
        plt.title('평일 시간별 혼잡도 비교')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 특성 중요도 출력 및 시각화
    feature_importance = model.feature_importance.sort_values(ascending=False)
    print('\n[특성 중요도 상위 15개]')
    print(feature_importance.head(15))
    plt.figure(figsize=(10,5))
    feature_importance.head(15).plot(kind='bar')
    plt.title('XGBoost 특성 중요도 (상위 15개)')
    plt.ylabel('중요도')
    plt.tight_layout()
    plt.show()

    # 하루씩 시간별 혼잡도 비교 (방향별)
    result_df = pd.read_csv('pred_result_hour_only.csv', encoding='utf-8-sig')
    result_df['datetime'] = pd.to_datetime(result_df['train_subway21.tm'], format='%Y%m%d%H%M')
    result_df['date'] = result_df['datetime'].dt.date
    result_df['hour'] = result_df['datetime'].dt.hour
    unique_dates = result_df['date'].unique()
    for d in unique_dates:
        day_df = result_df[result_df['date'] == d]
        plt.figure(figsize=(10, 4))
        for direction in day_df['direction'].unique():
            dir_df = day_df[day_df['direction'] == direction]
            hourly_mean = dir_df.groupby('hour')[['congestion1', 'congestion2']].mean()
            plt.plot(hourly_mean.index, hourly_mean['congestion1'], label=f'실제({direction})', marker='o')
            plt.plot(hourly_mean.index, hourly_mean['congestion2'], label=f'XGBoost({direction})', marker='x')
        plt.title(f'{d} 시간별 혼잡도(방향별)')
        plt.xlabel('시간')
        plt.ylabel('혼잡도')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 월별로 데이터 분리 및 방향별 subplot
    result_df['month'] = result_df['datetime'].dt.month
    months = sorted(result_df['month'].unique())
    directions = result_df['direction'].unique()
    fig, axes = plt.subplots(len(directions), len(months), figsize=(18, 10), sharey=True)
    for i, direction in enumerate(directions):
        for j, m in enumerate(months):
            month_dir_df = result_df[(result_df['month'] == m) & (result_df['direction'] == direction)]
            daily_mean = month_dir_df.groupby('date')[['congestion1', 'congestion2']].mean()
            axes[i, j].plot(daily_mean.index, daily_mean['congestion1'], label='실제 혼잡도', marker='o')
            axes[i, j].plot(daily_mean.index, daily_mean['congestion2'], label='XGBoost 예측', marker='x')
            axes[i, j].set_title(f'{m}월 {direction} 일별 혼잡도(평균)')
            axes[i, j].set_xlabel('날짜')
            if j == 0:
                axes[i, j].set_ylabel('혼잡도')
            axes[i, j].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()

    print(model.test_data['direction'].unique())

    # 피크 시간대 예측 오차 분석
    result_df['is_morning_peak'] = result_df['hour'].apply(lambda x: 1 if 8 <= x <= 9 else 0)
    result_df['is_evening_peak'] = result_df['hour'].apply(lambda x: 1 if 18 <= x <= 19 else 0)
    result_df['is_peak_time'] = result_df['is_morning_peak'] | result_df['is_evening_peak']
    
    # 피크 시간대 오차 계산
    result_df['xgb_error'] = abs(result_df['congestion1'] - result_df['congestion2'])

    # 시간대별 평균 오차
    plt.subplot(1, 3, 1)
    hourly_error = result_df.groupby('hour')[['xgb_error']].mean()
    plt.plot(hourly_error.index, hourly_error['xgb_error'], marker='o', label='XGBoost 오차', linewidth=2)
    plt.xlabel('시간 (Hour)')
    plt.ylabel('평균 절대 오차')
    plt.title('시간대별 예측 오차')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 주말 vs 평일 오차 비교
    plt.subplot(1, 3, 2)
    weekend_error = result_df[result_df['is_weekend'] == 1][['xgb_error']].mean()
    weekday_error = result_df[result_df['is_weekend'] == 0][['xgb_error']].mean()
    x = ['주말', '평일']
    plt.bar([x[0], x[0]], [weekend_error['xgb_error']], 
            label=['XGBoost'], alpha=0.7)
    plt.bar([x[1], x[1]], [weekday_error['xgb_error']], 
            alpha=0.7)
    plt.ylabel('평균 절대 오차')
    plt.title('주말 vs 평일 예측 오차')
    plt.legend()
    
    # 피크 vs 비피크 오차 비교
    plt.subplot(1, 3, 3)
    peak_error_mean = result_df[result_df['is_peak_time'] == 1][['xgb_error']].mean()
    non_peak_error_mean = result_df[result_df['is_peak_time'] == 0][['xgb_error']].mean()
    x = ['비피크', '피크']
    plt.bar([x[0], x[0]], [non_peak_error_mean['xgb_error']], 
            label=['XGBoost'], alpha=0.7)
    plt.bar([x[1], x[1]], [peak_error_mean['xgb_error']], 
            alpha=0.7)
    plt.ylabel('평균 절대 오차')
    plt.title('피크 vs 비피크 예측 오차')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 