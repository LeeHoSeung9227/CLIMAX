# CLIMAX
기상청 빅데이터 콘테스트 - 기상과 지하철 혼잡도 상관분석 및 예측


## 프로젝트 개요 (임시)
- 기상 데이터와 지하철 혼잡도 데이터를 활용해 혼잡도 변화 요인을 분석하고 예측 모델을 개발합니다.
- 목표는 날씨, 시간대, 요일 등의 변수로 혼잡도를 정확히 예측하고, 이를 바탕으로 운영 및 정책 개선 방향을 제시하는 것입니다.

## 폴더 구조
- `data/` : 원본 및 전처리 데이터 (용량 큰 원본 데이터는 Git 현재 .gitignore해둠)
- `EDA/` : Jupyter 노트북 (탐색적 데이터 분석EDA21~23등)
- `Model/` : 예측모델 (아직 미완성)
- `results/` : 분석 결과, 그래프, 예측 파일 등등 저장
- `report/` : 추후 추가 예정

## 환경 세팅
- 기본 Python 패키지는 `requirements.txt`에 명시되어 있습니다.
- 필요시 `conda` 환경 설정 파일 `environment.yaml`을 참고하세요.
- 따로 가상환경 세팅 시 conda 환경 설정 필요없고 requirements만 사용

```bash
pip install -r requirements.txt
# 또는
conda env create -f environment.yaml

