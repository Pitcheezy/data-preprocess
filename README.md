# Pitcheezy Data Preprocess

MLB Statcast 데이터를 수집하고 전처리하는 파이프라인입니다.

## 개요

지정한 기간(`--start`, `--end`)의 데이터를 아래 순서로 처리합니다.

1. **00 Fetch**: pybaseball으로 Statcast pitch-by-pitch 데이터 수집
2. **01 Preprocess**: 컬럼 선택, 타입 변환, 파생 변수 생성
3. **02 Embedding**: 투수별 UMAP + HDBSCAN 클러스터링 (구종 유사 그룹)
4. **03 Profiles**: 투수·타자 프로필 생성
5. **04 Matchup**: 투수-타자 매치업 테이블 생성

## 환경 설정

- Python 3.12+
- uv 또는 pip로 의존성 설치

```bash
# uv 사용 시
uv sync

# pip 사용 시
pip install -e .
```

## 사용법

### 전체 파이프라인 실행

기간을 지정하면 해당 기간 데이터를 수집하고 전처리합니다.

```bash
python main.py --start 2025-03-01 --end 2025-03-31
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--start` | 시작일 (YYYY-MM-DD, 필수) |
| `--end` | 종료일 (YYYY-MM-DD, 필수) |
| `--project-root` | 프로젝트 루트 경로 (기본: 자동 탐색) |
| `--skip-fetch` | 수집 생략 (기존 raw CSV 사용) |
| `--skip-embedding` | 02 embedding/clustering 생략 |
| `--skip-profiles` | 03 profiles 생략 |
| `--skip-matchup` | 04 matchup 생략 |
| `--top-pitchers N` | 투구수 상위 N명 투수만 사용 (모델 학습용) |
| `--top-batters N` | 타석 투구수 상위 N명 타자만 사용 (모델 학습용) |

### 예시

```bash
# 2025년 3월 전체 수집 및 전처리
python main.py --start 2025-03-01 --end 2025-03-31

# 수집 생략 후 전처리만 재실행 (이미 raw CSV가 있을 때)
python main.py --start 2025-03-01 --end 2025-03-31 --skip-fetch

# embedding 이후 단계만 재실행
python main.py --start 2025-03-01 --end 2025-03-31 --skip-fetch --skip-embedding

# 모델 학습용: 투구수 상위 300명 투수·400명 타자만 포함
python main.py --start 2025-03-01 --end 2025-03-31 --top-pitchers 300 --top-batters 400
```

## 디렉터리 구조

```
data_preprocess/
├── main.py           # 진입점
├── src/
│   ├── fetch.py      # Statcast 수집 (pybaseball)
│   ├── preprocess.py # 전처리 (컬럼, 타입, 파생변수)
│   ├── embedding.py  # UMAP + HDBSCAN
│   ├── profiles.py   # 투수·타자 프로필
│   ├── matchup.py    # 매치업 테이블
│   └── io_utils.py   # 경로, Parquet I/O
├── data/
│   ├── raw/          # 수집된 CSV
│   └── processed/    # Parquet 결과물
└── pyproject.toml
```

## 출력 파일 (기간별)

`data/raw/`, `data/processed/`에 `{start}_to_{end}` 형식으로 저장됩니다.

| 파일 | 설명 |
|------|------|
| `statcast_{start}_to_{end}.csv` | 수집 원시 데이터 |
| `pitch_clean_{start}_to_{end}.parquet` | 전처리 결과 |
| `pitch_umap_cluster_{start}_to_{end}.parquet` | UMAP·클러스터 결과 |
| `pitcher_cluster_summary_{start}_to_{end}.csv` | 투수별 요약 |
| `pitcher_profiles_{start}_to_{end}.parquet` | 투수 프로필 |
| `batter_profiles_{start}_to_{end}.parquet` | 타자 프로필 |
| `matchup_pitch_level_{start}_to_{end}.parquet` | 매치업 (투구 단위) |
| `matchup_pair_level_{start}_to_{end}.parquet` | 매치업 (쌍 단위) |

## 라이선스

(프로젝트에 맞게 설정)
