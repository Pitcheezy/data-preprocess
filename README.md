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

기간을 지정하면 해당 기간 데이터를 수집하고 전처리합니다. (uv 사용 시 `uv run` 필요)

```bash
cd data_preprocess
uv run python main.py --start 2025-03-01 --end 2025-03-31
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
cd data_preprocess

# 2025년 3월 전체 수집 및 전처리
uv run python main.py --start 2025-03-01 --end 2025-03-31

# 수집 생략 후 전처리만 재실행 (이미 raw CSV가 있을 때)
uv run python main.py --start 2025-03-01 --end 2025-03-31 --skip-fetch

# embedding 이후 단계만 재실행
uv run python main.py --start 2025-03-01 --end 2025-03-31 --skip-fetch --skip-embedding

# 모델 학습용: 투구수 상위 300명 투수·400명 타자만 포함
uv run python main.py --start 2025-03-01 --end 2025-03-31 --top-pitchers 300 --top-batters 400
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

## pybaseball(Statcast)에서 사용하는 컬럼

수집은 `pybaseball.statcast(start_dt, end_dt)`로 pitch-by-pitch 원시 데이터를 가져온 뒤, 아래 컬럼만 남깁니다.

### 전체 선택 컬럼 (SELECTED_COLS)

수집 직후 전처리 단계에서 **사용·유지하는 컬럼**만 남기고 나머지는 제거합니다.

| 컬럼 | 설명 | 투수용 | 타자용 |
|------|------|:------:|:------:|
| `pitcher` | 투수 MLBAM ID | ✓ | |
| `batter` | 타자 MLBAM ID | | ✓ |
| `pitch_type` | 구종 코드 (FF, SL, CU 등) | ✓ | |
| `game_date` | 경기일 | ✓ | ✓ |
| `release_speed` | 구속 (mph) | ✓ | |
| `release_spin_rate` | 회전수 (rpm) | ✓ | |
| `release_pos_x`, `release_pos_z` | 손 떼는 위치 | ✓ | |
| `release_extension` | 릴리스 연장 (ft) | ✓ | |
| `arm_angle` | 팔 각도 | ✓ | |
| `pfx_x`, `pfx_z` | 횡·종 무브먼트 (인치) | ✓ | |
| `plate_x`, `plate_z` | 홈플레이트 통과 위치 | ✓ | ✓ |
| `balls`, `strikes` | 볼·스트라이크 카운트 | ✓ | ✓ |
| `zone` | 스트라이크존 번호 | ✓ | ✓ |
| `on_1b`, `on_2b`, `on_3b` | 1·2·3루 주자 유무 | ✓ | ✓ |
| `outs_when_up` | 타석 시 아웃 수 | ✓ | ✓ |
| `inning`, `inning_topbot` | 이닝·초/말 | ✓ | ✓ |
| `bat_score_diff` | 타석 시 점수차 | ✓ | ✓ |
| `pitch_number` | 해당 타자 상대 누적 투구 수 | ✓ | ✓ |
| `events` | 경기 결과 (안타, 아웃 등) | | ✓ |
| `description` | 투구 결과 설명 (스트라이크, 볼 등) | | ✓ |
| `stand` | 타석 (L/R) | | ✓ |
| `p_throws` | 투수 손 (L/R) | ✓ | |
| `bb_type` | 타구 유형 | | ✓ |
| `launch_speed` | 타구속도 (mph) | | ✓ |
| `launch_angle` | 타구각 (도) | | ✓ |

전처리에서 추가하는 **파생 컬럼**: `description_group`, `events_group`, `base_state`, `count`.

### 투수별로 쓰는 컬럼

- **UMAP·HDBSCAN (구종 클러스터링)**  
  `release_speed`, `release_spin_rate`, `pfx_x`, `pfx_z`, `release_pos_x`, `release_pos_z`, `release_extension`, `arm_angle`  
  → 투수별로 위 8개로 스케일·PCA·UMAP·HDBSCAN 수행.

- **투수 프로필**  
  위 피처 + `pitch_type`, `pitch_cluster_id`, `pitch_cluster_local`  
  → 클러스터(또는 pitch_type) 비율·가중평균 구속/스핀 등으로 투수 1행 요약.

### 타자별로 쓰는 컬럼

- **타자 프로필**  
  `launch_speed`, `launch_angle` (평균), `stand` (최빈값), `events_group`, `description_group` (비율 분포)  
  → 타자 1행 요약.

### 매치업·공통

- **pitch_level / pair_level**  
  위 투수·타자 프로필 + `pitcher`, `batter`, `game_date`, `pitch_type`, `balls`, `strikes`, `plate_x`, `plate_z`, `zone`, 주자·이닝·점수차 등 상황 컬럼을 함께 사용.

## 라이선스

(프로젝트에 맞게 설정)
