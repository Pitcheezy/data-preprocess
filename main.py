"""
Pitcheezy 데이터 파이프라인 진입점

지정한 기간(--start, --end)의 MLB Statcast 데이터를 수집하고,
전처리 → 임베딩(UMAP+HDBSCAN) → 프로필 → 매치업 테이블 순으로 처리합니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.io_utils import get_paths, log, save_parquet, read_parquet
from src.preprocess import preprocess_statcast
from src.embedding import run_pitcher_umap_cluster, EmbeddingConfig
from src.profiles import build_pitcher_profiles, build_batter_profiles
from src.matchup import build_matchup_tables, MatchupConfig


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱"""
    p = argparse.ArgumentParser(description="기간별 Statcast 데이터 수집 및 전처리 파이프라인")

    p.add_argument("--start", type=str, required=True, help="시작일 (YYYY-MM-DD)")
    p.add_argument("--end", type=str, required=True, help="종료일 (YYYY-MM-DD)")
    p.add_argument("--project-root", type=str, default=None, help="프로젝트 루트 경로 (기본: 자동 탐색)")

    p.add_argument("--skip-fetch", action="store_true", help="수집 생략 (기존 raw CSV 사용)")
    p.add_argument("--skip-embedding", action="store_true", help="02 embedding/clustering 생략")
    p.add_argument("--skip-profiles", action="store_true", help="03 profiles 생략")
    p.add_argument("--skip-matchup", action="store_true", help="04 matchup 생략")

    # 모델 학습용: 투구수 많은 상위 N명만 포함 (데이터 품질·속도 개선)
    p.add_argument(
        "--top-pitchers",
        type=int,
        default=None,
        help="투구수 상위 N명 투수만 사용 (예: 300, 400). 미지정 시 전체 포함",
    )
    p.add_argument(
        "--top-batters",
        type=int,
        default=None,
        help="타석 투구수 상위 N명 타자만 사용 (예: 300, 400). 미지정 시 전체 포함",
    )

    return p.parse_args()


def filter_top_players(
    df: pd.DataFrame,
    top_pitchers: int | None,
    top_batters: int | None,
) -> pd.DataFrame:
    """
    모델 학습용: 투구수 상위 N명 투수·타자만 남김.

    전체 데이터 기준으로 상위 N명을 계산한 뒤,
    (pitcher, batter) 쌍이 두 집합에 모두 포함된 투구만 유지합니다.

    - top_pitchers: 투구한 공 개수 기준 상위 N명 투수
    - top_batters: 본 투구 개수(타석 투구수) 기준 상위 N명 타자

    Args:
        df: pitch_clean DataFrame (pitcher, batter 컬럼 필요)
        top_pitchers: 투수 상위 N명 (None이면 필터 안 함)
        top_batters: 타자 상위 N명 (None이면 필터 안 함)

    Returns:
        필터링된 DataFrame
    """
    if top_pitchers is None and top_batters is None:
        return df

    before_shape = df.shape

    # 전체 데이터 기준 상위 N명 계산 (필터 순서와 무관하게 동일한 결과)
    top_pitcher_ids: set | None = None
    top_batter_ids: set | None = None

    if top_pitchers is not None:
        pitcher_counts = df.groupby("pitcher").size().sort_values(ascending=False)
        top_pitcher_ids = set(pitcher_counts.head(top_pitchers).index.tolist())
        log(f"  → 투수 상위 {top_pitchers}명: {len(top_pitcher_ids)}명 선택")

    if top_batters is not None:
        batter_counts = df.groupby("batter").size().sort_values(ascending=False)
        top_batter_ids = set(batter_counts.head(top_batters).index.tolist())
        log(f"  → 타자 상위 {top_batters}명: {len(top_batter_ids)}명 선택")

    # 두 조건 교집합: 둘 다 포함된 투구만 유지
    if top_pitcher_ids is not None:
        df = df[df["pitcher"].isin(top_pitcher_ids)]
    if top_batter_ids is not None:
        df = df[df["batter"].isin(top_batter_ids)]

    log(f"  → 필터 결과: {before_shape[0]:,} → {len(df):,} 행")
    return df


def main() -> None:
    args = parse_args()
    start, end = args.start, args.end
    tag = f"{start}_to_{end}"

    project_root = Path(args.project_root).resolve() if args.project_root else None
    paths = get_paths(project_root)

    raw_csv = paths.raw_csv_range(start, end)
    out_clean = paths.processed_pitch_clean_range(start, end)
    out_embed = paths.processed_pitch_umap_cluster_range(start, end)
    out_pitcher_prof = paths.processed_pitcher_profiles_range(start, end)
    out_batter_prof = paths.processed_batter_profiles_range(start, end)
    out_pitch_level = paths.processed_matchup_pitch_level_range(start, end)
    out_pair_level = paths.processed_matchup_pair_level_range(start, end)
    out_summary = paths.processed_dir / f"pitcher_cluster_summary_{tag}.csv"

    log(f"[기간] {start} ~ {end}")

    # 00 fetch: Statcast 데이터 수집
    if (not args.skip_fetch) or (not raw_csv.exists()):
        log("00 fetch: Statcast 데이터 수집 중...")
        from src.fetch import fetch_statcast_by_date, FetchConfig
        df_raw = fetch_statcast_by_date(start, end, FetchConfig())
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        log(f"00 fetch 완료: {raw_csv} (shape={df_raw.shape})")
    else:
        log(f"00 fetch 생략: 기존 파일 사용 ({raw_csv})")
        df_raw = pd.read_csv(raw_csv, low_memory=False)

    # 01 preprocess: 컬럼 선택, 타입 정리, 파생 변수
    log("01 preprocess: 데이터 정제 중...")
    df_clean = preprocess_statcast(df_raw)

    # 모델 학습용: 투구수 상위 N명만 포함 (--top-pitchers, --top-batters 지정 시)
    if args.top_pitchers is not None or args.top_batters is not None:
        log("01 filter: 투구수 상위 선수만 추출 중...")
        df_clean = filter_top_players(df_clean, args.top_pitchers, args.top_batters)

    save_parquet(df_clean, out_clean)
    log(f"01 preprocess 완료: {out_clean} (shape={df_clean.shape})")

    # 02 embedding: 투수별 UMAP + HDBSCAN 클러스터링
    if not args.skip_embedding:
        log("02 embedding: 투수별 UMAP·클러스터링 진행 중...")
        cfg = EmbeddingConfig()
        df_out, summary = run_pitcher_umap_cluster(df_clean, cfg)
        save_parquet(df_out, out_embed)
        summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
        log(f"02 embedding 완료: {out_embed} (shape={df_out.shape})")
    else:
        log("02 embedding 생략: 기존 파일 사용")

    # 03 profiles: 투수/타자 프로필 생성
    if not args.skip_profiles:
        log("03 profiles: 투수·타자 프로필 생성 중...")
        df_emb = read_parquet(out_embed)
        summary = pd.read_csv(out_summary)
        pitcher_profiles = build_pitcher_profiles(df_emb, summary)
        batter_profiles = build_batter_profiles(df_emb)
        save_parquet(pitcher_profiles, out_pitcher_prof)
        save_parquet(batter_profiles, out_batter_prof)
        log(f"03 profiles 완료: pitcher={out_pitcher_prof}, batter={out_batter_prof}")
    else:
        log("03 profiles 생략")

    # 04 matchup: 투수-타자 매치업 테이블 생성
    if not args.skip_matchup:
        log("04 matchup: 매치업 테이블 생성 중...")
        df_emb = read_parquet(out_embed)
        pitcher_profiles = read_parquet(out_pitcher_prof)
        batter_profiles = read_parquet(out_batter_prof)
        pitch_level, pair_level, _ = build_matchup_tables(
            df_emb, pitcher_profiles, batter_profiles, MatchupConfig(topk=5)
        )
        save_parquet(pitch_level, out_pitch_level)
        save_parquet(pair_level, out_pair_level)
        log(f"04 matchup 완료: pitch_level={out_pitch_level}, pair_level={out_pair_level}")
    else:
        log("04 matchup 생략")

    log("파이프라인 완료")


if __name__ == "__main__":
    main()