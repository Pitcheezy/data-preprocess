# main.py
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
    p = argparse.ArgumentParser()

    # season 모드(기존)
    p.add_argument("--season", type=int, default=None, help="Season year (e.g., 2025). If set, expects raw CSV exists.")

    # range 모드(햄이 원하는 것)
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")

    p.add_argument("--project-root", type=str, default=None, help="Optional project root path")

    # 실행 제어
    p.add_argument("--skip-fetch", action="store_true", help="Skip fetching even in range mode (use existing raw CSV)")
    p.add_argument("--skip-embedding", action="store_true", help="Skip step 02 embedding/clustering")
    p.add_argument("--skip-profiles", action="store_true", help="Skip step 03 profiles")
    p.add_argument("--skip-matchup", action="store_true", help="Skip step 04 matchup")

    return p.parse_args()


def validate_mode(args: argparse.Namespace) -> str:
    season_mode = args.season is not None
    range_mode = args.start is not None and args.end is not None

    if season_mode and range_mode:
        raise ValueError("Use either --season OR (--start and --end), not both.")
    if not season_mode and not range_mode:
        raise ValueError("You must provide --season OR (--start and --end).")
    return "season" if season_mode else "range"


def main() -> None:
    args = parse_args()
    mode = validate_mode(args)

    project_root = Path(args.project_root).resolve() if args.project_root else None
    paths = get_paths(project_root)

    if mode == "season":
        season = args.season
        raw_csv = paths.raw_csv(season)
        tag = f"{season}"
        out_clean = paths.processed_pitch_clean(season)
        out_embed = paths.processed_pitch_umap_cluster(season)
        out_pitcher_prof = paths.processed_pitcher_profiles(season)
        out_batter_prof = paths.processed_batter_profiles(season)
        out_pitch_level = paths.processed_matchup_pitch_level(season)
        out_pair_level = paths.processed_matchup_pair_level(season)
        out_summary = paths.processed_dir / f"pitcher_cluster_summary_{season}.csv"

        # season 모드는 "이미 raw csv가 있다" 전제
        if not raw_csv.exists():
            raise FileNotFoundError(f"Season mode expects raw CSV exists: {raw_csv}")

        log(f"[MODE=SEASON] raw={raw_csv}")

        df_raw = pd.read_csv(raw_csv, low_memory=False)

    else:
        start = args.start
        end = args.end
        tag = f"{start}_to_{end}"

        raw_csv = paths.raw_csv_range(start, end)
        out_clean = paths.processed_pitch_clean_range(start, end)
        out_embed = paths.processed_pitch_umap_cluster_range(start, end)
        out_pitcher_prof = paths.processed_pitcher_profiles_range(start, end)
        out_batter_prof = paths.processed_batter_profiles_range(start, end)
        out_pitch_level = paths.processed_matchup_pitch_level_range(start, end)
        out_pair_level = paths.processed_matchup_pair_level_range(start, end)
        out_summary = paths.processed_dir / f"pitcher_cluster_summary_{tag}.csv"

        log(f"[MODE=RANGE] start={start} end={end}")

        # fetch 단계
        if (not args.skip_fetch) or (not raw_csv.exists()):
            log(f"00 fetch 시작: {start}~{end}")
            from src.fetch import fetch_statcast_by_date, FetchConfig
            df_raw = fetch_statcast_by_date(start, end, FetchConfig())
            # raw csv 저장
            raw_csv.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
            log(f"00 fetch 완료: raw saved {raw_csv} shape={df_raw.shape}")
        else:
            log(f"00 fetch 스킵: existing raw CSV 사용 {raw_csv}")
            df_raw = pd.read_csv(raw_csv, low_memory=False)

    # 01 preprocess
    log(f"01 preprocess 시작 tag={tag}")
    df_clean = preprocess_statcast(df_raw)
    save_parquet(df_clean, out_clean)
    log(f"01 preprocess 완료: {out_clean} shape={df_clean.shape}")

    # 02 embedding
    if not args.skip_embedding:
        log("02 embedding 시작 (투수별 UMAP+clustering)")
        cfg = EmbeddingConfig()
        df_out, summary = run_pitcher_umap_cluster(df_clean, cfg)
        save_parquet(df_out, out_embed)
        summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
        log(f"02 embedding 완료: {out_embed} shape={df_out.shape}")
        log(f"02 summary 저장: {out_summary} shape={summary.shape}")
    else:
        log("02 embedding 스킵: 기존 파일 사용")

    # 03 profiles
    if not args.skip_profiles:
        log("03 profiles 시작")
        df_emb = read_parquet(out_embed)
        summary = pd.read_csv(out_summary)
        pitcher_profiles = build_pitcher_profiles(df_emb, summary)
        batter_profiles = build_batter_profiles(df_emb)

        save_parquet(pitcher_profiles, out_pitcher_prof)
        save_parquet(batter_profiles, out_batter_prof)
        log(f"03 pitcher_profiles 저장: {out_pitcher_prof} shape={pitcher_profiles.shape}")
        log(f"03 batter_profiles 저장: {out_batter_prof} shape={batter_profiles.shape}")
    else:
        log("03 profiles 스킵")

    # 04 matchup
    if not args.skip_matchup:
        log("04 matchup 시작")
        df_emb = read_parquet(out_embed)
        pitcher_profiles = read_parquet(out_pitcher_prof)
        batter_profiles = read_parquet(out_batter_prof)

        pitch_level, pair_level, _ = build_matchup_tables(
            df_emb, pitcher_profiles, batter_profiles, MatchupConfig(topk=5)
        )

        save_parquet(pitch_level, out_pitch_level)
        save_parquet(pair_level, out_pair_level)

        log(f"04 matchup_pitch_level 저장: {out_pitch_level} shape={pitch_level.shape}")
        log(f"04 matchup_pair_level 저장: {out_pair_level} shape={pair_level.shape}")
    else:
        log("04 matchup 스킵")

    log("전체 파이프라인 완료")


if __name__ == "__main__":
    main()