"""
Statcast 데이터 수집 (pybaseball)

지정 기간(YYYY-MM-DD)의 MLB Statcast pitch-by-pitch 데이터를 수집합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class FetchConfig:
    """수집 설정 (향후 확장용)"""
    use_cache: bool = True  # pybaseball 캐시 사용


def fetch_statcast_by_date(start_date: str, end_date: str, cfg: FetchConfig) -> pd.DataFrame:
    """
    기간별 Statcast pitch-by-pitch 데이터 수집

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        cfg: 수집 설정

    Returns:
        Statcast pitch-by-pitch DataFrame (pybaseball 원본 컬럼)

    Raises:
        ValueError: 해당 기간에 데이터가 없을 경우
    """
    from pybaseball import statcast, cache

    if cfg.use_cache:
        cache.enable()
    df = statcast(start_dt=start_date, end_dt=end_date)

    if df is None or len(df) == 0:
        raise ValueError(f"해당 기간에 Statcast 데이터 없음: {start_date} ~ {end_date}")

    return df