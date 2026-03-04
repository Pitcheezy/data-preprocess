# src/fetch.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from linecache import cache
import pandas as pd


@dataclass(frozen=True)
class FetchConfig:
    use_cache: bool = True  # pybaseball 캐시 사용


def fetch_statcast_by_date(start_date: str, end_date: str, cfg: FetchConfig) -> pd.DataFrame:
    """
    start_date, end_date: 'YYYY-MM-DD'
    반환: Statcast pitch-by-pitch dataframe
    """
    # pybaseball은 설치되어 있어야 합니다.
    # uv: uv pip install pybaseball
    from pybaseball import statcast

    df = statcast(start_dt=start_date, end_dt=end_date)
    
    from pybaseball import cache
    cache.enable()

    # pybaseball 결과는 컬럼이 매우 많습니다. 01에서 SELECTED_COLS로 정리합니다.
    if df is None or len(df) == 0:
        raise ValueError(f"No statcast data returned for range {start_date} ~ {end_date}")

    return df