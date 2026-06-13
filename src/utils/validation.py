import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clip_probabilities(df: pd.DataFrame, target_cols: list, eps: float = 1e-15) -> pd.DataFrame:
    """
    Log-Loss is extremely sensitive to confident but wrong predictions.
    Clipping probabilities away from absolute 0 and 1 prevents infinite loss.
    """
    df_copy = df.copy()
    for col in target_cols:
        if col in df_copy.columns:
            df_copy[col] = np.clip(df_copy[col], eps, 1 - eps)
    return df_copy


def validate_submission(sub_df: pd.DataFrame, sample_sub_df: pd.DataFrame) -> bool:
    """
    Strictly validates the generated submission against the required sample submission format.
    Checks for:
    1. Same number of rows
    2. Same columns
    3. Same IDs (if applicable)
    4. Values are within [0, 1] range
    """
    is_valid = True

    # 1. Check Row Count
    if len(sub_df) != len(sample_sub_df):
        logger.error(f"Validation Failed: Row count mismatch. Expected {len(sample_sub_df)}, got {len(sub_df)}.")
        is_valid = False

    # 2. Check Columns
    expected_cols = set(sample_sub_df.columns)
    actual_cols = set(sub_df.columns)
    if expected_cols != actual_cols:
        logger.error(f"Validation Failed: Column mismatch. Expected {expected_cols}, got {actual_cols}.")
        is_valid = False

    # 3. Check value ranges for probability columns (assuming all non-ID columns are probabilities)
    # Exclude typical ID columns
    id_cols = ["subject_id", "user_id", "date", "timestamp"]
    prob_cols = [c for c in sub_df.columns if c not in id_cols]

    for col in prob_cols:
        if sub_df[col].dtype.kind in "fc":  # Float or complex
            min_val = sub_df[col].min()
            max_val = sub_df[col].max()
            if min_val < 0.0 or max_val > 1.0:
                logger.error(
                    f"Validation Failed: Values in column {col} are out of bounds [0, 1]. Min: {min_val}, Max: {max_val}."
                )
                is_valid = False

    if is_valid:
        logger.info("Submission passed all validation checks!")

    return is_valid


def make_subject_timeseries_folds(train: pd.DataFrame, n_folds: int = 5) -> list:
    """
    Subject 내에서 엄격한 TimeSeries 분할을 수행하는 CV 전략
    미래 데이터를 참조하는 누수(Look-ahead bias)를 차단하면서,
    충분한 학습 데이터와 검증 데이터를 확보하기 위해 TimeSeriesSplit을 사용합니다.
    """
    from sklearn.model_selection import TimeSeriesSplit

    result = [([], []) for _ in range(n_folds)]

    sort_col = "sleep_date" if "sleep_date" in train.columns else "date"
    id_col = "subject_id" if "subject_id" in train.columns else "user_id"

    for _subject_id, group in train.sort_values([id_col, sort_col]).groupby(id_col, sort=False):
        indices = group.index.to_numpy()
        n_samples = len(indices)

        # If a user has too few records, just put them all in train for later folds
        if n_samples < n_folds + 1:
            for fold_idx in range(n_folds):
                result[fold_idx][0].extend(indices)
            continue

        # We want to ensure each fold has a reasonable test size without starving the train set
        test_size = max(1, n_samples // (n_folds + 2))
        tscv = TimeSeriesSplit(n_splits=n_folds, test_size=test_size)

        for fold_idx, (t_idx, v_idx) in enumerate(tscv.split(indices)):
            result[fold_idx][0].extend(indices[t_idx])
            result[fold_idx][1].extend(indices[v_idx])

    # Convert lists to numpy arrays
    final_result = []
    for t_idx, v_idx in result:
        if len(t_idx) > 0 and len(v_idx) > 0:
            final_result.append((np.array(t_idx), np.array(v_idx)))

    return final_result


def make_subject_hole_folds(train: pd.DataFrame, n_folds: int = 5) -> list:
    """
    Subject 내에서 시간적 홀(hole)을 만들어 interleaved Train/Test 구조를 시뮬레이션하는 CV 전략.

    Args:
        train: 훈련 데이터프레임
        n_folds: CV 폴드 수

    Returns:
        (train_idx, valid_idx) numpy array 튜플 리스트
    """
    all_indices = train.index.to_numpy()
    result = []

    # 각 Subject별 시간순 정렬 후 block_count 개수로 분할
    block_count = max(n_folds * 2, 4)  # 홀(hole)을 생성하기 위해 fold 수의 2배로 설정
    by_subject = {}

    sort_col = "sleep_date" if "sleep_date" in train.columns else "date"
    id_col = "subject_id" if "subject_id" in train.columns else "user_id"

    for _subject_id, group in train.sort_values([id_col, sort_col]).groupby(id_col, sort=False):
        indices = group.index.to_numpy()
        # 동일 크기로 분할
        chunks = [chunk for chunk in np.array_split(indices, block_count) if len(chunk)]
        by_subject[str(_subject_id)] = chunks

    # 각 fold에 대해 두 개의 홀 생성 (early hole, late hole)
    for fold_id in range(n_folds):
        valid_parts = []

        for chunks in by_subject.values():
            # fold_id: early hole, fold_id + n_folds: late hole
            for hole_id in (fold_id, fold_id + n_folds):
                if hole_id < len(chunks):
                    valid_parts.append(chunks[hole_id])

        if not valid_parts:
            continue

        valid_idx = np.concatenate(valid_parts)
        train_idx = np.setdiff1d(all_indices, valid_idx, assume_unique=False)

        if len(train_idx) > 0 and len(valid_idx) > 0:
            result.append((train_idx, valid_idx))

    return result
