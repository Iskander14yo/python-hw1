import pandas as pd


def find_duplicates(df: pd.DataFrame,
                    key_cols: list[str],
                    return_bool: bool) -> pd.DataFrame | bool:
    dups = df[df.duplicated(subset=key_cols, keep='first')]
    if return_bool:
        return not dups.empty
    return dups


def drop_duplicates(df: pd.DataFrame, key_cols: list[str]):
    return df.drop_duplicates(subset=key_cols, keep='first')


def find_missing_values(df: pd.DataFrame,
                        return_bool: bool) -> pd.DataFrame | bool:
    missing_data = df.loc[df.isna().any(axis=1), :]
    if return_bool:
        return not missing_data.empty
    return missing_data
