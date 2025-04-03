import os
import pandas as pd

def apply_transform(df: pd.DataFrame, script_path: str) -> pd.DataFrame:
    """
    Применяет пользовательский скрипт (Python-файл), который должен обновить DataFrame df.
    """
    if not os.path.exists(script_path):
        return df

    import numpy as np
    global_env = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
    }
    local_env = {"df": df}

    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, global_env, local_env)

    return local_env.get("df", df)


def get_script_list(script_folder: str) -> list:
    """
    Возвращает список .py-файлов в указанной папке со скриптами.
    """
    return [f for f in os.listdir(script_folder) if f.endswith(".py")]
