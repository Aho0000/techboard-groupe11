"""
Utilitaires pour lire/ecrire les fichiers splittes
"""

import pandas as pd
import json
import glob
from pathlib import Path

def load_parquet_parts(directory, pattern):
    """
    Charge tous les fichiers parquet correspondant au pattern
    et les combine en un seul DataFrame
    """
    files = sorted(glob.glob(str(Path(directory) / f"{pattern}_part*.parquet")))

    if files:
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    else:
        single_file = Path(directory) / f"{pattern}.parquet"
        if single_file.exists():
            return pd.read_parquet(single_file)
        else:
            raise FileNotFoundError(f"No parquet files found for {pattern}")

def load_json_parts(directory, pattern):
    """
    Charge tous les fichiers JSON correspondant au pattern
    et les combine en une seule liste
    """
    files = sorted(glob.glob(str(Path(directory) / f"{pattern}_part*.json")))

    if files:
        all_data = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                all_data.extend(json.load(file))
        return all_data
    else:
        single_file = Path(directory) / f"{pattern}.json"
        if single_file.exists():
            with open(single_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No JSON files found for {pattern}")

def save_parquet_parts(df, output_path, max_size_mb=50):
    """
    Sauvegarde un DataFrame en plusieurs parts si trop gros
    """
    output_path = Path(output_path)
    total_size = len(df)

    # Estimation simple
    estimated_size_mb = (df.memory_usage(deep=True).sum() / 1024**2)

    if estimated_size_mb > max_size_mb:
        # Splitter
        num_parts = int(estimated_size_mb / max_size_mb) + 1
        chunk_size = len(df) // num_parts

        for i in range(num_parts):
            start = i * chunk_size
            end = (i+1) * chunk_size if i < num_parts-1 else len(df)
            chunk = df[start:end]

            part_file = output_path.parent / f"{output_path.stem}_part{i+1}.parquet"
            chunk.to_parquet(part_file, index=False)
    else:
        # Un seul fichier
        df.to_parquet(output_path, index=False)

def save_json_parts(data, output_path, max_size_mb=50):
    """
    Sauvegarde une liste JSON en plusieurs parts si trop gros
    """
    output_path = Path(output_path)

    # Estimation simple
    json_str = json.dumps(data[:min(10, len(data))], ensure_ascii=False)
    avg_item_size = len(json_str.encode('utf-8')) / min(10, len(data))
    estimated_size_mb = (len(data) * avg_item_size) / (1024**2)

    if estimated_size_mb > max_size_mb:
        # Splitter
        num_parts = int(estimated_size_mb / max_size_mb) + 1
        chunk_size = len(data) // num_parts

        for i in range(num_parts):
            start = i * chunk_size
            end = (i+1) * chunk_size if i < num_parts-1 else len(data)
            chunk = data[start:end]

            part_file = output_path.parent / f"{output_path.stem}_part{i+1}.json"
            with open(part_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
    else:
        # Un seul fichier
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
