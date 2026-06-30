import pandas as pd
import numpy as np
from collections import Counter
import re
from pathlib import Path

print("="*80)
print("PHASE 1: EXPLORATION & ANALYSE DES DONNEES")
print("="*80)

# Chemins dynamiques
project_root = Path(__file__).parent.parent
medical_datase_path = project_root / "medical_datase" / "dialogues.parquet"

# Charger les donnees
print("\nChargement des donnees...")
df = pd.read_parquet(medical_datase_path)
print(f"OK - {len(df)} lignes chargees\n")

# ====== STATISTIQUES GENERALES ======
print("="*80)
print("1. STATISTIQUES GENERALES")
print("="*80)
print(f"Nombre total de lignes: {len(df):,}")
print(f"Nombre de colonnes: {len(df.columns)}")
print(f"Colonnes: {list(df.columns)}")
print(f"Taille en memoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ====== VALEURS MANQUANTES ======
# Verifier s'il y a des colonnes vides - important pour la qualite
print("\n" + "="*80)
print("2. VALEURS MANQUANTES")
print("="*80)
missing = df.isnull().sum()
print(f"Description: {missing['Description']} valeurs manquantes")
print(f"Patient: {missing['Patient']} valeurs manquantes")
print(f"Doctor: {missing['Doctor']} valeurs manquantes")

# ====== DOUBLONS ======
# Detecter les lignes identiques et les questions repetees
# Important: les doublons complets n'ajoutent rien, on les supprimera
print("\n" + "="*80)
print("3. DOUBLONS")
print("="*80)
total_duplicates = df.duplicated().sum()
desc_duplicates = df['Description'].duplicated().sum()
print(f"Doublons (lignes entieres): {total_duplicates:,}")
print(f"Doublons (Description uniquement): {desc_duplicates:,}")

# ====== LONGUEUR DES TEXTES ======
print("\n" + "="*80)
print("4. ANALYSE DES LONGUEURS DE TEXTE (caracteres)")
print("="*80)

df['desc_len'] = df['Description'].str.len()
df['patient_len'] = df['Patient'].str.len()
df['doctor_len'] = df['Doctor'].str.len()

print("\nDescription:")
print(f"  Min: {df['desc_len'].min()}")
print(f"  Max: {df['desc_len'].max()}")
print(f"  Moyenne: {df['desc_len'].mean():.0f}")
print(f"  Median: {df['desc_len'].median():.0f}")

print("\nPatient:")
print(f"  Min: {df['patient_len'].min()}")
print(f"  Max: {df['patient_len'].max()}")
print(f"  Moyenne: {df['patient_len'].mean():.0f}")
print(f"  Median: {df['patient_len'].median():.0f}")

print("\nDoctor:")
print(f"  Min: {df['doctor_len'].min()}")
print(f"  Max: {df['doctor_len'].max()}")
print(f"  Moyenne: {df['doctor_len'].mean():.0f}")
print(f"  Median: {df['doctor_len'].median():.0f}")

# ====== PROBLEMES ======
print("\n" + "="*80)
print("5. PROBLEMES DETECTES")
print("="*80)

empty_desc = (df['Description'].str.len() == 0).sum()
empty_patient = (df['Patient'].str.len() == 0).sum()
empty_doctor = (df['Doctor'].str.len() == 0).sum()

very_short_desc = (df['Description'].str.len() < 10).sum()
very_short_doctor = (df['Doctor'].str.len() < 10).sum()

print(f"\nTextes vides:")
print(f"  Description: {empty_desc}")
print(f"  Patient: {empty_patient}")
print(f"  Doctor: {empty_doctor}")

print(f"\nTextes tres courts (< 10 caracteres):")
print(f"  Description: {very_short_desc}")
print(f"  Doctor: {very_short_doctor}")

print("\n" + "="*80)
print("EXPLORATION TERMINEE")
print("="*80)
