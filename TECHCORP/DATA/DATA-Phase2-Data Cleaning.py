import pandas as pd
import unicodedata
import re
from pathlib import Path

print("="*80)
print("PHASE 2: NETTOYAGE DES DONNEES")
print("="*80)

# Chemins dynamiques
project_root = Path(__file__).parent.parent
medical_datase_path = project_root / "medical_datase" / "dialogues.parquet"
output_path = project_root / "medical_datase_clean.parquet"

# Charger les donnees
print("\nChargement des donnees...")
df = pd.read_parquet(medical_datase_path)
print(f"Donnees chargees: {len(df):,} lignes")

initial_count = len(df)

# ====== 1. SUPPRIMER LES DOUBLONS COMPLETS ======
print("\n" + "="*80)
print("1. SUPPRESSION DES DOUBLONS COMPLETS")
print("="*80)

duplicates_before = df.duplicated().sum()
print(f"Doublons detectes: {duplicates_before:,}")

df = df.drop_duplicates()
duplicates_after = df.duplicated().sum()
removed_duplicates = duplicates_before - duplicates_after

print(f"Doublons supprimes: {removed_duplicates:,}")
print(f"Lignes restantes: {len(df):,}")

# ====== 2. NORMALISER LE TEXTE ======
# Rendre le texte coherent:
# - Supprimer les accents (é → e)
# - Supprimer les espaces doubles/triples
# Pourquoi? Pour que le modele voit "cafe" et non "café" ou "café   " comme des mots differents
print("\n" + "="*80)
print("2. NORMALISATION DU TEXTE")
print("="*80)

def normalize_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    # Supprimer les accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    # Supprimer espaces multiples
    text = re.sub(r'\s+', ' ', text)
    # Trim espacess au debut/fin
    text = text.strip()
    return text

print("Normalisation en cours...")
df['Description'] = df['Description'].apply(normalize_text)
df['Patient'] = df['Patient'].apply(normalize_text)
df['Doctor'] = df['Doctor'].apply(normalize_text)
print("OK - Texte normalise")

# ====== 3. FILTRER LES OUTLIERS ======
# Supprimer les textes extremement longs ou tres courts
# Pourquoi? Les textes trop longs ralentissent le modele et causent des "Out of Memory"
# On garde 99% des donnees, on jette seulement le 1% le plus extreme
print("\n" + "="*80)
print("3. FILTRAGE DES OUTLIERS (99e percentile)")
print("="*80)

# Calculer les limites (99e percentile = gardons 99% des donnees)
desc_max = df['Description'].str.len().quantile(0.99)
patient_max = df['Patient'].str.len().quantile(0.99)
doctor_max = df['Doctor'].str.len().quantile(0.99)

print(f"\nLimites 99e percentile:")
print(f"  Description: {desc_max:.0f} caracteres")
print(f"  Patient: {patient_max:.0f} caracteres")
print(f"  Doctor: {doctor_max:.0f} caracteres")

# Supprimer les lignes qui depassent les limites
outliers_before = len(df)
df = df[(df['Description'].str.len() <= desc_max) &
        (df['Patient'].str.len() <= patient_max) &
        (df['Doctor'].str.len() <= doctor_max)]
outliers_removed = outliers_before - len(df)

print(f"\nOutliers supprimes: {outliers_removed:,}")
print(f"Lignes restantes: {len(df):,}")

# ====== 4. VERIFICATION FINALE ======
print("\n" + "="*80)
print("4. VERIFICATION FINALE")
print("="*80)

df = df.dropna()
print(f"Lignes apres suppression des nulls: {len(df):,}")

# ====== RESUME ======
print("\n" + "="*80)
print("RESUME DU NETTOYAGE")
print("="*80)

print(f"\nAvant nettoyage:  {initial_count:,} lignes")
print(f"Apres nettoyage:  {len(df):,} lignes")
print(f"Lignes supprimees: {initial_count - len(df):,}")
print(f"Reduction: {((initial_count - len(df)) / initial_count * 100):.2f}%")

# ====== SAUVEGARDER ======
print("\n" + "="*80)
print("SAUVEGARDE DU DATASET NETTOYE")
print("="*80)

df_clean = df[['Description', 'Patient', 'Doctor']].reset_index(drop=True)
df_clean.to_parquet(output_path, index=False)
print(f"OK - Fichier parquet: {output_path} ({len(df_clean):,} lignes)")
print("\nNettoyage termine avec succes!")
