import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

print("="*80)
print("PHASE 4: PREPARATION POUR FINE-TUNING LoRA")
print("="*80)

# Chemins dynamiques
project_root = Path(__file__).parent.parent
clean_parquet_path = project_root / "medical_datase_clean.parquet"
train_json_path = project_root / "medical_lora_train.json"
val_json_path = project_root / "medical_lora_val.json"
test_json_path = project_root / "medical_lora_test.json"
results_json_path = project_root / "phase4_lora_results.json"

# Charger le dataset nettoyé
print("\nChargement du dataset...")
df = pd.read_parquet(clean_parquet_path)
print(f"Dataset chargé: {len(df):,} lignes")

# ====== 1. CREER LES PAIRS ======
# Formater les donnees au format LoRA (instruction-input-output)
# instruction = la question du patient
# input = contexte/infos patient (optionnel)
# output = reponse du docteur
print("\n" + "="*80)
print("1. CREATION DES PAIRES POUR LoRA")
print("="*80)

print("Creation des paires instruction-input-output...")
pairs = []
for idx, row in df.iterrows():
    pair = {
        "instruction": str(row['Description']),  # Question
        "input": str(row['Patient']) if str(row['Patient']).strip() else "",  # Contexte (optionnel)
        "output": str(row['Doctor'])  # Reponse
    }
    pairs.append(pair)

print(f"Total paires creees: {len(pairs):,}")

# ====== 2. SPLIT ======
# Diviser les donnees en 3 parties:
# - Train (80%): pour apprendre
# - Validation (10%): pour verifier pendant l'apprentissage
# - Test (10%): pour evaluer le resultat final
print("\n" + "="*80)
print("2. SPLIT TRAIN/VAL/TEST")
print("="*80)

# Split 80/20, puis diviser les 20% restant en 50/50 (10/10)
train_data, temp_data = train_test_split(pairs, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"\nTrain set: {len(train_data):,} lignes (80%)")
print(f"Validation set: {len(val_data):,} lignes (10%)")
print(f"Test set: {len(test_data):,} lignes (10%)")

# ====== 3. TOKENS ======
# Verifier que les donnees ne depassent pas la limite GPU (2048 tokens)
# Pourquoi? Si un texte est trop long, le modele plante (Out of Memory)
# Estimation simple: 1 token ≈ 4 caracteres
print("\n" + "="*80)
print("3. ANALYSE DES TOKENS")
print("="*80)

def count_tokens(num_chars):
    # Estimation: 1 token ≈ 4 caracteres en moyenne
    return num_chars / 4

train_tokens = []
for pair in train_data:
    total_chars = len(pair['instruction']) + len(pair['input']) + len(pair['output'])
    tokens = count_tokens(total_chars)
    train_tokens.append(tokens)

avg_tokens = np.mean(train_tokens)
max_tokens = np.max(train_tokens)
min_tokens = np.min(train_tokens)

print(f"Tokens par paire (estimation):")
print(f"  Min: {min_tokens:.0f}")
print(f"  Moyenne: {avg_tokens:.0f}")
print(f"  Max: {max_tokens:.0f}")

gpu_compatible = sum(1 for t in train_tokens if t < 2048) / len(train_tokens) * 100
print(f"\nCompatibilité GPU (< 2048 tokens): {gpu_compatible:.2f}%")

# ====== 4. EXPORT JSON ======
print("\n" + "="*80)
print("4. EXPORT EN JSON POUR LoRA")
print("="*80)

print("Export des fichiers JSON...")
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
print(f"OK - {train_json_path} ({len(train_data):,} paires)")

with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)
print(f"OK - {val_json_path} ({len(val_data):,} paires)")

with open(test_json_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
print(f"OK - {test_json_path} ({len(test_data):,} paires)")

# ====== 5. STATS ======
print("\n" + "="*80)
print("5. STATISTIQUES FINALES")
print("="*80)

instruction_lengths = [len(pair['instruction']) for pair in train_data]
input_lengths = [len(pair['input']) for pair in train_data]
output_lengths = [len(pair['output']) for pair in train_data]

print(f"\nInstruction (Question):")
print(f"  Moyenne: {np.mean(instruction_lengths):.0f} caracteres")
print(f"  Min: {np.min(instruction_lengths):.0f}, Max: {np.max(instruction_lengths):.0f}")

print(f"\nInput (Patient context):")
print(f"  Moyenne: {np.mean(input_lengths):.0f} caracteres")
print(f"  Min: {np.min(input_lengths):.0f}, Max: {np.max(input_lengths):.0f}")

print(f"\nOutput (Doctor response):")
print(f"  Moyenne: {np.mean(output_lengths):.0f} caracteres")
print(f"  Min: {np.min(output_lengths):.0f}, Max: {np.max(output_lengths):.0f}")

# ====== RESULTATS ======
phase4_results = {
    "total_pairs": len(pairs),
    "train_size": len(train_data),
    "val_size": len(val_data),
    "test_size": len(test_data),
    "train_ratio": float(len(train_data) / len(pairs) * 100),
    "val_ratio": float(len(val_data) / len(pairs) * 100),
    "test_ratio": float(len(test_data) / len(pairs) * 100),
    "avg_tokens_per_pair": float(avg_tokens),
    "max_tokens_per_pair": float(max_tokens),
    "min_tokens_per_pair": float(min_tokens),
    "gpu_compatible_percentage": float(gpu_compatible),
    "instruction_avg_length": float(np.mean(instruction_lengths)),
    "input_avg_length": float(np.mean(input_lengths)),
    "output_avg_length": float(np.mean(output_lengths)),
    "instruction_max_length": int(np.max(instruction_lengths)),
    "input_max_length": int(np.max(input_lengths)),
    "output_max_length": int(np.max(output_lengths))
}

with open(results_json_path, 'w') as f:
    json.dump(phase4_results, f, indent=2)

print("\n" + "="*80)
print("RECAP PHASE 4")
print("="*80)
print(f"\nDataset pret pour LoRA:")
print(f"  - {len(pairs):,} paires total")
print(f"  - Train: {len(train_data):,} (80%)")
print(f"  - Val: {len(val_data):,} (10%)")
print(f"  - Test: {len(test_data):,} (10%)")
print(f"\nTokens: Min {min_tokens:.0f}, Avg {avg_tokens:.0f}, Max {max_tokens:.0f}")
print(f"GPU Compatible: {gpu_compatible:.2f}%")
print(f"\nPhase 4 terminee avec succes!")
