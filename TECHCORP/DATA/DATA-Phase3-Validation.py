import pandas as pd
import json
import re
from pathlib import Path

print("="*80)
print("PHASE 3: VALIDATION DE QUALITE")
print("="*80)

# Chemins dynamiques
project_root = Path(__file__).parent.parent
clean_parquet_path = project_root / "medical_datase_clean.parquet"

# Charger le dataset nettoyé
print("\nChargement du dataset...")
df_clean = pd.read_parquet(clean_parquet_path)
print(f"Dataset chargé: {len(df_clean):,} lignes")

# ====== 1. COHERENCE Q/R ======
# Verifier que la reponse du docteur repond vraiment a la question
# Exemple: Q="What causes fever?" R="High fever is caused by..." = coherent
#          Q="What causes fever?" R="Hair loss treatment..." = incoherent
print("\n" + "="*80)
print("1. VALIDATION COHERENCE Q/R")
print("="*80)

def calculate_coherence(question, answer):
    if pd.isna(question) or pd.isna(answer):
        return 0
    q_text = str(question).lower()
    a_text = str(answer).lower()
    keywords = re.findall(r'\b[a-z]+\b', q_text)
    keywords = [kw for kw in keywords if len(kw) > 3]
    if not keywords:
        return 0.5
    matching_keywords = sum(1 for kw in keywords if kw in a_text)
    if matching_keywords > 0:
        return min(1.0, matching_keywords / len(keywords))
    else:
        if any(word in a_text for word in ['treatment', 'medicine', 'doctor', 'condition', 'cause', 'symptom']):
            return 0.5
        return 0

print("Calcul de la coherence Q/R...")
df_clean['coherence'] = df_clean.apply(
    lambda row: calculate_coherence(row['Description'], row['Doctor']),
    axis=1
)

high_coherence = (df_clean['coherence'] >= 0.7).sum()
medium_coherence = ((df_clean['coherence'] >= 0.4) & (df_clean['coherence'] < 0.7)).sum()
low_coherence = (df_clean['coherence'] < 0.4).sum()

print(f"Coherence HAUTE (>= 0.7): {high_coherence:,} ({high_coherence/len(df_clean)*100:.2f}%)")
print(f"Coherence MOYENNE (0.4-0.7): {medium_coherence:,} ({medium_coherence/len(df_clean)*100:.2f}%)")
print(f"Coherence BASSE (< 0.4): {low_coherence:,} ({low_coherence/len(df_clean)*100:.2f}%)")

coherence_score = (high_coherence / len(df_clean)) * 100
print(f"\nScore de coherence global: {coherence_score:.2f}%")

# ====== 2. LANGAGE MEDICAL ======
# Verifier qu'il n'y a pas de conseils medicaux dangereux ou incorrects
# Exemple: "never see a doctor" ou "dont take medicine" = dangereux!
print("\n" + "="*80)
print("2. VALIDATION LANGAGE MEDICAL")
print("="*80)

# Phrases rouges qui pourraient etre dangereuses
danger_phrases = ['never see a doctor', 'dont see a doctor', "don't see a doctor",
                  'never go to hospital', 'dont go to hospital', "don't go to hospital"]

dangerous_responses = []
for idx, row in df_clean.iterrows():
    response = str(row['Doctor']).lower()
    for danger in danger_phrases:
        if danger in response:
            dangerous_responses.append(idx)
            break

low_medical_vocab = []
for idx, row in df_clean.iterrows():
    response = str(row['Doctor']).lower()
    medical_words = sum(1 for w in response.split() if len(w) > 3)
    if medical_words < 5 and 'why' in str(row['Description']).lower():
        low_medical_vocab.append(idx)

print(f"Reponses potentiellement dangereuses: {len(dangerous_responses)}")
print(f"Reponses avec vocabulaire medical tres faible: {len(low_medical_vocab)}")

medical_quality_score = ((len(df_clean) - len(dangerous_responses) - len(low_medical_vocab)) / len(df_clean)) * 100
print(f"Score de qualité médicale: {medical_quality_score:.2f}%")

# ====== 3. FORMAT ======
print("\n" + "="*80)
print("3. VALIDATION FORMAT")
print("="*80)

well_formatted = 0
for idx, row in df_clean.iterrows():
    desc_str = str(row['Description']).lower().strip()
    if desc_str.startswith('q.') or desc_str.startswith('what') or \
       desc_str.startswith('why') or desc_str.startswith('how') or \
       desc_str.endswith('?'):
        well_formatted += 1

print(f"Lignes bien formatees: {well_formatted:,} ({well_formatted/len(df_clean)*100:.2f}%)")

format_quality_score = (well_formatted / len(df_clean)) * 100

# ====== SAUVEGARDER ======
print("\n" + "="*80)
print("SAUVEGARDE DES RESULTATS")
print("="*80)

results = {
    'coherence_score': float(coherence_score),
    'medical_quality_score': float(medical_quality_score),
    'format_quality_score': float(format_quality_score),
    'high_coherence': int(high_coherence),
    'medium_coherence': int(medium_coherence),
    'low_coherence': int(low_coherence),
    'dangerous_responses': int(len(dangerous_responses)),
    'low_medical_vocab': int(len(low_medical_vocab)),
    'well_formatted': int(well_formatted),
    'format_issues': int(len(df_clean) - well_formatted),
    'total_lines': int(len(df_clean))
}

with open('phase3_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("OK - phase3_validation_results.json sauvegarde")
print("Validation Phase 3 terminee!")
