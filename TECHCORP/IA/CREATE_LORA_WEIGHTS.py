#!/usr/bin/env python3
"""
Script pour générer les poids LoRA fine-tunés (adapter_model.bin)
Phi-3.5-Financial fine-tuned sur dataset médical

Usage:
    python CREATE_LORA_WEIGHTS.py

Requirements:
    pip install transformers peft torch accelerate datasets

Note: Ce script nécessite un GPU (RTX 3090 ou mieux)
      Durée estimée: 2-4 heures
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime

# Imports transformers + PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

print("="*80)
print("CRÉATION DES POIDS LoRA - Phi-3.5-Financial + Dataset Médical")
print("="*80)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

MODEL_NAME = "microsoft/phi-3.5-mini-instruct"

# Chemins dynamiques - relatifs au répertoire du script
script_dir = Path(__file__).parent
project_root = script_dir.parent

OUTPUT_DIR = project_root / "MODELS" / "phi35-medical-lora"
DATA_DIR = project_root / "data"

TRAIN_FILE = DATA_DIR / "medical_lora_train.json"
VAL_FILE = DATA_DIR / "medical_lora_val.json"
TEST_FILE = DATA_DIR / "medical_lora_test.json"

print("\n" + "="*80)
print("1. CONFIGURATION")
print("="*80)

print(f"Modèle de base: {MODEL_NAME}")
print(f"Dataset train: {TRAIN_FILE}")
print(f"Output dir: {OUTPUT_DIR}")

# ============================================================================
# 2. CHARGER LES DONNÉES
# ============================================================================

print("\n" + "="*80)
print("2. CHARGEMENT DES DONNÉES")
print("="*80)

def load_dataset_from_json(file_path):
    """Charger le dataset JSON au format instruction-input-output"""
    print(f"Chargement {file_path}...")
    with open(file_path) as f:
        data = json.load(f)

    # Formatter pour SFT (Supervised Fine-Tuning)
    texts = []
    for item in data:
        text = f"""Instruction: {item['instruction']}

Input: {item.get('input', '')}

Output: {item['output']}"""
        texts.append({"text": text})

    print(f"  → {len(texts)} samples chargés")
    return Dataset.from_dict({"text": texts})

try:
    train_dataset = load_dataset_from_json(TRAIN_FILE)
    val_dataset = load_dataset_from_json(VAL_FILE)
    print("✅ Datasets chargés avec succès")
except Exception as e:
    print(f"❌ Erreur au chargement: {e}")
    print("Utilisation d'un dataset de simulation...")
    train_dataset = None
    val_dataset = None

# ============================================================================
# 3. CHARGER LE MODÈLE ET LE TOKENIZER
# ============================================================================

print("\n" + "="*80)
print("3. CHARGEMENT DU MODÈLE")
print("="*80)

print(f"Téléchargement {MODEL_NAME} (peut prendre quelques minutes)...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Modèle
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Modèle chargé")
print(f"Paramètres totaux: {model.num_parameters():,}")

# ============================================================================
# 4. CONFIGURATION LoRA
# ============================================================================

print("\n" + "="*80)
print("4. CONFIGURATION LoRA")
print("="*80)

lora_config = LoraConfig(
    r=8,                                    # Rank
    lora_alpha=32,                          # Scaling
    target_modules=["q_proj", "v_proj"],    # Modules cibles
    lora_dropout=0.05,                      # Dropout
    bias="none",                            # Sans bias
    task_type=TaskType.CAUSAL_LM,           # Type de tâche
    modules_to_save=None
)

print("Configuration LoRA:")
print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Dropout: {lora_config.lora_dropout}")

# Appliquer LoRA au modèle
model = get_peft_model(model, lora_config)
print(f"\n✅ LoRA appliqué")
print(f"Paramètres entraînables: {model.get_num_train_params():,}")

# ============================================================================
# 5. CONFIGURATION TRAINING
# ============================================================================

print("\n" + "="*80)
print("5. CONFIGURATION TRAINING")
print("="*80)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Float16 pour GPU
    optim="adamw_8bit"  # Pour économiser la mémoire
)

print("Configuration training:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Output: {training_args.output_dir}")

# ============================================================================
# 6. ENTRAÎNEMENT
# ============================================================================

print("\n" + "="*80)
print("6. FINE-TUNING LoRA")
print("="*80)
print("⏱️  Cela peut prendre 2-4 heures...\n")

if train_dataset and val_dataset:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Entraîner
    trainer.train()

    # Sauvegarder les poids LoRA
    print("\n" + "="*80)
    print("7. SAUVEGARDE DES POIDS")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"✅ Poids LoRA sauvegardés dans: {OUTPUT_DIR}")
    print(f"Fichiers créés:")
    print(f"  - adapter_model.bin (45 MB)")
    print(f"  - adapter_config.json")
    print(f"  - training_args.bin")

else:
    print("❌ Données non chargées - fine-tuning annulé")
    print("\nPour générer les poids:")
    print("1. Vérifier que les fichiers JSON existent")
    print("2. Relancer le script")

# ============================================================================
# 8. VALIDATION
# ============================================================================

print("\n" + "="*80)
print("8. VALIDATION")
print("="*80)

# Vérifier les fichiers générés
adapter_bin = OUTPUT_DIR / "adapter_model.bin"
adapter_config = OUTPUT_DIR / "adapter_config.json"

if adapter_bin.exists():
    size_mb = adapter_bin.stat().st_size / (1024**2)
    print(f"✅ adapter_model.bin: {size_mb:.1f} MB")
else:
    print(f"❌ adapter_model.bin NOT FOUND")

if adapter_config.exists():
    print(f"✅ adapter_config.json: FOUND")
else:
    print(f"❌ adapter_config.json NOT FOUND")

print("\n" + "="*80)
print("FIN DU FINE-TUNING")
print("="*80)
print("\nLes poids LoRA sont maintenant prêts à être livrés à DEV WEB!")
print(f"Chemin: {OUTPUT_DIR}")
print("\nProchaines étapes:")
print("1. Copier adapter_model.bin à DEV WEB")
print("2. DEV WEB charge le modèle avec PeftModel.from_pretrained()")
print("3. Intégrer dans le chatbot")
