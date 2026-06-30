# TechCorp AI Chat

Assistant IA spécialisé finance pour TechCorp Industries, déployé dans le cadre du Challenge IA 7h.

## Le projet

Le projet répond à deux missions : rendre un modèle de langage spécialisé finance accessible via une interface de chat en production, et explorer le fine-tuning LoRA d'un modèle médical à titre expérimental (non déployé).

## Accès

- **Interface de chat** : https://techcorp.mpinfo.fr

## Architecture

Le navigateur ne parle qu'au front-end ; toute la communication avec le modèle reste interne au serveur, sans exposition directe de l'API au public.

- **Inférence** : Ollama (CPU), modèle `phi3.5-financial`
- **Reverse proxy** : Traefik (TLS automatique via Cloudflare DNS-01)
- **Interface** : React + Vite, servi par nginx, qui relaie les requêtes `/api/*` vers Ollama en interne (same-origin, zéro CORS exposé au navigateur)

## Statut & transparence

Le modèle "Phi-3.5-Financial" laissé par l'équipe précédente n'a pas pu être validé : les poids fine-tunés étaient absents des fichiers transmis, seules les configurations l'accompagnaient. Le service tourne donc sur le modèle de base officiel **microsoft/Phi-3.5-mini-instruct**, avec les paramètres d'inférence optimisés fournis par l'équipe IA (`temperature`, `top_p`, `top_k`, etc.).

Le fine-tuning médical (mission expérimentale) reste exploratoire et n'est pas déployé en production — voir les livrables des équipes IA et DATA.

## Stack technique

Docker Compose, Ollama, Traefik v3, nginx, React/Vite.
