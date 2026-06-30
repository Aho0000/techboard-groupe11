# Documentation de déploiement — Équipe INFRA

**Projet** : TechCorp AI Chat — Challenge IA 7h
**Auteurs** : Mathéo Pawelec, Clément Charry, Pablo Rey
**Rôle** : INFRA — Architecte du Système
**Mission** : Déploiement d'un serveur d'inférence pour le modèle Phi-3.5-Financial, accessible à l'équipe DEV WEB et sécurisé en bout de chaîne.

---

## 1. Choix technique : Ollama

Trois options étaient possibles : Ollama, Triton Inference Server, ou un serveur maison (FastAPI/vLLM).

Le choix s'est porté sur **Ollama**, pour trois raisons précises liées au contexte du projet :

- **Contrainte matérielle** : l'inférence tourne en CPU uniquement, sans GPU dédié. Triton est conçu pour des déploiements GPU avancés (backend TensorRT) ; même son backend Python, plus simple, demande une configuration de modèle séparée (`config.pbtxt`, structure de répertoire spécifique) qui n'apporte aucun bénéfice en l'absence de GPU.
- **Contrainte de temps** : sur un format de 7h, la rapidité de mise en place prime sur la flexibilité. Ollama expose une API REST complète (`/api/generate`, `/api/chat`) sans étape de conversion de format ni configuration de runtime additionnelle.
- **Gestion native de la quantization** : Ollama charge directement des modèles quantisés (4-bit/8-bit) et gère l'allocation mémoire CPU automatiquement, ce qui correspond à la piste technique recommandée dans le brief.

Un serveur maison aurait demandé d'écrire et de maintenir une couche API (gestion du streaming, du chargement du modèle, de la concurrence) pour un gain fonctionnel nul sur la durée du challenge.

---

## 2. Architecture du déploiement

```
Internet
   │
   ▼
Traefik (TLS, Cloudflare DNS-01 — resolver cfresolver)
   │
   ├── techcorp.mpinfo.fr        → conteneur "web" (nginx + build React/Vite)
   │                                   │
   │                                   ▼ (réseau Docker interne traefik-proxy)
   │                              proxy /api/* → http://ollama-phi3-financial:11434
   │
   └── techcorp-oa.mpinfo.fr     → conteneur "ollama-phi3-financial" (accès debug, restreint par IP)
```

**Point clé de l'architecture** : le navigateur de l'utilisateur final ne communique jamais directement avec le serveur d'inférence. Toutes les requêtes du chat passent par le front-end (`techcorp.mpinfo.fr`), dont le nginx interne relaie en HTTP simple vers Ollama via le réseau Docker `traefik-proxy` — un réseau bridge privé, jamais exposé à internet. Les deux conteneurs se résolvent entre eux par leur `container_name` grâce au DNS interne Docker (`ollama-phi3-financial`), sans passer par Traefik pour ce trajet interne.

Cette conception a évolué en cours de déploiement : la première version exposait l'API Ollama publiquement avec une restriction par IP (`ipallowlist`) pour les tests manuels de l'équipe. Une fois l'interface de chat destinée à être ouverte à n'importe quel visiteur (juges, autres équipes), cette allowlist serait devenue bloquante pour tout le monde sauf les 3 IP de test. La solution retenue a été de faire transiter le trafic réel par un proxy interne côté nginx, et de conserver le routeur public d'Ollama uniquement comme outil de débogage pour l'équipe, toujours protégé par l'allowlist.

### Composants

| Composant | Image / Build | Rôle | Réseau |
|---|---|---|---|
| `ollama-phi3-financial` | `ollama/ollama:latest` | Serveur d'inférence | `traefik-proxy` |
| `phi-finance-chat` | build local (Dockerfile React/Vite + nginx) | Front-end + proxy interne | `traefik-proxy` |
| `traefik` | `traefik:v3.6` | Reverse proxy, TLS, routing (stack existante) | `traefik-proxy` |

---

## 3. Configuration — docker-compose

### 3.1 Service Ollama

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-phi3-financial
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
      - ./models/phi3_financial:/import:ro
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_CONTEXT_LENGTH=4096
      - OLLAMA_NUM_THREAD=0
      - OLLAMA_ORIGINS=*
    networks:
      - traefik-proxy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ollama.rule=Host(`techcorp-oa.mpinfo.fr`)"
      - "traefik.http.routers.ollama.entrypoints=websecure"
      - "traefik.http.routers.ollama.tls=true"
      - "traefik.http.routers.ollama.tls.certresolver=cfresolver"
      - "traefik.http.middlewares.ollama-ipallow.ipallowlist.sourcerange=130.180.208.168/32,79.174.192.82/32,162.120.187.130/32"
      - "traefik.http.routers.ollama.middlewares=ollama-ipallow@docker,security-headers@docker"
      - "traefik.http.services.ollama.loadbalancer.server.port=11434"

volumes:
  ollama_data:

networks:
  traefik-proxy:
    external: true
    name: traefik-proxy
```

**Explication détaillée :**

- `volumes` — `ollama_data` est un volume nommé qui persiste les modèles téléchargés (`ollama pull`) entre les redémarrages du conteneur ; sans lui, chaque `up` repartirait d'un Ollama vide. `./models/phi3_financial:/import:ro` monte le dossier hérité de l'équipe précédente en lecture seule à l'intérieur du conteneur sous `/import` — c'est ce qui permet d'exécuter `ollama create` depuis un `Modelfile` placé dans ce dossier.
- `OLLAMA_HOST=0.0.0.0` — Ollama n'écoute par défaut que sur `127.0.0.1` ; sans cette variable, même Traefik ne pourrait pas l'atteindre depuis le réseau Docker.
- `OLLAMA_NUM_PARALLEL=1` et `OLLAMA_MAX_LOADED_MODELS=1` — en CPU-only, traiter plusieurs requêtes ou charger plusieurs modèles en mémoire simultanément risque l'OOM (le modèle 4-bit pèse environ 8.5 Go de RAM d'après `inference_config.json`). On force le traitement séquentiel pour rester stable au prix d'un peu de latence en cas de charge.
- `OLLAMA_CONTEXT_LENGTH=4096` — taille de fenêtre de contexte ; un compromis volontaire entre capacité de mémoire conversationnelle et consommation mémoire CPU.
- `OLLAMA_NUM_THREAD=0` — laisse Ollama détecter automatiquement le nombre de cœurs disponibles sur l'hôte.
- `OLLAMA_ORIGINS=*` — autorise les requêtes CORS de n'importe quelle origine. Risque accepté pour la durée du challenge ; n'a plus d'impact réel depuis que le trafic de production passe par le proxy interne nginx (origine unique, jamais de requête cross-origin réelle).
- `networks: traefik-proxy` — rejoint le réseau bridge **déjà créé par la stack Traefik existante** (`external: true`, nom exact `traefik-proxy`). C'est ce réseau que le provider Docker de Traefik scrute (`--providers.docker.network=traefik-proxy`) pour router le trafic.
- `traefik.enable=true` — obligatoire : la stack Traefik est configurée avec `--providers.docker.exposedbydefault=false`, donc aucun conteneur n'est routé par défaut sans ce label explicite.
- `routers.ollama.rule=Host(...)` — règle de routage : seules les requêtes avec l'en-tête `Host: techcorp-oa.mpinfo.fr` sont envoyées vers ce service.
- `entrypoints=websecure` — le routeur n'écoute que sur le point d'entrée TLS (port 443). Pas besoin de déclarer l'entrypoint `web` (port 80) séparément : la stack Traefik redirige déjà globalement tout le trafic `web` vers `websecure` au niveau de sa configuration d'entrypoint, avant même d'évaluer les règles de routage par service.
- `tls.certresolver=cfresolver` — déclenche la génération automatique d'un certificat Let's Encrypt via challenge DNS-01 chez Cloudflare. `cfresolver` est utilisé (et non `njresolver` ou `leresolver`) car `mpinfo.fr` est la zone gérée par le `CF_DNS_API_TOKEN` déjà configuré dans la stack Traefik — les deux autres resolvers sont réservés à d'autres domaines (teknoantirep.org, radiobigbro.org, resistance23.org pour `njresolver`).
- `ipallowlist.sourcerange` — restreint l'accès à une liste d'IP exactes en notation CIDR ; `/32` signifie "une seule adresse IP précise", par opposition à un sous-réseau plus large.
- `routers.ollama.middlewares=ollama-ipallow@docker,security-headers@docker` — **l'ordre compte** : les middlewares s'exécutent dans l'ordre de la liste. L'allowlist IP est évaluée en premier ; une IP non autorisée reçoit un `403` immédiatement, avant même que les en-têtes de sécurité ne soient appliqués.
- `loadbalancer.server.port=11434` — indique à Traefik le port interne du conteneur à cibler (le port natif d'Ollama), indépendamment de toute publication de port vers l'hôte (qu'on a justement supprimée — aucun `ports:` n'est déclaré, tout le trafic transite par le réseau Docker et Traefik).

### 3.2 Service Web (front-end + proxy)

```yaml
services:
  web:
    build: .
    image: phi-finance-chat
    container_name: phi-finance-chat
    env_file:
      - .env
    restart: unless-stopped
    networks:
      - traefik-proxy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.phi-finance-chat.rule=Host(`techcorp.mpinfo.fr`)"
      - "traefik.http.routers.phi-finance-chat.entrypoints=websecure"
      - "traefik.http.routers.phi-finance-chat.tls=true"
      - "traefik.http.routers.phi-finance-chat.tls.certresolver=cfresolver"
      - "traefik.http.routers.phi-finance-chat.middlewares=security-headers@docker"
      - "traefik.http.services.phi-finance-chat.loadbalancer.server.port=80"

networks:
  traefik-proxy:
    external: true
    name: traefik-proxy
```

**Explication détaillée :**

- `build: .` + `image: phi-finance-chat` — l'image est construite localement à partir du Dockerfile du dépôt DEV WEB (build React/Vite, servi en statique par nginx), plutôt que tirée d'un registry.
- `env_file: .env` — injecte la variable `INFERENCE_URL` dans l'environnement du conteneur **au runtime**. Cette variable n'est pas une variable Vite (`VITE_...` injectée au build du bundle JS) : elle est consommée par le script de démarrage du conteneur nginx, via `envsubst`, qui réécrit dynamiquement le fichier de configuration nginx (`proxy_pass ${INFERENCE_URL};`) avant de lancer le serveur. Sa valeur : `http://ollama-phi3-financial:11434` — l'adresse interne du conteneur Ollama sur le réseau Docker, jamais une URL publique.
- Pas de `ipallowlist` sur ce routeur, contrairement à celui d'Ollama : c'est volontaire, ce service est la surface publique destinée à tous les visiteurs (juges, équipe, démonstration).
- `routers.phi-finance-chat.rule=Host(techcorp.mpinfo.fr)` — domaine distinct de celui d'Ollama (`techcorp-oa.mpinfo.fr`), nécessitant son propre enregistrement DNS A/CNAME chez Cloudflare.
- `loadbalancer.server.port=80` — port d'écoute de nginx à l'intérieur du conteneur.

La sécurité de ce service repose sur deux autres couches plutôt que sur une restriction d'IP : la Content-Security-Policy du nginx (`connect-src 'self'`), qui empêche techniquement le navigateur d'appeler autre chose que son propre domaine, et le fait que l'URL réelle d'Ollama ne quitte jamais le serveur (elle n'apparaît dans aucun fichier JS envoyé au navigateur).

---

## 4. Sécurité — synthèse en trois couches

1. **Réseau** : aucun port de conteneur n'est publié directement sur l'hôte (pas de `ports:` exposés) ; tout transite par le réseau bridge privé `traefik-proxy` et par Traefik en frontal.
2. **Périmètre public** : TLS automatique (Let's Encrypt / Cloudflare DNS-01) sur les deux domaines, en-têtes de sécurité (`security-headers` : HSTS, anti-clickjacking, nosniff) appliqués à tous les routeurs, allowlist IP sur le routeur de debug Ollama.
3. **Application** : Content-Security-Policy stricte côté front-end, proxy interne empêchant toute exposition directe de l'API d'inférence aux visiteurs.

---

## 5. Optimisation CPU — paramètres d'inférence

Paramètres fournis par l'équipe IA (`inference_config.json`), intégrés via un `Modelfile` Ollama :

```
FROM phi3.5

PARAMETER temperature 0.6
PARAMETER top_p 0.85
PARAMETER top_k 50
PARAMETER repeat_penalty 1.0
PARAMETER num_predict 110

SYSTEM """Tu es un assistant spécialisé en finance et business pour TechCorp Industries."""
```

| Paramètre source | Valeur | Équivalent Ollama |
|---|---|---|
| `temperature` | 0.6 | `temperature` |
| `top_p` | 0.85 | `top_p` |
| `top_k` | 50 | `top_k` |
| `repetition_penalty` | 1.0 | `repeat_penalty` |
| `max_tokens` | 110 | `num_predict` |

Le champ `do_sample: true` du fichier source n'a pas d'équivalent à déclarer : Ollama échantillonne automatiquement dès que `temperature > 0`.

---

## 6. Statut du modèle — transparence

Le dossier `models/phi3_financial/` hérité de l'équipe précédente ne contenait que des fichiers de configuration (`inference_config.json`, `adapter_config.json`), aucun poids de modèle (`.gguf`, `.safetensors`). Vérification croisée : `base_model` et `base_model_name_or_path` pointent tous deux vers `microsoft/Phi-3.5-mini-instruct` — le modèle de base officiel, pas un fine-tune.

En l'absence de poids fine-tunés récupérables, le service tourne donc sur ce **modèle de base officiel** (tag Ollama `phi3.5`), avec les paramètres d'inférence optimisés ci-dessus appliqués sous le nom `phi3.5-financial`. Cette anomalie a été signalée à l'équipe CYBER pour investigation sur l'intégrité des livrables de l'équipe précédente.

---

## 7. Procédure de déploiement

```bash
# Prérequis DNS : enregistrements A/CNAME pour techcorp.mpinfo.fr et techcorp-oa.mpinfo.fr
# pointant vers l'IP publique du serveur (zone Cloudflare).

# 1. Pull du modèle de base
docker compose exec ollama ollama pull phi3.5

# 2. Création du modèle optimisé à partir du Modelfile
docker compose exec ollama ollama create phi3.5-financial -f /import/Modelfile

# 3. Déploiement des deux stacks
docker compose up -d --build        # stack ollama
docker compose up -d --build        # stack web (depuis son propre répertoire)

# 4. Vérification de bout en bout
curl -X POST https://techcorp.mpinfo.fr/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3.5-financial", "prompt": "test", "stream": false}'
```

---

## 8. Accès

- **Interface de chat (public)** : https://techcorp.mpinfo.fr
- **API d'inférence (debug, accès restreint par IP)** : https://techcorp-oa.mpinfo.fr
- **Nom de modèle pour intégration** : `phi3.5-financial`
