# ═══════════════════════════════════════════════════════════════
#  Flowmerce — API Machine Learning (FastAPI + uvicorn)
#  Build multi-stage : builder → runtime
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  STAGE 1 : Builder — installe les dépendances dans un layer isolé
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ═══════════════════════════════════════════════════════════════
#  STAGE 2 : Runtime — image finale propre et légère
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# PYTHONUNBUFFERED : les logs uvicorn arrivent immédiatement
# dans `docker compose logs` au lieu d'être bufferisés.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1 : runtime OpenMP exigé par LightGBM (absent de python:*-slim).
# Sans lui, le chargement du modèle échoue sur « libgomp.so.1: cannot open
# shared object file ».
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (sécurité : si l'API est compromise,
# l'attaquant n'a pas les droits root dans le container)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copier UNIQUEMENT les packages installés depuis le builder
# (pas pip, pas les fichiers temporaires de build)
COPY --from=builder /install /usr/local

# Copier le code source, les artefacts ML (models/) et data/raw
# avec les bons droits. Voir .dockerignore pour ce qui est exclu.
COPY --chown=appuser:appuser . .

# Cache Hugging Face : les artefacts sont téléchargés depuis le Hub
# au démarrage (USE_HF_MODELS), l'utilisateur non-root doit pouvoir y écrire.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache

EXPOSE 8000

# Docker vérifie automatiquement que l'API répond toutes les 30s.
# start-period élevé : le chargement des artefacts ML (Hugging Face
# ou disque) précède le premier byte servi par uvicorn.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Basculer sur l'utilisateur non-root AVANT le CMD
USER appuser

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
