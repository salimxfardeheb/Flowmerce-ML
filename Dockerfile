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

# Utilisateur non-root (sécurité : si l'API est compromise,
# l'attaquant n'a pas les droits root dans le container)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copier UNIQUEMENT les packages installés depuis le builder
# (pas pip, pas les fichiers temporaires de build)
COPY --from=builder /install /usr/local

# Copier le code source avec les bons droits
COPY --chown=appuser:appuser . .

EXPOSE 8000

# Docker vérifie automatiquement que l'API répond toutes les 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Basculer sur l'utilisateur non-root AVANT le CMD
USER appuser

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]