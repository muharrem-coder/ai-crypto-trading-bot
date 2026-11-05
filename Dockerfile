# syntax=docker/dockerfile:1.7
# quant-core: tek imaj, iki aşama. İlk aşama bağımlılıkları kurar, ikinci aşama sadece çalışma için gerekeni taşır.

# ---------- 1) builder: bağımlılıkları bir venv'e kur ----------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app

# Normal makinede proxy yok, bu satır düz "pip install ." çalıştırır.
# Şirket / sandbox proxy'si gerekirse SADECE build sırasında verilir (imaja girmez):
#   docker build --build-arg HTTPS_PROXY=... --secret id=proxyca,src=ca-bundle.crt .
# Secret verilmediyse hiçbir şey değişmez.
RUN --mount=type=secret,id=proxyca,required=false \
    if [ -s /run/secrets/proxyca ]; then export PIP_CERT=/run/secrets/proxyca; fi; \
    pip install .

# Uygulama kodu /app/app altından çalışır (migrations/ ve static/ yolları bu konuma göre bulunur).
# Bu yüzden venv'deki kopyayı atıyoruz, bağımlılıklar kalıyor.
RUN pip uninstall -y quant-core


# ---------- 2) runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    QUANT_IN_DOCKER=1

RUN groupadd --system --gid 10001 quant \
 && useradd --system --uid 10001 --gid quant --home-dir /app --shell /usr/sbin/nologin quant

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=quant:quant app ./app
COPY --chown=quant:quant migrations ./migrations
COPY --chown=quant:quant static ./static

USER quant

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "app.api.server"]
