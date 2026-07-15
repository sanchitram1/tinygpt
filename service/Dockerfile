# Build from the repository root so the model bundle can be baked in
# (Plan 3 bundle-in-image artifact mode):
#
#   docker build -f service/Dockerfile -t tinygpt-chat .
#
# core/artifacts/ is gitignored; the bundle must exist in the local checkout
# used as the build context.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY service/pyproject.toml service/README.md /srv/service/
COPY service/tinygpt_service /srv/service/tinygpt_service
COPY core /srv/core
RUN pip install "/srv/service[model,tracing]"

# One verified artifact bundle: checkpoint + tokenizer + manifest.json. Override
# BUNDLE_SOURCE when the bundle is produced outside the recovered baseline path.
ARG BUNDLE_SOURCE=model-bundles/xlarge-plus
COPY ${BUNDLE_SOURCE}/ /srv/model-bundle/
ENV TINYGPT_BUNDLE_DIR=/srv/model-bundle
ENV PYTHONPATH=/srv

# Cloud Run injects PORT; default to 8080 for local runs.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn tinygpt_service.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
