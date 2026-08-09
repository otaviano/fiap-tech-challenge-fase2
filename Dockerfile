# Imagem do serviço de inferência + interpretação (escalado pela IaC em infra/).
FROM python:3.11-slim

# Usuário não-root (boa prática de segurança).
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-serving.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src

USER appuser
EXPOSE 8000

# Healthcheck alinhado ao target group do ALB (/health).
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "diag_opt.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
