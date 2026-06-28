FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install -r requirements.txt
COPY company_search_scores.csv .
COPY company_index.csv .
COPY company_scores_full.csv .
COPY ticker_names.csv .
COPY api/main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
