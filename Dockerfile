FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py features.py scoring.py ./
COPY data/motor_features.csv data/motor_features.csv
COPY outputs/fleet_ranking.csv outputs/fleet_ranking.csv

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
