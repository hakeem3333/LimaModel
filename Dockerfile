FROM python:3.10-slim

WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile & Pipfile.lock first (for caching)
COPY Pipfile Pipfile.lock ./

# Install dependencies system-wide (not in virtualenv inside container)
RUN pipenv install --deploy --system

# Copy project
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
