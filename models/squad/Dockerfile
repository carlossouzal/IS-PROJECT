FROM python:3.10

WORKDIR /app

RUN pip install transformers pandas numpy torch

COPY . .

CMD ["python","release.py"]