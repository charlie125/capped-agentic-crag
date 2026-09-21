FROM python:3.14-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 15076

CMD [ "python", "manage.py", "runserver", "0.0.0.0:15076" ]