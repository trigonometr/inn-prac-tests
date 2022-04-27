FROM python:latest

WORKDIR /bot/

COPY requirements.txt /bot/requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY . /bot/
