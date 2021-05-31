FROM python:3.6-stretch

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/splink-server/

COPY ./requirements.txt /usr/src/pca-server/requirements.txt

RUN pip3 install -r /usr/src/pca-server/requirements.txt
RUN pip3 install gunicorn

COPY docker-entrypoint.sh /entrypoint.sh

COPY . /usr/src/splink-server/

EXPOSE 8000

ENTRYPOINT ["sh", "/entrypoint.sh"]
