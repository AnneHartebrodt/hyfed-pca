#!/bin/bash

python3 manage.py makemigrations
python3 manage.py makemigrations hyfed_server pca_server
python3 manage.py migrate
gunicorn hyfed_server.wsgi --bind 0.0.0.0:8000 --timeout 1200 --workers 1
