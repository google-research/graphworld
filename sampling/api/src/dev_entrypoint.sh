#!/bin/bash
DATABASE_URL="'postgres://synth-db:synth-db@synth-db:5432'"

until psql $DATABASE_URL -c '\l'; do
    >&2 echo "Synth DB is unavailable - sleeping."
    sleep 10s
done

echo "Connected to synth-db"

python3 /dreservoir/manage.py makemigrations
python3 /dreservoir/manage.py migrate
python3 manage.py runserver 8080
