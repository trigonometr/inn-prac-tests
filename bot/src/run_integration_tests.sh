#!/bin/bash

cd /bot/src/bot_app/db
echo "Running tests for database"
python3 -m pytest -v test_database.py
echo
