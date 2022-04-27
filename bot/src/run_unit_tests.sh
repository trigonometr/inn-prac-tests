#!/bin/bash

cd /bot/src/bot_app/utils
echo "Running tests for images"
python3 -m pytest -v test_images
echo
echo "Running tests for matcher"
python3 -m pytest -v test_matcher
echo
echo "Running tests for models"
python3 -m pytest -v test_models
echo
