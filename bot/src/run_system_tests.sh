#!/bin/bash

echo "Do you want to proceed running system tests? y/n:"
read answer
if [ $answer == "y" ]
then
  cd /bot/src
  echo "Running tests for bot_app"
  python3 -m unittest discover -v test_bot_app
  echo
fi
echo "Finishing system testing procedure"
