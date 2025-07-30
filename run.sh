#!/bin/bash

docker build -t omni_api .
docker run --env-file .env omni_api