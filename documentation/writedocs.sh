#!/bin/bash

sphinx-autobuild documentation/source documentation/build/html --open-browser --port 0 --watch $(dirname "$(pwd)"/bayesian_lora)
