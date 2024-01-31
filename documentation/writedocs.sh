#!/bin/bash

sphinx-autobuild docs/source docs/build/html --open-browser --port 0 --watch $(dirname "$(pwd)"/bayesian_lora)
