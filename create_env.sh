#!/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
	Linux*)		MACHINE="$(lsb_release -is)";;
	Darwin*)	MACHINE=mac;;
	CYGWIN*)	MACHINE=cygwin;;
	MINGW*)		MACHINE=mingw;;
	*)		MACHINE="UNKNOWN:${unameOut}"
esac

MACHINE=$(echo "${MACHINE}" | awk '{print tolower($0)}')

if [ "$MACHINE" == "mingw" ]; then
	SCRIPTDIR=Scripts
	PYTHON=python
else
	SCRIPTDIR=bin
	PYTHON=$(which python3)
fi

MYVENV="venv"
"${PYTHON}" -u -m venv "${MYVENV}"
# Activate the new environment
echo "Activating the virtual environment..."
source "${MYVENV}/${SCRIPTDIR}/activate"

# Install dependencies (using the pip from the venv)
echo "Installing requirements..."
pip install -r requirements.txt

echo "Virtual environment '$MYVENV' created and activated."
echo "Requirements installed."
