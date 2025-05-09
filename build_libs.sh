#!/bin/bash

# Default build type
BUILD_TYPE="Release"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--debug) BUILD_TYPE="Debug"; shift ;;
        -r|--release) BUILD_TYPE="Release"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Selected build type: ${BUILD_TYPE}"

unameOut="$(uname -s)"
case "${unameOut}" in
  Linux*)   MACHINE=linux;;
  Darwin*)	MACHINE=macos;;
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
# Activate the new environment
echo "Activating the virtual environment..."
source "${MYVENV}/${SCRIPTDIR}/activate"

cd externals/ma-libs
./cbuild.sh --build-type "${BUILD_TYPE}" --cmake-params "-DCPP_LIBNN=ON -DPYTHON_LIBNN=ON -DCPP_LIBCONFIG_LOADER=ON"

cd ../.. # Go back to project root to ensure consistent paths
mkdir -p externals/ma-libs/build/lunar_lander_cpp/"${BUILD_TYPE}"
cd externals/ma-libs/build/lunar_lander_cpp

if [[ -z "${NPROC}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    NPROC=$(sysctl -n hw.ncpu)
  else
    NPROC=$(nproc)
  fi
  (( NPROC = NPROC > 1 ? NPROC - 1 : 1 ))
fi

cd "${BUILD_TYPE}"

if [ "${MACHINE}" == "macos" ]; then
  cmake ../../../../.. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
elif [ "${MACHINE}" == "linux" ]; then
  cmake ../../../../.. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
else
  WINARCH="x64"
  if [[ $PROCESSOR_IDENTIFIER == *"ARM"* ]]; then WINARCH="ARM64"; fi
  cmake -A "${WINARCH}" ../../../../.. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
fi


if [ "${MACHINE}" == "cygwin" ] || [ "${MACHINE}" == "mingw" ]; then
  CMAKE_BUILD_PARALLEL_LEVEL=3 cmake --build ./ --config "${BUILD_TYPE}"
else
  cmake --build ./ --config "${BUILD_TYPE}" -j "${NPROC}"
fi

cd ../../..
