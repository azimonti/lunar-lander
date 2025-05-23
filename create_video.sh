#!/bin/bash

BASHSCRIPTDIR="$(realpath "$(dirname "$0")")"

FPS=60
SAVE_PATH_REL="./data/img"

# --- Resolve absolute path for image directory ---
SAVE_PATH_ABS="$(realpath "${BASHSCRIPTDIR}/${SAVE_PATH_REL}")"

# --- Define ffmpeg parameters ---
NOW=$(date +'%Y%m%d_%H%M')
OUTPUT_VIDEO="${SAVE_PATH_ABS}/${NOW}_video.mp4"
INPUT_PATTERN="${SAVE_PATH_ABS}/frame_%06d.png"

# --- Check if image directory exists ---
if [ ! -d "${SAVE_PATH_ABS}" ]; then
    echo "Error: Image directory not found: ${SAVE_PATH_ABS}" >&2
    exit 1
fi

# --- Check if any input frames exist ---
shopt -s nullglob # Prevent literal pattern if no files match
FRAME_FILES=("${SAVE_PATH_ABS}"/frame_*.png)
shopt -u nullglob # Turn off nullglob

if [ ${#FRAME_FILES[@]} -eq 0 ]; then
    echo "Error: No input frames found matching pattern: ${INPUT_PATTERN}" >&2
    exit 1
fi

# --- Run ffmpeg ---
echo "Creating video:"
echo "  Input : ${INPUT_PATTERN}"
echo "  Output: ${OUTPUT_VIDEO}"
echo "  FPS   : ${FPS}"

ffmpeg -framerate "${FPS}" -i "${INPUT_PATTERN}" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -y "${OUTPUT_VIDEO}"

# Write 4K
# ffmpeg -framerate "${FPS}" -i "${INPUT_PATTERN}" -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -b:v 25M -y "${OUTPUT_VIDEO}"

echo "Video creation complete."

# concat multiple videos
# cd data/img
# ffmpeg -f concat -safe 0 -i <(for F in *_video.mp4; do echo "file '$PWD/$F'"; done) -c copy _video_concat.mp4
# cd ../..
