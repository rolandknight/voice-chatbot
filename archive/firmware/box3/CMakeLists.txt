# Top-level ESP-IDF project file for the ESP32-S3-BOX-3 voice client.
# Builds against ESP-IDF v5.1+ with Espressif's esp-webrtc-solution and esp-sr.

cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS
    "$ENV{IDF_PATH}/examples/common_components"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(box3_voice_client)
