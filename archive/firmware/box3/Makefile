# Convenience wrapper around idf.py for the ESP32-S3-BOX-3 voice client.
# All real work is in idf.py; this just bundles the README recipes and
# autodetects the serial port across Linux / macOS.
#
# Requires an ESP-IDF shell:  . ~/esp/esp-idf/export.sh

SHELL := /usr/bin/env bash

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
MODELS_DIR     := main/models
MWW_OUTPUT_DIR := ../../scripts/microwakeword/_work/output

BABEL_MODEL    := $(MODELS_DIR)/hey_babel.tflite
MARVIN_MODEL   := $(MODELS_DIR)/hey_marvin.tflite
MODELS         := $(BABEL_MODEL) $(MARVIN_MODEL)

# Stock microWakeWord model used to validate the firmware/transport before
# custom phrases are trained. See main/models/README.md.
STOCK_MODEL_URL := https://github.com/esphome/micro-wake-word-models/raw/refs/heads/main/models/hey_jarvis.tflite

# ------------------------------------------------------------------------------
# Serial port autodetect — override with `make PORT=/dev/...`
# ------------------------------------------------------------------------------
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  PORT ?= $(firstword $(wildcard /dev/cu.usbmodem*) /dev/cu.usbmodem-not-found)
else
  PORT ?= /dev/ttyACM0
endif

TARGET ?= esp32s3

# ------------------------------------------------------------------------------
# Flash dump — full 16 MB read-back. See README "Back up the factory firmware".
# ------------------------------------------------------------------------------
DUMP_FILE ?= box3-factory-backup.bin
DUMP_SIZE := 0x1000000
DUMP_BAUD ?= 460800

# ------------------------------------------------------------------------------
# ESP-IDF install — clones the pinned release and runs install.sh for $(TARGET).
# See README "Install ESP-IDF". After it completes, source the env in each new
# shell:  . $(IDF_DIR)/export.sh
# ------------------------------------------------------------------------------
IDF_VERSION ?= v5.1.4
IDF_DIR     ?= $(HOME)/esp/esp-idf
IDF_REPO    := https://github.com/espressif/esp-idf.git

# ------------------------------------------------------------------------------
# Phony targets
# ------------------------------------------------------------------------------
.PHONY: help install install-stock install-idf set-target build flash monitor \
        flash-monitor erase-flash menuconfig size clean fullclean reconfigure \
        dump check-idf check-port check-models install-os

help:
	@echo "firmware/box3 — ESP32-S3-BOX-3 voice client"
	@echo
	@echo "First-time setup:"
	@echo "  make install-idf    # one-time: clones ESP-IDF + installs toolchain"
	@echo "  . $(IDF_DIR)/export.sh"
	@echo "  make install        # or 'make install-stock' for the okay_nabu fallback"
	@echo "  make set-target"
	@echo "  make flash-monitor"
	@echo
	@echo "Targets:"
	@echo "  install-idf     clone ESP-IDF $(IDF_VERSION) to \$$(IDF_DIR) and run install.sh"
	@echo "  install         copy hey_babel/hey_marvin .tflite from scripts/microwakeword"
	@echo "  install-stock   download a stock microWakeWord model for both slots"
	@echo "  set-target      idf.py set-target $(TARGET)"
	@echo "  build           idf.py build"
	@echo "  flash           idf.py -p $(PORT) flash"
	@echo "  monitor         idf.py -p $(PORT) monitor"
	@echo "  flash-monitor   idf.py -p $(PORT) flash monitor    (most common)"
	@echo "  erase-flash     idf.py -p $(PORT) erase-flash"
	@echo "  dump            esptool.py read_flash 0..16MB -> \$$(DUMP_FILE)"
	@echo "  menuconfig      idf.py menuconfig"
	@echo "  reconfigure     idf.py reconfigure"
	@echo "  size            idf.py size"
	@echo "  clean           idf.py clean       (keeps CMake cache)"
	@echo "  fullclean       idf.py fullclean   (wipes build/)"
	@echo
	@echo "Variables (override on the command line):"
	@echo "  PORT=$(PORT)"
	@echo "  TARGET=$(TARGET)"
	@echo "  DUMP_FILE=$(DUMP_FILE)"
	@echo "  DUMP_BAUD=$(DUMP_BAUD)"
	@echo "  IDF_VERSION=$(IDF_VERSION)"
	@echo "  IDF_DIR=$(IDF_DIR)"

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
check-idf:
	@command -v idf.py >/dev/null || { \
	    echo "error: idf.py not on PATH. Source the ESP-IDF env first:"; \
	    echo "  . ~/esp/esp-idf/export.sh"; \
	    exit 1; }

check-port:
	@test -e "$(PORT)" || { \
	    echo "error: serial port '$(PORT)' not found."; \
	    echo "Plug in the BOX-3 over USB-C, or pass PORT=/dev/xxx."; \
	    exit 1; }

check-models:
	@test -f $(BABEL_MODEL) -a -f $(MARVIN_MODEL) || { \
	    echo "error: wake-word models missing under $(MODELS_DIR)/."; \
	    echo "Run one of:"; \
	    echo "  make install        # if you've trained them in scripts/microwakeword"; \
	    echo "  make install-stock  # to use okay_nabu for both slots (transport testing)"; \
	    exit 1; }

# ------------------------------------------------------------------------------
# ESP-IDF
# ------------------------------------------------------------------------------
install-idf:
	@if [ ! -d "$(IDF_DIR)" ]; then \
	    echo "Cloning ESP-IDF $(IDF_VERSION) into $(IDF_DIR)..."; \
	    mkdir -p "$(dir $(IDF_DIR))"; \
	    git clone -b $(IDF_VERSION) --recursive $(IDF_REPO) "$(IDF_DIR)"; \
	else \
	    echo "ESP-IDF dir already exists at $(IDF_DIR) — skipping clone."; \
	fi
	@echo "Running install.sh $(TARGET)..."
	cd "$(IDF_DIR)" && ./install.sh $(TARGET)
	@echo
	@echo "ESP-IDF installed. Source the env in each new shell:"
	@echo "  . $(IDF_DIR)/export.sh"

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
install: | $(MODELS_DIR)
	@missing=""; \
	for phrase in hey_babel hey_marvin; do \
	    src=$(MWW_OUTPUT_DIR)/$$phrase/$$phrase.tflite; \
	    if [ ! -f "$$src" ]; then missing="$$missing $$src"; fi; \
	done; \
	if [ -n "$$missing" ]; then \
	    echo "error: trained model(s) not found:"; \
	    for m in $$missing; do echo "  $$m"; done; \
	    echo "Train them first:  (cd ../../scripts/microwakeword && make all)"; \
	    echo "Or run:            make install-stock"; \
	    exit 1; \
	fi; \
	for phrase in hey_babel hey_marvin; do \
	    cp -v $(MWW_OUTPUT_DIR)/$$phrase/$$phrase.tflite $(MODELS_DIR)/$$phrase.tflite; \
	done
	@echo "models installed under $(MODELS_DIR)/"

$(MODELS_DIR):
	mkdir -p $@

install-stock: | $(MODELS_DIR)
	@echo "Downloading stock okay_nabu model into both wake slots..."
	curl -fL -o $(BABEL_MODEL) $(STOCK_MODEL_URL)
	cp $(BABEL_MODEL) $(MARVIN_MODEL)
	@echo "stock models installed; both phrases now respond to 'okay nabu'."

# ------------------------------------------------------------------------------
# Build / flash / monitor
# ------------------------------------------------------------------------------
set-target: check-idf
	idf.py set-target $(TARGET)

build: check-idf check-models
	idf.py build

flash: check-idf check-port check-models
	idf.py -p $(PORT) flash

monitor: check-idf check-port
	idf.py -p $(PORT) monitor

flash-monitor: check-idf check-port check-models
	idf.py -p $(PORT) flash monitor

erase-flash: check-idf check-port
	idf.py -p $(PORT) erase-flash

dump: check-idf check-port
	@if [ -e "$(DUMP_FILE)" ]; then \
	    echo "error: '$(DUMP_FILE)' already exists. Refusing to overwrite."; \
	    echo "Pass DUMP_FILE=<path> or remove the existing file."; \
	    exit 1; \
	fi
	@echo "Dumping 16 MB flash from $(PORT) to $(DUMP_FILE) (~3-4 min at $(DUMP_BAUD) baud)..."
	esptool.py --chip $(TARGET) -p $(PORT) -b $(DUMP_BAUD) \
	    read_flash 0x0 $(DUMP_SIZE) $(DUMP_FILE)
	@echo "done — store $(DUMP_FILE) outside the repo; it's a 16 MB binary and shouldn't be committed."

menuconfig: check-idf
	idf.py menuconfig

reconfigure: check-idf
	idf.py reconfigure

size: check-idf
	idf.py size

clean: check-idf
	idf.py clean

fullclean: check-idf
	idf.py fullclean

# after install: . ~/esp/esp-idf/export.sh
install-os:
	sudo apt install git wget flex bison gperf python3 python3-pip python3-venv \
    	cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
