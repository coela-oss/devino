SHELL := /bin/bash

.PHONY: install-poetry install-model-server uninstall-model-server

# Variables
OVMS_VERSION = v2025.0
OVMS_ARCHIVE = ovms_ubuntu24.tar.gz
OVMS_URL = https://github.com/openvinotoolkit/model_server/releases/download/$(OVMS_VERSION)/$(OVMS_ARCHIVE)
OVMS_DIR = $$(pwd)/ovms

help:	## Show available commands on this Makefile.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


install-poetry:	## Install Poetry From Offial script.
	@echo "Installing Poetry..."
	@curl -sSL https://install.python-poetry.org | python3
	@echo "Poetry installation completed."


install-model-server: ## Install Openvino GenAI Model Server(OVMS) from Github Latest Release.
	@echo "Installing OpenVINO Model Server (OVMS)..."
	@if [ ! -d "$(OVMS_DIR)" ]; then \
		echo "Downloading OVMS..."; \
		wget $(OVMS_URL); \
	else \
		echo "OVMS archive already exists, skipping download."; \
	fi
	@if [ ! -d "$(OVMS_DIR)" ]; then \
		echo "Extracting OVMS..."; \
		tar -xzf $(OVMS_ARCHIVE); \
	else \
		echo "OVMS directory already exists, skipping extraction."; \
	fi

	@if [ -f "$(OVMS_ARCHIVE)" ]; then \
		echo "Cleaning up temporary files..."; \
		rm -f $(OVMS_ARCHIVE); \
		echo "Clean Archive completed."; \
	fi

	@echo "Updating packages and installing dependencies..."
	@sudo apt update -y && sudo apt install -y libxml2 curl
	@echo "Setting environment variables..."
	@if ! grep -q "OVMS_LD_LIBRARY_PATH=$(OVMS_DIR)/lib" ./.env; then \
		echo "OVMS_LD_LIBRARY_PATH=$(OVMS_DIR)/lib" >> ./.env; \
	fi
	@if [ ! -f "/etc/ld.so.conf.d/ovms.conf" ]; then \
		echo "$(OVMS_DIR)/lib" | sudo tee /etc/ld.so.conf.d/ovms.conf; \
		sudo ldconfig; \
	fi
	@if ! grep -q "OVMS_PATH=$(OVMS_DIR)/bin" ./.env; then \
		echo "OVMS_PATH=$(OVMS_DIR)/bin" >> ./.env; \
	fi
	@echo "Installation completed. You need run "
	@echo "  source .env"
	@echo '  export PATH=$${OVMS_PATH}:$${PATH}'


uninstall-model-server: ## Uninstall Openvino GenAI Model Server (Only remove files).
	@echo "Uninstalling OpenVINO Model Server (OVMS)..."
	@if [ -d "$(OVMS_DIR)" ]; then \
		echo "Removing OVMS directory..."; \
		rm -rf $(OVMS_DIR); \
	else \
		echo "OVMS directory not found, skipping removal."; \
	fi
	@if [ -f "$(OVMS_ARCHIVE)" ]; then \
		echo "Removing OVMS archive..."; \
		rm -f $(OVMS_ARCHIVE); \
	else \
		echo "OVMS archive not found, skipping removal."; \
	fi
	@if [ -f "/etc/ld.so.conf.d/ovms.conf" ]; then \
		sudo rm -f /etc/ld.so.conf.d/ovms.conf; \
		sudo ldconfig; \
	fi

	@echo "Uninstallation completed."
