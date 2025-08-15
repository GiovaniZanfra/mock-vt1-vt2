#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mock-vt1-vt2
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                       #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset - process raw data
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mock_vt1_vt2/dataset.py --config config/dataset.yaml --save

## Load dataset with custom config
.PHONY: data-custom
data-custom: requirements
	@echo "Usage: make data-custom CONFIG=path/to/config.yaml"
	$(PYTHON_INTERPRETER) mock_vt1_vt2/dataset.py --config $(CONFIG) --save

## Load dataset without saving (for testing)
.PHONY: data-test
data-test: requirements
	$(PYTHON_INTERPRETER) mock_vt1_vt2/dataset.py --config config/dataset.yaml

## Load dataset with specific protocols
.PHONY: data-vt1
data-vt1: requirements
	@echo "Loading VT1 protocol only..."
	$(PYTHON_INTERPRETER) mock_vt1_vt2/dataset.py --config config/dataset.yaml --save --output-file vt1_dataset.parquet

## Load dataset with specific protocols
.PHONY: data-vt2
data-vt2: requirements
	@echo "Loading VT2 protocol only..."
	$(PYTHON_INTERPRETER) mock_vt1_vt2/dataset.py --config config/dataset.yaml --save --output-file vt2_dataset.parquet

## Create features from processed data
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) -m mock_vt1_vt2.features

## Train models
.PHONY: train
train: features
	$(PYTHON_INTERPRETER) -m mock_vt1_vt2.modeling.train

## Make predictions
.PHONY: predict
predict: train
	$(PYTHON_INTERPRETER) -m mock_vt1_vt2.modeling.predict

## Create visualizations
.PHONY: plots
plots: data
	$(PYTHON_INTERPRETER) mock_vt1_vt2/plots.py

## Run complete pipeline
.PHONY: pipeline
pipeline: requirements data features train predict plots
	@echo "Complete pipeline executed successfully!"

## Clean all generated data
.PHONY: clean_data
clean_data:
	rm -rf data/processed/*
	rm -rf data/interim/*
	rm -rf models/*
	rm -rf reports/figures/*

## Validate dataset configuration
.PHONY: validate-config
validate-config:
	@echo "Validating dataset configuration..."
	$(PYTHON_INTERPRETER) -c "from mock_vt1_vt2.dataset import _load_config; config = _load_config('config/dataset.yaml'); print('✅ Configuration loaded successfully'); print(f'Protocols: {config.get(\"protocols\", [])}'); print(f'HR columns: {config.get(\"hr_columns\", [])}'); print(f'Speed columns: {config.get(\"speed_columns\", [])}')"

## Show dataset statistics
.PHONY: data-stats
data-stats: data-test
	@echo "Dataset statistics displayed above"

## Create backup of current config
.PHONY: backup-config
backup-config:
	@cp config/dataset.yaml config/dataset.yaml.backup.$(shell date +%Y%m%d_%H%M%S)
	@echo "Configuration backed up with timestamp"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
