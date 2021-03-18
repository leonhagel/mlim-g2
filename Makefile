NAME?="mlim-g2"
VIRTUALENV?=./.env
PORT?=8888

help:
	@echo "Make targets:"
	@echo "  build          install dependencies and prepare environment located at ./.env"
	@echo "  build-lab      build + lab extensions"
	@echo "  freeze         view installed packages"
	@echo "  clean-cache    remove all files in the cache directory"
	@echo "  clean          clean-cache + remove *.pyc files and __pycache__ directory"
	@echo "  distclean      clean +  remove virtual environment"
	@echo "  lab            run jupyter lab (default port $(PORT))"
	@echo "  create-plots   create plots for the report"
	@echo "  create         create final output containing the optimized coupons"
	@echo "Check the Makefile for details"

build:
	mkdir cache; \
	mkdir data; \
	mkdir output; \
	mkdir src; \
	virtualenv  $(VIRTUALENV); \
	source $(VIRTUALENV)/bin/activate; \
	python -m pip install --upgrade pip; \
	python -m pip install -r requirements.txt;

build-lab: build
	source $(VIRTUALENV)/bin/activate; \
	jupyter labextension install jupyterlab-plotly@4.14.1; \
	jupyter serverextension enable --py jupyterlab_code_formatter

freeze:
	source $(VIRTUALENV)/bin/activate; \
	pip freeze

clean-cache:
	rm -rf ./cache; \
	mkdir cache; 

clean: clean-cache
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr

distclean: clean
	rm -rf $(VIRTUALENV)

lab:
	source $(VIRTUALENV)/bin/activate; \
	jupyter lab --port=$(PORT)

create-plots:
	source $(VIRTUALENV)/bin/activate; \
	python src/clustering.py

create:
	source $(VIRTUALENV)/bin/activate; \
	python src/create.py
