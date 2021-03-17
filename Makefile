NAME?="mlim-g2"
VIRTUALENV?=./env
PORT?=8888

help:
	@echo "Make targets:"
	@echo "  build          install dependencies and prepare environment"
	@echo "  build-lab      build + lab extensions"
	@echo "  freeze         view installed packages"
	@echo "  clean          remove *.pyc files and __pycache__ directory"
	@echo "  distclean      remove virtual environment"
	@echo "  run            run jupyter lab (default port $(PORT))"
	@echo "Check the Makefile for details"

build:
	virtualenv  $(VIRTUALENV); \
	source $(VIRTUALENV)/bin/activate; \
	python3 -m pip install --upgrade pip; \
	python3 -m pip install -r requirements.txt;

build-lab: build
	source $(VIRTUALENV)/bin/activate; \
	jupyter labextension install jupyterlab-plotly@4.14.1; \
	jupyter serverextension enable --py jupyterlab_code_formatter
	python3 -m ipykernel install --user --name=$(NAME);

freeze:
	source $(VIRTUALENV)/bin/activate; \
	pip freeze > freeze.txt

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr

distclean: clean
	rm -rf $(VIRTUALENV)
	yes | jupyter kernelspec uninstall $(NAME)

run:
	source $(VIRTUALENV)/bin/activate; \
	jupyter lab --port=$(PORT)
