install:
	pip install -r requirements.txt

train:
	python main.py train

train-quick:
	python main.py train --quick

test:
	python -m pytest tests/ -q

evaluate:
	python main.py evaluate

clean:
	@echo "Cleaning up generated artifacts..."
	rm -f models/*.pkl models/*.skops models/*.json models/*.txt
	rm -f results/*.json config/best_params.json

.PHONY: install train train-quick test evaluate clean
