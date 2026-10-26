.PHONY: install train app test docker

install:
	python -m pip install -r requirements-dev.txt

train:
	python -m churn.train

app:
	streamlit run app.py

test:
	python -m pytest -q

docker:
	docker build -t churn-signal .
	docker run --rm -p 8501:8501 churn-signal
