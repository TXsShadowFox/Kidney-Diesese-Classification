FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . /app
# the app only serves predictions, so fail the build rather than ship an image without a model
RUN test -f artifacts/training/model.h5 && test -f artifacts/training/classes.json \
    || (echo "No trained model in artifacts/training: run 'dvc repro' before building" && exit 1)
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
