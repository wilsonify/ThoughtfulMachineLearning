FROM tml-wine-infer-builder:latest as build
WORKDIR /app
ADD . /app

FROM tml-wine-infer-base:latest
EXPOSE 8000
CMD ["python", "predict_ws_from_other.py"]
