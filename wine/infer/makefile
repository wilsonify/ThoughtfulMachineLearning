

base:
	docker build -f Dockerfile-base -t tml-wine-infer-base:latest .

builder:
	docker build -f Dockerfile-builder -t tml-wine-infer-builder:latest .

image:
	docker build -f Dockerfile -t tml-wine-infer:latest .