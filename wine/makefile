

base:
	docker build -f Dockerfile-base -t tml-wine-base:latest .

builder:
	docker build -f Dockerfile-builder -t tml-wine-builder:latest .

image:
	docker build -f Dockerfile -t tml-wine:latest .