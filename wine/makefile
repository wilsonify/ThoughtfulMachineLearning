
clean:
	rm -rf build
	rm -rf dist
	find . -type f -name "*.c" -exec rm {} \;
	find . -type f -name "*.o" -exec rm {} \;
	find . -type f -name "*.so" -exec rm {} \;

base: clean
	docker build -f Dockerfile-base -t tml-wine-base:latest .

builder: base
	docker build -f Dockerfile-builder -t tml-wine-builder:latest .

buildest: builder
	docker build -f Dockerfile-buildest -t tml-wine-buildest:latest .

image: buildest
	docker build -f Dockerfile -t tml-wine:latest .

run: image
	docker run --rm --name tml-wine -P tml-wine:latest