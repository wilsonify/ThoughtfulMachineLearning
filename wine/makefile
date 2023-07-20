ver_file = echo "1.0.0" > "build/$(1)" && date +%Y-%m-%d >> $(1)

all: clean run

clean:
	rm -rf logs.log
	rm -rf build
	rm -rf dist
	find . -type f -name "*.c" -exec rm {} \;
	find . -type f -name "*.o" -exec rm {} \;
	find . -type f -name "*.so" -exec rm {} \;

version.txt:
	mkdir -p build && $(call ver_file, $@)

base: version.txt
	docker build --progress plain -f Dockerfile-base -t tml-wine-base:latest . && $(call ver_file, $@)

builder: base
	docker build --progress plain -f Dockerfile-builder -t tml-wine-builder:latest . && $(call ver_file, $@)

buildest: builder
	docker build --progress plain -f Dockerfile-buildest -t tml-wine-buildest:latest . && $(call ver_file, $@)

image: buildest
	docker build --progress plain -f Dockerfile -t tml-wine:latest . && $(call ver_file, $@)

run: image
	docker run --rm --name tml-wine -P tml-wine:latest