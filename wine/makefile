ver_file = echo "1.0.0" > $(1) && date +%Y-%m-%d >> $(1)

all: clean run

clean:
	rm -rf logs.log
	rm -rf build
	rm -rf dist
	find . -type f -name "*.c" -exec rm {} \;
	find . -type f -name "*.o" -exec rm {} \;
	find . -type f -name "*.so" -exec rm {} \;

build_dir:
	mkdir -p build

build/version.txt: build_dir
	$(call ver_file, $@)

build/base: build/version.txt
	docker build --progress plain -f Dockerfile-base -t tml-wine-base:latest . && $(call ver_file, $@)

build/builder: build/base
	docker build --progress plain -f Dockerfile-builder -t tml-wine-builder:latest . && $(call ver_file, $@)

build/buildest: build/builder
	docker build --progress plain -f Dockerfile-buildest -t tml-wine-buildest:latest . && $(call ver_file, $@)

build/image: build/buildest
	docker build --progress plain -f Dockerfile -t tml-wine:latest . && $(call ver_file, $@)

run: build/image
	docker run --rm --name tml-wine -p 32774:8000 tml-wine:latest

dist:
	docker create --rm --name tml-wine tml-wine:latest
	docker cp tml-wine:/app/dist .