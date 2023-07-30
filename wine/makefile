ver_file = echo "1.0.0" > $(1) && date +%Y-%m-%d >> $(1)

all: clean run

clean:
	rm -rf build
	rm -rf dist
	find . -type f -name "*.c" -exec rm {} \;
	find . -type f -name "*.o" -exec rm {} \;
	find . -type f -name "*.so" -exec rm {} \;
	find . -type f -name "*.log" -exec rm {} \;

build_dir:
	mkdir -p build

build/version.txt: build_dir
	$(call ver_file, $@)

build/base.txt: build/version.txt
	docker build --progress plain -f Dockerfile-base -t tml-wine-base:latest . && $(call ver_file, $@)

build/builder.txt: build/base.txt
	docker build --progress plain -f Dockerfile-builder -t tml-wine-builder:latest . && $(call ver_file, $@)

build/buildest.txt: build/builder.txt
	docker build --progress plain -f Dockerfile-buildest -t tml-wine-buildest:latest . && $(call ver_file, $@)

build/image.txt: build/builder.txt
	docker build --progress plain -f Dockerfile -t tml-wine:latest . && $(call ver_file, $@)

run:
	docker run --rm --name tml-wine -p 32774:8000 tml-wine:latest

dist:
	docker rm tml-wine-tmp || true
	docker create --name tml-wine-tmp tml-wine:latest
	docker cp tml-wine-tmp:/app/dist .
	docker rm tml-wine-tmp || true