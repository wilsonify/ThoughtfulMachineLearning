
flask:
	docker run --rm \
-u $(id -u):$(id -g) \
-v "$(shell pwd):/local" \
openapitools/openapi-generator-cli:latest generate \
-i /local/openapi.json \
-g python-flask \
-o /local/tml_wine_generated