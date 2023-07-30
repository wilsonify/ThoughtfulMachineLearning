FROM tml-wine-builder:latest as build
WORKDIR /app
ADD . /app
RUN python setup.py install
RUN pyinstaller \
    --onedir \
    --console \
    --specpath build \
    --name tml_wine_entrypoint \
    --distpath dist \
    --workpath build \
    --collect-all pydantic \
    --collect-all fastapi \
    --collect-all uvicorn \
    --collect-all sklearn \
    --collect-all passlib \
    tml_wine/__main__.py \

FROM tml-wine-base:latest
COPY --from=build /app/dist /app/dist
COPY --from=build /app/tml_wine/models /app/dist/tml_wine/tml_wine/models
WORKDIR /app/dist
EXPOSE 8000
CMD [ "/app/dist/tml_wine/tml_wine_entrypoint" ]
