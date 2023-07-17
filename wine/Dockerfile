FROM tml-wine-builder:latest as build
WORKDIR /app
ADD . /app
RUN python setup.py install
RUN pyinstaller tml_wine/__main__.py

FROM tml-wine-base:latest
COPY --from=build /app/dist /app
EXPOSE 8000
CMD ["/app/dist/__main__"]
