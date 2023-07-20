FROM tml-wine-buildest:latest as build

FROM tml-wine-base:latest
COPY --from=build /app/dist /app/dist
COPY --from=build /app/tml_wine/models /app/dist/tml_wine/models
WORKDIR /app/dist/__
EXPOSE 8000
CMD ["/app/dist/tml_wine/tml_wine"]
