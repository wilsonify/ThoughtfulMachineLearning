FROM tml-wine-buildest:latest as build

FROM tml-wine-base:latest
COPY --from=build /lib/x86_64-linux-gnu/libm.so.6 /lib/x86_64-linux-gnu/libm.so.6
COPY --from=build /app/dist /app/dist
COPY --from=build /app/tml_wine/models /app/dist/tml_wine/tml_wine/models
WORKDIR /app/dist
EXPOSE 8000
CMD ["/app/dist/tml_wine/tml_wine_entrypoint"]
