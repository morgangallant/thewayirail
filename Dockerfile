# Build step.
FROM golang:1.17 as build
ADD . /src
WORKDIR /src/
RUN go get .
RUN go build -o rail .

# Run Step using Distroless.
FROM gcr.io/distroless/base
WORKDIR /mg
COPY --from=build /src/rail /mg/
ENTRYPOINT [ "/mg/rail" ]