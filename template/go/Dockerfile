FROM golang:{{go_version}}-alpine AS builder

LABEL author="{{authors}}"

# Set necessary environmet variables needed for our image
ENV GO111MODULE=on
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64
