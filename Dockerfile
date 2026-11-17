FROM python:3.14-alpine

# unrar serves games kept in RAR archives. Alpine packages none that can decompress RAR, so
# this is RARLAB's unrar built for musl (github.com/EDM115/unrar-alpine), pinned by checksum.
ARG UNRAR_VERSION=7.3.2
ARG UNRAR_SHA256=1265955fd5c63bb7632aef1a9256722f68c469ab3b175e0534d887f00ed0a371
RUN apk update && apk add --no-cache bash sudo libstdc++ libgcc \
    && wget -q -O /tmp/unrar "https://github.com/EDM115/unrar-alpine/releases/download/${UNRAR_VERSION}/unrar" \
    && echo "${UNRAR_SHA256}  /tmp/unrar" | sha256sum -c - \
    && install -m755 /tmp/unrar /usr/local/bin/unrar \
    && rm /tmp/unrar

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && rm /tmp/requirements.txt

RUN mkdir /app
COPY ./app /app
COPY ./docker/run.sh /app/run.sh

RUN mkdir -p /app/data

WORKDIR /app

ENTRYPOINT [ "/app/run.sh" ]
