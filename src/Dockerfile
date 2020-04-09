FROM python:3.7-buster
#alpine

#RUN apk update && apk --no-cache add build-base linux-headers python-dev

WORKDIR /app
ADD requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

ADD . /app
