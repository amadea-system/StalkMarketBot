FROM python:3.7-buster
#alpine

#RUN apk update && apk --no-cache add build-base linux-headers python-dev

WORKDIR /app
ADD requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /app/cy_src
ADD cy_src/setupStonks.py /app/cy_src
ADD cy_src/fastStonks.pyx /app/cy_src

RUN pwd
RUN ls
RUN python3 ./setupStonks.py build_ext --inplace

WORKDIR /app

ADD . /app


