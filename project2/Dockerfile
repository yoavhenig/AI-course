# specify the base image
FROM python:3.7.3-strech
# make dir
RUN mkdir /app
# set the work dir
WORKDIR /app
# copy files
COPY . .
# install python and req txt file
RUN python -m pip3 install --upgrade pip3
RUN pip3 install -r requirements.txt

# run
CMD ["python", "upload.py"]