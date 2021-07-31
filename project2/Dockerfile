FROM python 3.7.3-stretch
RUN mkdir /app
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
# run
CMD ["python3","apps.py"]
