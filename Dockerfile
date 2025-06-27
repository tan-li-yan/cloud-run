# FROM python:3.10

# WORKDIR /app

# # Copy requirements and install them
# COPY requirements.txt ./
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy all source code and credential files into the image
# COPY . .

# EXPOSE 8080

# CMD ["python", "main.py"]
FROM python:3.10

# Install system dependencies for building numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && apt-get clean

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]

