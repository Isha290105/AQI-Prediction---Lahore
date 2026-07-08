FROM python:3.11-slim

WORKDIR /code

# Install system packages (for any C-extensions like numpy, pandas, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose the port Hugging Face uses (7860)
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
