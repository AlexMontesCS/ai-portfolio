# Deployment Guide

This guide covers how to deploy the RAG Chatbot application in different environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Environment Variables](#environment-variables)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)

## Local Development

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Create a virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the backend**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Production Deployment

### Backend Production Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set production environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   export PINECONE_ENVIRONMENT="your-pinecone-environment"
   export VECTOR_STORE_TYPE="pinecone"  # or faiss
   ```

3. **Run with Gunicorn**:
   ```bash
   pip install gunicorn
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

### Frontend Production Build

1. **Build the application**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Serve with nginx or Apache**:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           root /path/to/frontend/build;
           index index.html;
           try_files $uri $uri/ /index.html;
       }
       
       location /api/ {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## Environment Variables

### Backend Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and chat | Yes | - |
| `PINECONE_API_KEY` | Pinecone API key (if using Pinecone) | No | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment | No | - |
| `PINECONE_INDEX_NAME` | Pinecone index name | No | `chatbot-index` |
| `VECTOR_STORE_TYPE` | Vector store type (`faiss` or `pinecone`) | No | `faiss` |
| `FAISS_INDEX_PATH` | Path to FAISS index | No | `./vector_store/faiss_index` |
| `UPLOAD_DIR` | Directory for uploaded files | No | `./uploads` |
| `MAX_FILE_SIZE` | Maximum file size in bytes | No | `10485760` |
| `CHUNK_SIZE` | Text chunk size for processing | No | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | No | `200` |
| `CORS_ORIGINS` | Allowed CORS origins | No | `["http://localhost:3000"]` |

### Frontend Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `REACT_APP_API_URL` | Backend API URL | No | `http://localhost:8000` |

## Docker Deployment

### Docker Compose Setup

1. **Create docker-compose.yml**:
   ```yaml
   version: '3.8'
   
   services:
     backend:
       build: ./backend
       ports:
         - "8000:8000"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - PINECONE_API_KEY=${PINECONE_API_KEY}
         - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
         - VECTOR_STORE_TYPE=faiss
       volumes:
         - ./data:/app/data
   
     frontend:
       build: ./frontend
       ports:
         - "3000:3000"
       environment:
         - REACT_APP_API_URL=http://localhost:8000
       depends_on:
         - backend
   ```

2. **Create Dockerfiles**:

   **Backend Dockerfile** (`backend/Dockerfile`):
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000
   
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

   **Frontend Dockerfile** (`frontend/Dockerfile`):
   ```dockerfile
   FROM node:16-alpine as build
   
   WORKDIR /app
   
   COPY package*.json ./
   RUN npm install
   
   COPY . .
   RUN npm run build
   
   FROM nginx:alpine
   COPY --from=build /app/build /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/nginx.conf
   
   EXPOSE 80
   
   CMD ["nginx", "-g", "daemon off;"]
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

## Cloud Deployment

### AWS Deployment

1. **EC2 Instance**:
   - Launch an EC2 instance with Ubuntu 20.04+
   - Install Docker and Docker Compose
   - Clone the repository and run with Docker Compose

2. **ECS Deployment**:
   - Create ECR repositories for frontend and backend
   - Build and push Docker images
   - Create ECS task definitions and services

3. **Lambda + API Gateway** (for backend only):
   - Use AWS Lambda with Mangum for FastAPI
   - Deploy frontend to S3 + CloudFront

### Google Cloud Platform

1. **Cloud Run**:
   ```bash
   # Build and deploy backend
   gcloud builds submit --tag gcr.io/PROJECT_ID/rag-chatbot-backend
   gcloud run deploy --image gcr.io/PROJECT_ID/rag-chatbot-backend
   
   # Deploy frontend to Firebase Hosting
   npm run build
   firebase deploy
   ```

### Heroku Deployment

1. **Backend**:
   ```bash
   # Create Procfile
   echo "web: uvicorn app.main:app --host 0.0.0.0 --port \$PORT" > Procfile
   
   # Deploy
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your-key
   git push heroku main
   ```

2. **Frontend**:
   ```bash
   # Deploy to Netlify or Vercel
   npm run build
   # Follow platform-specific deployment steps
   ```

## Monitoring and Logging

### Application Monitoring

1. **Health Checks**:
   - Backend: `GET /health`
   - Monitor vector store status
   - Track document count

2. **Logging**:
   ```python
   # Add structured logging
   import structlog
   logger = structlog.get_logger()
   ```

3. **Metrics**:
   - Request/response times
   - Error rates
   - Document processing times
   - Vector store performance

### Security Considerations

1. **API Keys**:
   - Store in environment variables
   - Use secrets management services
   - Rotate keys regularly

2. **CORS**:
   - Configure appropriate origins
   - Use HTTPS in production

3. **File Upload**:
   - Validate file types and sizes
   - Scan for malware
   - Use separate storage bucket

4. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Check CORS_ORIGINS environment variable
   - Ensure frontend URL is included

2. **Vector Store Issues**:
   - Check FAISS index permissions
   - Verify Pinecone credentials

3. **Memory Issues**:
   - Limit concurrent file uploads
   - Optimize chunk sizes
   - Use streaming for large files

4. **Performance Issues**:
   - Enable caching
   - Optimize vector search parameters
   - Use CDN for frontend assets

### Debugging

1. **Enable Debug Logging**:
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **Check Container Logs**:
   ```bash
   docker-compose logs -f backend
   docker-compose logs -f frontend
   ```

3. **Health Check Endpoints**:
   ```bash
   curl http://localhost:8000/health
   ```
