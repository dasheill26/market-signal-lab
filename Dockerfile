# Market Signal Lab
#
# Multi-stage build: stage 1 builds the React frontend with Node, stage 2
# is the actual runtime image (Python + Flask) with only the built static
# files copied in - no Node.js, no npm, no frontend source in the final
# image. Single service, single port, matching the deployment pattern
# that's worked reliably for every other project in this portfolio.

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend, serving the built frontend ----
FROM python:3.12-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY --from=frontend-build /frontend/dist ./frontend_dist

EXPOSE 5003
# Shell form (not exec/JSON array form) deliberately - exec form can't expand
# environment variables, and this needs to respect whatever $PORT a hosting
# platform assigns (Render, and most PaaS platforms, assign their own port
# and expect the container to bind to it) rather than always using 5003.
ENTRYPOINT gunicorn --worker-class gthread --threads 4 --bind 0.0.0.0:${PORT:-5003} --timeout 60 run:app
