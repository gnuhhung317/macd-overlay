# 🐳 Docker Setup for MACD Monitor Dashboard

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the dashboard
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the dashboard
docker-compose down
```

Dashboard sẽ chạy tại: **http://localhost:8501**

### Option 2: Docker Only

```bash
# Build image
docker build -t macd-dashboard .

# Run container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/monitor_config.json:/app/monitor_config.json \
  --name macd-dashboard \
  macd-dashboard

# View logs
docker logs -f macd-dashboard

# Stop container
docker stop macd-dashboard
docker rm macd-dashboard
```

## 📋 Features

### 🔄 Auto-restart
- Container tự động khởi động lại nếu crash
- Sử dụng `restart: unless-stopped` policy

### 💾 Persistent Configuration
- File `monitor_config.json` được mount vào container
- Cấu hình của bạn được lưu ngay cả khi container restart

### 🏥 Health Check
- Tự động kiểm tra sức khỏe mỗi 30 giây
- Restart container nếu Streamlit không phản hồi

### 🌐 Network
- Sử dụng bridge network riêng (`macd-network`)
- Dễ dàng mở rộng với các services khác

## ⚙️ Configuration

### Environment Variables

Có thể thêm các biến môi trường trong `docker-compose.yml`:

```yaml
environment:
  - TZ=Asia/Ho_Chi_Minh
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Port Mapping

Thay đổi port trong `docker-compose.yml`:

```yaml
ports:
  - "8080:8501"  # Host:Container
```

Dashboard sẽ chạy tại: `http://localhost:8080`

### Volume Mounts

Mount thêm files/folders:

```yaml
volumes:
  - ./monitor_config.json:/app/monitor_config.json
  - ./config.py:/app/config.py
  - ./data:/app/data  # Thêm folder data
```

## 🔧 Advanced Usage

### Build with Custom Tag

```bash
docker build -t myusername/macd-dashboard:v1.0 .
```

### Push to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag macd-dashboard myusername/macd-dashboard:latest

# Push
docker push myusername/macd-dashboard:latest
```

### Run with Custom Config

```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/my_custom_config.json:/app/monitor_config.json \
  -e TZ=America/New_York \
  --name macd-dashboard \
  macd-dashboard
```

### Multi-platform Build

```bash
# Build for AMD64 and ARM64
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/macd-dashboard:latest \
  --push .
```

## 📊 Monitoring

### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Docker only
docker logs -f macd-dashboard

# Last 100 lines
docker logs --tail 100 macd-dashboard
```

### Check Health Status

```bash
# Docker Compose
docker-compose ps

# Docker only
docker inspect --format='{{.State.Health.Status}}' macd-dashboard
```

### Resource Usage

```bash
docker stats macd-dashboard
```

## 🛠️ Troubleshooting

### Container keeps restarting

```bash
# Check logs
docker-compose logs -f

# Check health
docker inspect --format='{{json .State.Health}}' macd-dashboard
```

### Cannot connect to dashboard

1. Kiểm tra container đang chạy:
   ```bash
   docker ps | grep macd-dashboard
   ```

2. Kiểm tra port mapping:
   ```bash
   docker port macd-dashboard
   ```

3. Kiểm tra logs:
   ```bash
   docker logs macd-dashboard
   ```

### Permission denied on volume

```bash
# Fix permissions
chmod 666 monitor_config.json

# Or run container with user
docker run --user $(id -u):$(id -g) ...
```

### Out of memory

Thêm memory limit trong `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 1G
    reservations:
      memory: 512M
```

## 🔐 Security

### Remove sensitive data from config

Sử dụng environment variables thay vì hardcode:

```yaml
environment:
  - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
  - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
```

Tạo file `.env`:

```env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Network Isolation

```yaml
networks:
  macd-network:
    driver: bridge
    internal: true  # Chặn truy cập internet
```

## 📦 Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml macd
```

### Using Kubernetes

Tạo deployment:

```bash
kubectl create deployment macd-dashboard --image=macd-dashboard:latest
kubectl expose deployment macd-dashboard --port=8501 --type=LoadBalancer
```

### Behind Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name macd.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🎯 Best Practices

1. **Always use volumes** cho persistent data
2. **Set resource limits** để tránh container chiếm hết tài nguyên
3. **Enable health checks** cho auto-restart
4. **Use .dockerignore** để giảm image size
5. **Separate sensitive data** từ codebase
6. **Tag your images** với version numbers
7. **Use multi-stage builds** nếu cần optimize image size

## 📝 Example Production Setup

```yaml
version: '3.8'

services:
  macd-dashboard:
    image: macd-dashboard:latest
    container_name: macd-dashboard
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./monitor_config.json:/app/monitor_config.json
      - ./logs:/app/logs
    environment:
      - TZ=Asia/Ho_Chi_Minh
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - macd-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  macd-network:
    driver: bridge
```

## 🆘 Support

Nếu gặp vấn đề, kiểm tra:

1. Logs: `docker-compose logs -f`
2. Health status: `docker-compose ps`
3. Resource usage: `docker stats`
4. Port availability: `netstat -an | grep 8501`

Happy monitoring! 🚀📊
