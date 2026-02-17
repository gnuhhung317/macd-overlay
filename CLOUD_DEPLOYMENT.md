# ☁️ Cloud Deployment Guide

This guide explains how to run the MACD 3-Stage ML Bot and Monitor on any cloud provider (AWS, DigitalOcean, Google Cloud, etc.) using Docker.

## 🧰 Prerequisites
1.  A Cloud VPS (Ubuntu recommended, e.g., AWS EC2, DigitalOcean Droplet).
2.  Docker and Docker Compose installed.
3.  Your `bot_config.json` filled with API Keys (Binance/Bitget).

---

## 🚀 Deployment Steps (VPS)

### 1. Prepare your files
Copy the following files to your server (using `scp` or `git clone`):
*   `docker-compose.yml`
*   `bot_config.json`
*   `monitor_config.json`
*   `ml/` directory (contains models)
*   `data/` (optional, for existing history)

### 2. Configure API Keys
Make sure `bot_config.json` on the server has `dry_run: false` if you want to trade live, and your API keys are correct.

### 3. Launch with Docker Compose
Run the following command in the project root:

```bash
docker-compose up -d
```

This will start:
1.  **macd-bot**: The trading bot (`bot.main`).
2.  **macd-monitor**: The API and background monitor.
3.  **signal-dashboard**: The Streamlit visualization (accessible via browser).

### 4. Verify Services
Check if everything is running:

```bash
docker-compose ps
```

View bot logs:
```bash
docker-compose logs -f macd-bot
```

### 5. Access the Dashboard
Open your browser and navigate to:
`http://YOUR_VPS_IP:8501`

---

## 🛠 Advanced Deployment

### Running behind Nginx (SSL)
If you want to access the dashboard securely via HTTPS, use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name trade.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Resource Management
The ML models can be memory-intensive. For small VPS (1GB RAM), ensure you have a swap file enabled or limit container memory in `docker-compose.yml`:

```yaml
    deploy:
      resources:
        limits:
          memory: 1G
```

---

## 🆘 Troubleshooting
*   **Port 8501 locked?**: Check firewall rules (Security Groups in AWS) to allow TCP 8501.
*   **Volume Permissions**: If the bot can't write to `data/`, run `chmod -R 777 data`.
*   **API Errors**: Ensure your server's clock is synced (`ntpdate` or automatic cloud sync).
