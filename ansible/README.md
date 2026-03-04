# Quản lý hàng loạt Bot bằng Ansible + Git

Đây là cấu trúc thư mục giúp bạn quản lý hàng chục con bot trên các VPS micro siêu nhẹ.

## Luồng hoạt động:
1. Bạn `git push` code mới lên Github/Gitlab của bạn (nhánh `main`).
2. API Key, Secret (`credentials.json`) được lưu TRÊN MÁY BẠN ở thư mục `bots-configs`, KHÔNG push lên Github.
3. Kích chạy dòng lệnh Ansible, Ansible sẽ tự SSH vào các server, pull code từ Git về, sau đó copy file API Key tương ứng dán đè lên server, và khởi động lại Bot.

## Cấu trúc thư mục này:
- `inventory.ini`: Cấu hình danh sách Server (IP) và định nghĩa tên bot ở server đó.
- `deploy-bot.yml`: File kịch bản (Playbook) chạy mọi thứ.
- `templates/bot.service.j2`: File cấu hình Systemd siêu nhẹ để giữ bot luôn sống trên linux.
- `bots-configs/`: Chứa file `credentials.json` cho từng con bot (phân tách theo thư mục tên bot).

## Các bước thiết lập ban đầu:

1. Vào file `deploy-bot.yml` sửa biến `repo_url` thành link Git của bạn.
2. Sửa lại biến `bot_command` thành file python chính của bot (ví dụ `/usr/bin/python3 main_bot.py`).
3. Sửa file `inventory.ini` nhập IP VPS của bạn.
4. Đảm bảo cấu trúc file trong `bots-configs/bot_binance_01/credentials.json` khớp với format thật của bạn.

## Lệnh Deploy Cập Nhật Code:

Từ thư mục `ansible/` trên máy local, mở cửa sổ Terminal và gõ:

```bash
ansible-playbook -i inventory.ini deploy-bot.yml
```
*(Deploy bot sẽ tự động thực hiện **Health Check** sau 5 giây. Nếu bot crash ngay lúc bật, Script sẽ báo FAILED màu đỏ để bạn biết ngay lập tức).*

## Kiểm tra Trạng thái và Logs của toàn bộ Bot:
Để xem bot nào đang sống/chết và 10 dòng log mới nhất của tất cả các bot trên mọi server:
```bash
ansible-playbook -i inventory.ini check-status.yml
```

## Rollback (Quay về bản code cũ) khi bị lỗi:
Nếu bạn vừa deploy code mới mà bot bị crash (script Health check báo FAILED), bạn có thể rollback mọi server về một mã `commit_hash` ổn định trên Git:

```bash
# Thay 'a1b2c3d' bằng mã commit bạn muốn quay lại
ansible-playbook -i inventory.ini rollback-bot.yml -e target_version=a1b2c3d
```

## Triển khai (Deploy) PnL Dashboard (Streamlit):
1. **Chuẩn bị credentials**:
   Tạo thư mục `dashboard-configs/pnl_dashboard/` ngay trong thư mục `ansible/` và tạo file `credentials.json` chứa API theo mẫu.
2. **Chạy lệnh cài đặt**:
```bash
ansible-playbook -i inventory.ini deploy-dashboard.yml
```

## Dừng (Thủ tiêu) Bot vĩnh viễn:
Muốn tắt 1 hoặc toàn bộ bot, không cho chạy nền nữa (ví dụ đổi chiến lược, dời nhà sang code khác):
```bash
ansible-playbook -i inventory.ini stop-bot.yml
```

## Xem Log Bot trực tiếp trên server (Realtime):
Nếu muốn xem log chạy liên tục của 1 con bot cụ thể, SSH vào server và gõ lệnh:
```bash
journalctl -u bot_binance_01 -f
```
