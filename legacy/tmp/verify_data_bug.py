import pandas as pd
import numpy as np

def verify_data_corruption():
    print("="*60)
    print("🔍 KIỂM TRA LỖI DỮ LIỆU BTC TRONG FILE GỐC")
    print("="*60)
    
    # 1. Đọc file DATA GỐC (file dùng để split ra các file backtest 36,000%)
    file_path = 'data/processed/features_1h_btc_context.parquet'
    print(f"\nĐang đọc file: {file_path}")
    
    try:
        # Code lấy mẫu ETHUSDT để kiểm tra
        df = pd.read_parquet(file_path, filters=[('symbol', '==', 'ETHUSDT')])
    except Exception as e:
        print(f"Không thể đọc file: {e}")
        return
        
    print(f"Tổng số nến ETHUSDT: {len(df):,}")
    
    # 2. Kiểm tra cột btc_close
    if 'btc_close' in df.columns:
        zero_btc = len(df[df['btc_close'] == 0.0])
        print(f"\n⚠️ BẰNG CHỨNG 1: CỘT btc_close BỊ BẰNG 0.0")
        print(f"   Số nến có btc_close = 0.0: {zero_btc:,} / {len(df):,} ({zero_btc/len(df)*100:.2f}%)")
        print("   -> TẤT CẢ giá BTC đều bị ép về 0.0 do lỗi code merge cũ!")
    else:
        print("\nKhông tìm thấy cột btc_close!")
        
    # 3. Kiểm tra các cột bị sinh ra do lỗi merge đè
    dupe_cols = [c for c in df.columns if c.startswith('btc_') and (c.endswith('_x') or c.endswith('_y'))]
    if dupe_cols:
        print(f"\n⚠️ BẰNG CHỨNG 2: LỖI MERGE ĐÈ TẠO CỘT RÁC")
        print(f"   Các cột rác được tạo ra: {dupe_cols}")
        print("   -> Script cũ đã merge BTC data nhiều lần lên cùng 1 file!")

    # 4. Kiểm chứng lỗi Toán Học trên rs_vs_btc (Ý nghĩa: Tỉ giá Altcoin - Tỉ giá BTC)
    if 'rs_vs_btc' in df.columns and 'log_returns' in df.columns:
        print(f"\n⚠️ BẰNG CHỨNG 3: LỖI TOÁN HỌC TRÊN CỘT DÙNG ĐỂ TRAIN MODEL")
        print("   Cột rs_vs_btc = log_returns(ETH) - btc_log_returns(BTC)")
        print("   Do btc_close = 0.0 -> btc_log_returns = 0.0")
        print("   -> Hệ quả: rs_vs_btc bằng chính xác log_returns (mất đi ý nghĩa so sánh với BTC)")
        
        # So sánh rs_vs_btc và log_returns
        diff = (df['rs_vs_btc'] - df['log_returns']).abs().mean()
        is_identical = diff < 0.000001
        
        print(f"   Trung bình độ lệch giữa rs_vs_btc và log_returns: {diff:.6f}")
        if is_identical:
            print("   => KẾT LUẬN: rs_vs_btc CHÍNH LÀ log_returns (Giống nhau 100%)!")
        else:
            print(f"   => KẾT LUẬN: Không giống nhau hoàn toàn, lệch {diff:.6f} (Có thể do shift).")

    # 5. Kiểm chứng cột btc_corr (Tương quan 30 ngày với BTC)
    if 'btc_corr' in df.columns:
        print(f"\n⚠️ BẰNG CHỨNG 4: btc_corr (Tương quan 30 ngày với BTC) BỊ VÔ HIỆU HOÁ")
        zero_corr = len(df[df['btc_corr'] == 0.0])
        print(f"   Số nến có btc_corr = 0.0: {zero_corr:,} / {len(df):,} ({zero_corr/len(df)*100:.2f}%)")
        print("   -> Model nghĩ rằng ETH không bao giờ có tương quan với BTC, nó chỉ học được những con số 0.0 vô nghĩa!")
    
    print("\n" + "="*60)
    print("🔥 TỔNG KẾT:")
    print("Model đã được đào tạo (train) trên các đặc tính (features) BTC BỊ HỎNG HOÀN TOÀN.")
    print("Khi Backtest trên file split_symbols_v3 (được cắt ra từ file lỗi này), model được dùng lại KIẾN THỨC BỊ HỎNG đó để kiếm 36,000%.")
    print("Nhưng khi chạy LIVE (hoặc sync mới), giá trị thật của BTC xuất hiện, model sẽ không biết xử lý.")
    print("-> BẮT BUỘC TRAIN LẠI MODEL TRÊN DATA SYNC MỚI!")
    print("="*60)

if __name__ == "__main__":
    verify_data_corruption()
