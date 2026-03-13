import joblib
from pathlib import Path

# Đường dẫn đến file joblib của bạn
file_path = Path("ml") / "optuna_best_params_wfv_median.joblib"

def read_saved_params():
    if not file_path.exists():
        print(f"❌ Không tìm thấy file tại: {file_path}")
        return

    try:
        # Load dữ liệu từ file
        params = joblib.load(file_path)
        
        print("✅ Đã đọc thành công file joblib!\n")
        print("📊 THAM SỐ ĐANG LƯU TRONG FILE:")
        print("-" * 40)
        
        # In ra từng tham số cho dễ nhìn
        if isinstance(params, dict):
            for key, value in params.items():
                print(f"  {key:<20}: {value:.4f}")
        else:
            print(params)
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    read_saved_params()