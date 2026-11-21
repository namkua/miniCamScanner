#!/usr/bin/env bash
# Script khởi chạy nhanh ứng dụng CamScanner Web UI

if command -v /opt/anaconda3/bin/streamlit &> /dev/null; then
    /opt/anaconda3/bin/streamlit run src/ui/app.py
elif command -v streamlit &> /dev/null; then
    streamlit run src/ui/app.py
else
    echo "Không tìm thấy streamlit. Vui lòng cài đặt: pip install streamlit"
    exit 1
fi
