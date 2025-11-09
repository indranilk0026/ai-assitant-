@echo off
echo Starting Streamlit app on network...
echo Share this URL with others on your network:
echo http://10.220.120.119:8501
echo.
streamlit run app.py --server.address 0.0.0.0 --server.port 8501


