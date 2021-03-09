export PYTHONPATH=ranzcr_clip:${PYTHONPATH}

nohup streamlit run demo_app/app.py &> ranzcr_clip_demo_app.log 2>&1 &
