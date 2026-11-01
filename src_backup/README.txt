NETSENTINELS DASHBOARD - how to run

FILES
  app.py              small Flask server (serves the dashboard + reads the CSV)
  index.html          the dashboard (all pages, charts, live replay)
  data/               put your results CSV here (sample included)
  requirements.txt    flask, pandas, numpy
  run_dashboard.bat   double-click to install + start (Windows)

OPTION A - in VS Code
  1. Extract the zip. VS Code: File > Open Folder > netsentinels_dashboard
  2. Terminal > New Terminal, then:
       pip install -r requirements.txt
       python app.py
  3. Browser opens at http://127.0.0.1:5050 (keep the terminal running)

OPTION B - no VS Code: double-click run_dashboard.bat

TO USE YOUR OWN RESULTS: copy the CSV made by predict.py into the data folder
(it needs rf_attack_probability and if_anomaly_score columns) and refresh the page,
or use the Upload CSV button.
