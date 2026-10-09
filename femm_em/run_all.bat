@echo off
rem Full calibrated FEMM chain -> report. Resumes from results_partial.json if a previous run was stopped.
rem Double-click, or run from a terminal. Takes ~2 h. Leave the window open until it says DONE.
cd /d "%~dp0"
echo [%time%] full study (femm_motor.py)...
python femm_motor.py >> run.log 2>&1 || goto :fail
echo [%time%] plots...
python plot_results.py > plot_run.log 2>&1 || goto :fail
echo [%time%] field maps (field_maps.py)...
python field_maps.py > field_run.log 2>&1 || goto :fail
echo [%time%] d-q map and torque-speed (dq_study.py; reuses dq.json if present)...
python dq_study.py > dq_run.log 2>&1 || goto :fail
echo [%time%] efficiency map (eff_map.py)...
python eff_map.py > eff_run.log 2>&1 || goto :fail
echo [%time%] report (build_report.py)...
python build_report.py > report_run.log 2>&1 || goto :fail
echo.
echo DONE>> run.log
echo DONE - report: %~dp0report\v7_motor_report.html
pause
exit /b 0
:fail
echo.
echo FAILED>> run.log
echo FAILED at the step above - rerun this file to resume.
pause
exit /b 1
