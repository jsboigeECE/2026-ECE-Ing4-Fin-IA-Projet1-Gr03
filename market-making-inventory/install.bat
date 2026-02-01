@echo off
REM Installation script for market-making-inventory

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install numpy scipy pandas matplotlib ortools gymnasium

echo Installation complete!
echo.
echo To activate the environment, run: venv\Scripts\activate.bat
echo To run backtests, run: python experiments\run_backtest.py --seed 42
