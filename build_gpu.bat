@echo off
set APP_NAME=RatTracker
set DIST=dist\%APP_NAME%
set ZIP_OUT=%~dp0..\RatTracker_lab.zip
set PYTHON=C:\Users\sohai\anaconda3\envs\yolorat_gpu\python.exe
set PYINST=C:\Users\sohai\anaconda3\envs\yolorat_gpu\Scripts\pyinstaller.exe

echo.
echo ============================================================
echo  RatTracker Build  GPU - yolorat_gpu env
echo ============================================================
echo.

echo [1/4] Limpiando builds anteriores...
if exist dist "%PYTHON%" -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
if exist build "%PYTHON%" -c "import shutil; shutil.rmtree('build', ignore_errors=True)"
if exist %APP_NAME%.spec del /Q %APP_NAME%.spec

echo [2/4] Construyendo ejecutable...
"%PYINST%" --onedir --windowed --name %APP_NAME% ^
  --icon "gui\assets\icons\app_icon.ico" ^
  --paths "scripts" ^
  --add-data "gui\main_window.ui;gui" ^
  --add-data "gui\assets\icons\app_icon.ico;gui\assets\icons" ^
  --add-data "scripts\config\translations.json;scripts\config" ^
  run_gui.py
if errorlevel 1 ( echo ERROR en PyInstaller. & exit /b 1 )

echo [3/4] Preparando carpetas de usuario...
if not exist "%DIST%\models" mkdir "%DIST%\models"
if exist "models\yolo_ratas_v1.pt" copy /Y "models\yolo_ratas_v1.pt" "%DIST%\models\yolo_ratas_v1.pt" > nul
(echo path: datasets& echo train: unified/train/images& echo val:   unified/valid/images& echo test:  unified/test/images& echo.& echo nc: 5& echo names: ['rat_climbing', 'rat_grooming', 'rat_head_dipping', 'rat_inmobile', 'rat_rearing']& echo kpt_shape: [3, 3]& echo flip_idx: [0, 1, 2]) > "%DIST%\models\data.yaml"
if not exist "%DIST%\datasets" mkdir "%DIST%\datasets"
if not exist "%DIST%\videosPrueba" mkdir "%DIST%\videosPrueba"

echo [4/4] Creando zip...
powershell -NoProfile -Command "Compress-Archive -Path '%DIST%' -DestinationPath '%ZIP_OUT%' -Force"
if errorlevel 1 ( echo ERROR al crear el zip. & exit /b 1 )

"%PYTHON%" -c "import shutil; shutil.rmtree('build', ignore_errors=True)"
del /Q %APP_NAME%.spec 2>nul

echo.
echo ============================================================
echo  LISTO: %ZIP_OUT%
echo    RatTracker\RatTracker.exe
echo    RatTracker\_internal\
echo    RatTracker\models\  yolo_ratas_v1.pt + data.yaml
echo    RatTracker\datasets\
echo    RatTracker\videosPrueba\
echo ============================================================
echo.
