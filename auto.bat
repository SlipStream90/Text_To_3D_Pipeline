@echo off
ECHO =======================================
ECHO     INITIALIZING CONDA ENVIRONMENT
ECHO =======================================

:: -------- CONFIGURATION --------
SET "CONDA_ENV_NAME=cudatorch"

SET "INTERMEDIATE_FILE=interim_output\spaceship.glb"
SET "FINAL_FILE=Final\spaceship_final.glb"

:: Blender script
SET "BLENDER_SCRIPT=finisher.py"

:: Path to Blender executable
SET "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"

:: -------- ACTIVATE CONDA ENVIRONMENT --------
CALL conda.bat activate %CONDA_ENV_NAME%
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Could not activate conda environment '%CONDA_ENV_NAME%'
    PAUSE
    EXIT /B
)
ECHO [OK] Conda environment '%CONDA_ENV_NAME%' activated.

:: -------- CREATE OUTPUT DIRECTORIES --------

:: -------- STAGE 1: TRELLIS 3D MODEL GENERATION --------
ECHO.
ECHO =======================================
ECHO     STAGE 1: TRELLIS 3D MODEL GENERATION
ECHO =======================================

modal run TReS.py


:: -------- STAGE 2: BLENDER POST-PROCESSING --------
ECHO.
ECHO =======================================
ECHO     STAGE 2: BLENDER POST-PROCESSING
ECHO =======================================

:: Construct Blender arguments
SET "BLENDER_ARGS=--input %INTERMEDIATE_FILE% --output %FINAL_FILE%"

:: Run Blender in background
"%BLENDER_PATH%" --background --python %BLENDER_SCRIPT% -- %BLENDER_ARGS%
IF NOT EXIST "%FINAL_FILE%" (
    ECHO.
    ECHO [ERROR] Stage 2 failed. File not found: %FINAL_FILE%
    PAUSE
    EXIT /B
)

:: -------- PIPELINE COMPLETE --------
ECHO.
ECHO =======================================
ECHO         PIPELINE COMPLETE!
ECHO =======================================
ECHO Final asset saved to: %FINAL_FILE%
PAUSE