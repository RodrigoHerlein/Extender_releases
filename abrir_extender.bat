@echo off
cd /d "%~dp0.."

echo Verificando Flask...
python -c "import flask" 2>nul || (
    echo Instalando Flask...
    pip install flask --quiet
)

echo.
echo =====================================================
echo   EXTENDER UI — http://localhost:5000
echo   Las dependencias pesadas (SBERT, HDBSCAN, UMAP)
echo   se instalan automaticamente si faltan.
echo   El mapa conceptual y best_configs cargan
echo   de inmediato. El corpus tarda ~30s.
echo =====================================================
echo.

:: Iniciar servidor en segundo plano (misma ventana)
start /b python extender_app.py

:: Esperar hasta que el puerto 5000 esté abierto
echo Esperando que EXTENDER este listo...
:wait_loop
python -c "import socket; s=socket.socket(); s.settimeout(1); r=s.connect_ex(('127.0.0.1',5000)); s.close(); exit(r)" 2>nul
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

echo Listo. Abriendo navegador...
start "" http://localhost:5000

pause
