#!/bin/bash
# exit if pipeline fails or unset variables
set -eu

# default values, satellites list with sstv transmitter
# 25544 - ISS 
# 57172 - UmKA-1 
# 67290 - SAKHACUBE
# 40931 - LAPAN-A2 
# 57180 - Monitor-3
# 61762 - ArcticSat-1
# 67291 - QMR-KWT 2
# 57189 - VIZARD-meteo
# 48274 - CSS (Tianhe)
# 57203 - UTMN2
# 67279 - GALAPAGOS-UTE-SWSU
# 59112 - SONATE-2
# 57190 - Nanozond-1

: "${SSTV_ENABLE:=true}"
: "${SSTV_NORAD:= 25544 57172 67290 40931 57180 61762 67291 57189 48274 57203 67279 59112 57190}"
: "${SATNOGS_OUTPUT_PATH:=/tmp/.satnogs/data}"
: "${SSTV_APP_DIR:=/app/sstv2satno/sstv}"
: "${MAX_WAIT_TIME:=180}" # Maximum time, script will waiting audio .ogg from satnogs-client. Default: 3 minutes in seconds
: "${CHECK_INTERVAL:=1}"  # Check every 1 second

# Launch with: {command} {{ID}} {{FREQ}} {{TLE}} {{TIMESTAMP}} {{BAUD}} {{SCRIPT_NAME}}
# /app/satnogs-post 13779045 437550000 '{"tle0": "ISS", "tle1": "1 25544U 98067A   26101.48935997  .00006038  00000-0  11816-3 0  9991", "tle2": "2 25544  51.6326 270.1822 0006432 299.5706  60.4641 15.48873216561413"}' 2026-04-11T20-56-48 64000 satnogs_fsk.py

ID="$2"      # $2 observation ID
TLE="$4"     # $4 used tle's
DATE="$5"    # $5 timestamp Y-m-dTH-M-S
BAUD="$6"    # $6 baudrate

# Extract satellite name and NORAD
SATNAME=${TLE#*tle0\": \"}
SATNAME=${SATNAME%%\"*}
NORAD=${TLE#*tle2\": \"2 }
NORAD=${NORAD%% *}
OGG_FILE="satnogs_${ID}_${DATE}.ogg"
ELAPSED=0

echo "INFO: $ID, Norad: $NORAD, Sat: $SATNAME, Baud: $BAUD, TLE: $TLE" 

if [[ " $SSTV_NORAD " =~ .*\ ${NORAD}\ .* && "$SSTV_ENABLE" ]]; then
        echo "[sstv] ✓ SSTV Decoder Start"
        
        SLANT_FACTOR=0
		if [[ " $NORAD " =~ "59112" ]]; then
			SLANT_FACTOR=-0.45
		fi
		
		echo "$NORAD, $SLANT_FACTOR" 

        cd $SATNOGS_OUTPUT_PATH
        rm -rf sstv.wav

        # Loop until file is ready or timeout
        while [ $ELAPSED -lt $MAX_WAIT_TIME ]; do
                # Check if file exists and is readable
                if [ -f "$OGG_FILE" ] && [ -r "$OGG_FILE" ]; then
                        # Optional: Check if file size is not zero
                        if [ -s "$OGG_FILE" ]; then
                                echo "[sstv] ✓ $OGG_FILE file is ready.."                              
                                cd "${SSTV_APP_DIR}"
                                ./sox "${SATNOGS_OUTPUT_PATH}/${OGG_FILE}" "${SATNOGS_OUTPUT_PATH}/sstv.wav"
                                ./sstv_general.py -d "${SATNOGS_OUTPUT_PATH}/sstv.wav" -o "data_${ID}_general.png" --dir "${SATNOGS_OUTPUT_PATH}" --slant "${SLANT_FACTOR}"                               
                                ./sstv_bandpass.py -d "${SATNOGS_OUTPUT_PATH}/sstv.wav" -o "data_${ID}_bandpass.png" --dir "${SATNOGS_OUTPUT_PATH}" --slant "${SLANT_FACTOR}"
                                ./sstv_overlap.py -d "${SATNOGS_OUTPUT_PATH}/sstv.wav" -o "data_${ID}_overlap.png" --dir "${SATNOGS_OUTPUT_PATH}" --slant "${SLANT_FACTOR}"
                                exit 0
                        fi
                fi
                
                # Sleep before next check
                sleep $CHECK_INTERVAL
                ELAPSED=$((ELAPSED + CHECK_INTERVAL))
                
                # Optional: Print progress every 10 seconds
                if [ $((ELAPSED % 10)) -eq 0 ]; then
                        echo "[sstv] Still waiting... ($ELAPSED seconds elapsed)"
                fi
        done
    
fi
