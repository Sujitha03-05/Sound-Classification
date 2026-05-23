# Sound-Classification
Implementation of DPNet, ACDNet, and SincNet for the classification of bathroom sounds. This project includes model training, quantization (INT8), and on-device deployment using an ESP32-S3 microcontroller.

**Models (INT8 Quantized TFLite):**
    * `DPNet_INT8.tflite` - DPNet model optimized for edge deployment.
    * `ACDNet_INT8.tflite` - ACDNet model optimized for edge deployment.
    * `SyncNet_INT8.tflite` - SincNet model optimized for edge deployment.
**Edge Deployment:**
    * `DPNet_ESP32.ino` - Main Arduino sketch for ESP32-S3 deployment.
    * `scaler_params.h` - StandardScaler mean/scale arrays compiled directly into the firmware.
**Source Code & Notebooks:**
    * `syncnet-acdnet.ipynb` - Jupyter notebook containing model implementation and training pipelines.
**Documentation & Results:**
    * `results.pdf` - Evaluation metrics and project results.
    * `DPNet_Final_Presentation_2 [Autosaved].pptx` - Project presentation slides.
    * `Demo Image` & `Demo Video` - Visual demonstrations of the working hardware setup.

DPNet ESP32-S3 Deployment Guide

The following instructions detail how to deploy the DPNet model onto an ESP32-S3 microcontroller using an I2S microphone, an SD card module, and an OLED display.

1. SD Card Setup
The ESP32 reads the TensorFlow Lite model directly from the SD card to save memory.
1. Format your SD card to FAT32.
2. Create a folder named `model` in the root directory.
3. Copy the `DPNet_INT8.tflite` (rename it to `DPNet.tflite`) into the folder. 
   Path: `/model/DPNet.tflite`

2. Required Arduino Libraries
Open the Arduino IDE and install the following via **Tools → Manage Libraries**:
* ESP8266 and ESP32 OLED driver (by ThingPulse) - For the display.
* TensorFlowLite_ESP32 (by TensorFlowLite_ESP32) - Search for "tflite esp32" to run the ML model.

3. Pin Wiring 

| Component | ESP32-S3 Pin | Function |
| :--- | :--- | :--- |
| OLED Display | GPIO11 | SDA |
| | GPIO12 | SCL |
| SD Card Module | GPIO48 | CLK |
| | GPIO47 | MISO |
| | GPIO38 | MOSI |
| | GPIO21 | CS |
| I2S Microphone | GPIO2 | SCK |
| | GPIO3 | WS |
| | GPIO1 | DIN |

4. Arduino IDE Board Settings
To ensure the firmware compiles and runs correctly, configure your board under the **Tools** menu as follows:

* Board: ESP32-S3 Dev Module
* USB CDC On Boot: Enabled
* USB Mode: Hardware CDC and JTAG
* Flash Size: 4MB
* Partition Scheme: Huge APP (3MB No OTA) ⚠️ *Important for accommodating large ML firmware.*

5. Execution Flow (OLED Display Stages)
Once flashed and powered, the OLED will guide you through the following operational stages:
1.  Initializing... (Booting up and connecting peripherals)
2.  Stage: Recording (5-second countdown to capture audio)
3.  Stage: Processing (Applying hamming window + standard scaler)
4.  Stage: Loading (Reading the `.tflite` model from the SD card)
5.  Stage: Inference (Running the TensorFlow Lite model)
6.  Result: Outputs the `<ClassName>` alongside a confidence bar.
