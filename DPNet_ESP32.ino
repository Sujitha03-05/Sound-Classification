/*
 * Sound-based Assistive Monitoring System — ESP32-S3
 * Architecture: DPNet (INT8 Quantized)
 * Hardware: INMP441 Mic, SH1106 OLED, 8MB PSRAM [cite: 93, 95]
 */

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include "SH1106Wire.h"
#include "driver/i2s_std.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "scaler_params.h" // 

// ===================== PIN CONFIGURATION [cite: 97, 99, 101] =====================
#define I2S_SCK_PIN   GPIO_NUM_2
#define I2S_WS_PIN    GPIO_NUM_3
#define I2S_DIN_PIN   GPIO_NUM_1

#define SD_CLK_PIN    48
#define SD_MISO_PIN   47
#define SD_MOSI_PIN   38
#define SD_CS_PIN     21

// ===================== PROJECT PARAMETERS [cite: 138, 140, 475] =====================
#define SAMPLE_RATE       20000     
#define INPUT_LENGTH      30225     // 1.51s Window
#define MIC_THRESHOLD     0.05f     // Sound trigger
#define TENSOR_ARENA_SIZE (1024 * 1024) // 1MB in PSRAM
#define NO_CLASSES        7         

const char* CLASS_NAMES[NO_CLASSES] = {
  "Flush", "Door", "Basin_Tap", "Walker_Crutch", "Bathroom_Tap", "No_Class", "Shower"
}; // [cite: 116]

// ===================== GLOBALS =====================
SH1106Wire display(0x3C, 11, 12);
i2s_chan_handle_t rx_handle;
float* shared_audio_buffer = nullptr; 

uint8_t* model_data   = nullptr;
uint8_t* tensor_arena = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TaskHandle_t RecordTask;
SemaphoreHandle_t audioSemaphore;

// ===================== CORE FUNCTIONS =====================

void showStatus(const char* msg, const char* sub = "") {
  display.clear();
  display.setFont(ArialMT_Plain_16);
  display.drawString(0, 0, msg);
  display.setFont(ArialMT_Plain_10);
  display.drawString(0, 20, sub);
  display.display();
  Serial.printf(">> %s : %s\n", msg, sub);
}

void initI2S() {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  i2s_new_channel(&chan_cfg, NULL, &rx_handle);
  i2s_std_config_t std_cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = I2S_SCK_PIN, .ws = I2S_WS_PIN, .dout = I2S_GPIO_UNUSED, .din = I2S_DIN_PIN}
  };
  i2s_channel_init_std_mode(rx_handle, &std_cfg);
  i2s_channel_enable(rx_handle);
}

bool loadModel() {
  File f = SD.open("/model/DPNet_INT8.tflite"); // [cite: 451]
  if (!f) return false;
  size_t size = f.size();
  model_data = (uint8_t*)ps_malloc(size); // Load to PSRAM [cite: 459]
  if (!model_data) return false;
  f.read(model_data, size);
  f.close();
  return true;
}

bool initTFLite() {
  tflite_model = tflite::GetModel(model_data);
  
  static tflite::MicroMutableOpResolver<25> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D(); // [cite: 56, 178]
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddTranspose(); // For Permute layer [cite: 209]
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddDequantize();

  // FORCE PSRAM ALLOCATION to fix "Failed to resize buffer" error [cite: 475]
  tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
  if (!tensor_arena) return false;

  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;
  
  return (interpreter->AllocateTensors() == kTfLiteOk); 
}

// Core 0: Continuous recording and thresholding
void recordingLoop(void* pvParameters) {
  int32_t raw_sample;
  size_t bytes_read;
  uint32_t idx = 0;
  bool is_active = false;

  while (true) {
    i2s_channel_read(rx_handle, &raw_sample, sizeof(int32_t), &bytes_read, portMAX_DELAY);
    float sample = (float)(raw_sample >> 14) / 32768.0f;

    if (!is_active && abs(sample) > MIC_THRESHOLD) {
      is_active = true;
      Serial.println("Sound Triggered...");
    }

    if (is_active) {
      shared_audio_buffer[idx++] = sample;
      if (idx >= INPUT_LENGTH) {
        xSemaphoreGive(audioSemaphore); 
        idx = 0;
        is_active = false; 
      }
    }
  }
}

// Core 1: Preprocessing and Inference [cite: 441, 693]
void preprocessAndInference() {
  TfLiteTensor* input = interpreter->input(0);
  
  for (int i = 0; i < INPUT_LENGTH; i++) {
    // Hamming Window [cite: 159, 161]
    float hamming = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (INPUT_LENGTH - 1));
    float val = shared_audio_buffer[i] * hamming;
    
    // Z-Score Scaling using PROGMEM flash [cite: 167, 176]
    float scaled = (val - pgm_read_float(&SCALER_MEAN[i])) / (pgm_read_float(&SCALER_SCALE[i]) + 1e-8f);
    
    // INT8 Quantization Mapping [cite: 438]
    input->data.int8[i] = (int8_t)(scaled / input->params.scale + input->params.zero_point);
  }

  if (interpreter->Invoke() == kTfLiteOk) {
    TfLiteTensor* output = interpreter->output(0);
    int bestIdx = 0; 
    int8_t maxVal = -128;

    for (int i = 0; i < NO_CLASSES; i++) {
      if (output->data.int8[i] > maxVal) {
        maxVal = output->data.int8[i];
        bestIdx = i;
      }
    }
    
    float confidence = (maxVal - output->params.zero_point) * output->params.scale;
    
    display.clear();
    display.setFont(ArialMT_Plain_16);
    display.drawString(0, 0, CLASS_NAMES[bestIdx]);
    display.setFont(ArialMT_Plain_10);
    display.drawString(0, 20, "Conf: " + String(confidence * 100, 1) + "%");
    display.display();
    Serial.printf("Result: %s (%.2f%%)\n", CLASS_NAMES[bestIdx], confidence * 100);
  }
}

void setup() {
  Serial.begin(115200);
  display.init();
  showStatus("DPNet INT8", "Booting...");

  // Alloc buffer in PSRAM to save internal SRAM [cite: 176, 697]
  shared_audio_buffer = (float*)ps_malloc(INPUT_LENGTH * sizeof(float));
  audioSemaphore = xSemaphoreCreateBinary();

  initI2S();
  SPI.begin(SD_CLK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  
  if (SD.begin(SD_CS_PIN) && loadModel() && initTFLite()) {
    showStatus("System Ready", "Awaiting Sound...");
    // Pinned to Core 0 
    xTaskCreatePinnedToCore(recordingLoop, "RecordTask", 8192, NULL, 1, &RecordTask, 0);
  } else {
    showStatus("ERROR", "Init Failed");
    while(1);
  }
}

void loop() {
  // Take buffer from Core 0 and process on Core 1 [cite: 441, 693]
  if (xSemaphoreTake(audioSemaphore, portMAX_DELAY)) {
    preprocessAndInference();
  }
}