#include <SPI.h>
#include <SD.h>
#include <RTClib.h>
#include <ArduTFLite.h> 

// TensorFlow Lite includes
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Ini penting
#include "tensorflow/lite/schema/schema_generated.h"

// Custom Libraries (Pastikan file-file ini ada di folder sketch Anda)
#include "PeakDetector.h"
#include "CircularBuffer.h"
#include "ElasticEnvelope.h"
#include "DetectionResponder.h"
#include "model.h" 

// FreeRTOS specific includes
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h> // Untuk Mutex (Semaphore)

// --- Pin Definitions for ADS1293---
int mosi = 35;
int miso = 37;
int sclk = 36;
int dredy = 41;    
int csb = 39;      

// Pin definitions for SD card SPI communication (HSPI)
#define HSPI_MOSI 11 
#define HSPI_MISO 13 
#define HSPI_SCLK 12 
#define HSPI_CS   10 // <-- PIN CS YANG ANDA GUNAKAN! Verifikasi Ulang!

// Logging control pin
#define LOGGING_SWITCH_PIN 2 

// LED Indicator Pin (Used for debugging SD Card errors and Classification)
#define LED_INDICATOR_PIN 21 

// --- SPI Instances ---
SPIClass vspi(FSPI); 
SPIClass hspi(HSPI); 

// --- RTC Instance ---
RTC_DS3231 rtc;

// --- FreeRTOS Handles ---
QueueHandle_t ecgRawDataQueue;
QueueHandle_t processedDataQueue;
SemaphoreHandle_t loggingMutex;

// --- Global Variables ---
volatile bool loggingEnabled = false;
volatile bool lastLoggingState = false; 
String currentLogFileName = ""; 
volatile float bpm = 0; 
volatile int classification = 0; 
volatile uint32_t sampleCounterGlobal = 0; 
volatile uint32_t currentLoggingMicros = 0; 

// --- TensorFlow Lite Variables ---
namespace {
  const int WINDOW_SIZE = 10; 
  PeakDetector<5> peakDetector(0.005f, 0.2f); 
  CircularBuffer<int, WINDOW_SIZE> buffer; 
  ElasticEnvelope envelope(0.1f); 

  constexpr int kTensorArenaSize = 2 * 1024; 
  uint8_t tensor_arena[kTensorArenaSize]; 

  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  float model_output_probabilities[3] = {0.0f, 0.0f, 0.0f}; 
} // namespace

// --- FIR Filter Variables ---
const int firLength = 480;    
float firCoefficients[firLength] = {
  // Koefisien FIR Anda (PASTE SELURUH KOEFISIEN DARI KODE ASLI ANDA DI SINI)
  -0.00009513, -0.00002638, 0.00006214, 0.00010473, 0.00006844, -0.00002235,
  -0.00010293, -0.00011490, -0.00004911, 0.00004653, 0.00010095, 0.00007154,
  -0.00002365, -0.00011745, -0.00014168, -0.00007744, 0.00002919, 0.00009898,
  0.00007697, -0.00002576, -0.00013774, -0.00017768, -0.00011458, 0.00000706,
  0.00009694, 0.00008427, -0.00002859, -0.00016433, -0.00022490, -0.00016377,
  -0.00002328, 0.00009252, 0.00009266, -0.00003195, -0.00019745, -0.00028504,
  -0.00022821, -0.00006548, 0.00008299, 0.00010112, -0.00003564, -0.00023700,
  -0.00035938, -0.00031094, -0.00012334, 0.00006529, 0.00010838, -0.00003938,
  -0.00028249, -0.00044877, -0.00041467, -0.00020073, 0.00003605, 0.00011286,
  -0.00004288, -0.00033305, -0.00055348, -0.00054175, -0.00030147, -0.00000835,
  0.00011274, -0.00004581, -0.00038740, -0.00067315, -0.00069399, -0.00042920,
  -0.00007169, 0.00010597, -0.00004784, -0.00044385, -0.00080677, -0.00087259,
  -0.00058731, -0.00015786, 0.00009026, -0.00004865, -0.00050032, -0.00095261,
  -0.00107808, -0.00077882, -0.00027078, 0.00006310, -0.00004794, -0.00055431,
  -0.00110817, -0.00131019, -0.00100627, -0.00041433, 0.00002184, -0.00004542,
  -0.00060293, -0.00127022, -0.00156782, -0.00127169, -0.00059230, -0.00003633,
  -0.00004087, -0.00064294, -0.00143472, -0.00184899, -0.00157650, -0.00080834,
  -0.00011455, -0.00003411, -0.00067072, -0.00159684, -0.00215080, -0.00192148,
  -0.00106594, -0.00021581, -0.00002503, -0.00068230, -0.00175096, -0.00246944,
  -0.00230676, -0.00136843, -0.00034346, -0.00001359, -0.00067335, -0.00189063,
  -0.00280013, -0.00273186, -0.00171900, -0.00050100, 0.00000018, -0.00063913,
  -0.00200857, -0.00313718, -0.00319572, -0.00212083, -0.00069221, 0.00001616,
  -0.00057443, -0.00209654, -0.00347395, -0.00369681, -0.00257726, -0.00092125,
  0.00003416, -0.00047348, -0.00214522, -0.00380280, -0.00423329, -0.00309203,
  -0.00119290, 0.00005392, -0.00032966, -0.00214396, -0.00411508, -0.00480327,
  -0.00366972, -0.00151290, 0.00007511, -0.00013521, -0.00208034, -0.00440099,
  -0.00540516, -0.00431645, -0.00188845, 0.00009733, 0.00011937, -0.00193946,
  -0.00464927, -0.00603816, -0.00504088, -0.00232909, 0.00012015, 0.00044628,
  -0.00170270, -0.00484675, -0.00670308, -0.00585594, -0.00284810, 0.00014308,
  0.00086202, -0.00134565, -0.00497742, -0.00740362, -0.00678163, -0.00346488,
  0.00016564, 0.00139014, -0.00083439, -0.00502077, -0.00814848, -0.00784999,
  -0.00420929, 0.00018729, 0.00206638, -0.00011860, -0.00494848, -0.00895521,
  -0.00911444, -0.00512974, 0.00020756, 0.00294852, 0.00088200, -0.00471783,
  -0.00985792, -0.01066860, -0.00631006, 0.00022595, 0.00413746, 0.00230711,
  -0.00425707, -0.01092402, -0.01268870, -0.00790799, 0.00024202, 0.00582715,
  0.00442710, -0.00342907, -0.01229641, -0.01554308, -0.01025711, 0.00025539,
  0.00844393, 0.00784747, -0.00192367, -0.01432181, -0.02013704, -0.01419998,
  0.00026571, 0.01314035, 0.01426755, 0.00116198, -0.01807788, -0.02941632,
  -0.02263788, 0.00027274, 0.02447192, 0.03098758, 0.01009353, -0.02930554,
  -0.06135051, -0.05654396, 0.00027630, 0.09804386, 0.20161572, 0.26796519,
  0.26796519, 0.20161572, 0.09804386, 0.00027630, -0.05654396, -0.06135051,
  -0.02930554, 0.01009353, 0.03098758, 0.02447192, 0.00027274, -0.02263788,
  -0.02941632, -0.01807788, 0.00116198, 0.01426755, 0.01314035, 0.00026571,
  -0.01419998, -0.02013704, -0.01432181, -0.00192367, 0.00784747, 0.00844393,
  0.00025539, -0.01025711, -0.01554308, -0.01229641, -0.00342907, 0.00442710,
  0.00582715, 0.00024202, -0.00790799, -0.01268870, -0.01092402, -0.00425707,
  0.00230711, 0.00413746, 0.00022595, -0.00631006, -0.01066860, -0.00985792,
  -0.00471783, 0.00088200, 0.00294852, 0.00020756, -0.00512974, -0.00911444,
  -0.00895521, -0.00494848, -0.00011860, 0.00206638, 0.00018729, -0.00420929,
  -0.00784999, -0.00814848, -0.00502077, -0.00083439, 0.00139014, 0.00016564,
  -0.00346488, -0.00678163, -0.00740362, -0.00497742, -0.00134565, 0.00086202,
  0.00014308, -0.00284810, -0.00585594, -0.00670308, -0.00484675, -0.00170270,
  0.00044628, 0.00012015, -0.00232909, -0.00504088, -0.00603816, -0.00464927,
  -0.00193946, 0.00011937, 0.00009733, -0.00188845, -0.00431645, -0.00540516,
  -0.00440099, -0.00208034, -0.00013521, 0.00007511, -0.00151290, -0.00366972,
  -0.00480327, -0.00411508, -0.00214396, -0.00032966, 0.00005392, -0.00119290,
  -0.00309203, -0.00423329, -0.00380280, -0.00214522, -0.00047348, 0.00003416,
  -0.00092125, -0.00257726, -0.00369681, -0.00347395, -0.00209654, -0.00057443,
  0.00001616, -0.00069221, -0.00212083, -0.00319572, -0.00313718, -0.00200857,
  -0.00063913, 0.00000018, -0.00050100, -0.00171900, -0.00273186, -0.00280013,
  -0.00189063, -0.00067335, -0.00001359, -0.00034346, -0.00136843, -0.00230676,
  -0.00246944, -0.00175096, -0.00068230, -0.00002503, -0.00021581, -0.00106594,
  -0.00192148, -0.00215080, -0.00159684, -0.00067072, -0.00003411, -0.00011455,
  -0.00080834, -0.00157650, -0.00184899, -0.00143472, -0.00064294, -0.00004087,
  -0.00003638, -0.00059230, -0.00127169, -0.00156782, -0.00127022, -0.00060293,
  -0.00004542, 0.00002184, -0.00041433, -0.00100627, -0.00131019, -0.00110817,
  -0.00055431, -0.00004794, 0.00006310, -0.00027078, -0.00077882, -0.00107808,
  -0.00095261, -0.00050032, -0.00004865, 0.00009026, -0.00015786, -0.00058731,
  -0.00087259, -0.00080677, -0.00044385, -0.00004784, 0.00010597, -0.00007169,
  -0.00042920, -0.00069399, -0.00067315, -0.00038740, -0.00004581, 0.00011274,
  -0.00000835, -0.00030147, -0.00054175, -0.00055348, -0.00033305, -0.00004288,
  0.00011286, 0.00003605, -0.00020073, -0.00041467, -0.00044877, -0.00028249,
  -0.00003938, 0.00010838, 0.00006529, -0.00012334, -0.00031094, -0.00035938,
  -0.00023700, -0.00003564, 0.00010112, 0.00008299, -0.00006548, -0.00022821,
  -0.00028504, -0.00019745, -0.00003195, 0.00009266, 0.00009252, -0.00002328,
  -0.00016377, -0.00022490, -0.00016433, -0.00002859, 0.00008427, 0.00009694,
  0.00000706, -0.00011458, -0.00017768, -0.00013774, -0.00002576, 0.00007697,
  0.00009898, 0.00002919, -0.00007744, -0.00014168, -0.00011745, -0.00002365,
  0.00007154, 0.00010095, 0.00004653, -0.00004911, -0.00011490, -0.00010293,
  -0.00002235, 0.00006844, 0.00010473, 0.00006214, -0.00002638, -0.00009513,
};

float firBuffer1[firLength] = {0.0f}; 
float firBuffer2[firLength] = {0.0f}; 
float firBuffer3[firLength] = {0.0f}; 

// --- Logging Buffer (untuk menulis banyak data sekaligus ke SD) ---
#define LOG_BUFFER_SIZE 64 
struct LogEntry {
  char timestamp[16]; 
  int32_t ecg1, ecg2, ecg3, leadIII;
  float bpm; 
  int classification;
};
LogEntry logBuffer[LOG_BUFFER_SIZE];
uint16_t logBufferIndex = 0; 

// --- Data Structures for FreeRTOS Queues ---
struct RawECGData { int32_t ch1, ch2, ch3; };
struct ProcessedECGData {
  uint32_t sampleCount; 
  int32_t ecg1Filtered, ecg2Filtered, ecg3Filtered, leadIII;
  float bpmValue;
  int classValue;
};

// --- Function Prototypes ---
float applyFIRFilter(float newValue, float* buffer_to_use, int buffer_len);
void readECGStreamingData(int32_t *ecg1, int32_t *ecg2, int32_t *ecg3);
void setup_ECG();
void writeRegister(byte reg, byte data);
byte readRegister(byte reg);

void checkLoggingSwitch();
void startLogging();
void stopLogging();
void writeBufferToSD();

void ecgReadTask(void* pvParameters);
void processDataTask(void* pvParameters);
void logDataTask(void* pvParameters);

// ========================================================================
// --- Implementasi Fungsi Helper Lainnya ---
// ========================================================================

float applyFIRFilter(float newValue, float* buffer_to_use, int buffer_len) {
  for (int i = buffer_len - 1; i > 0; i--) { buffer_to_use[i] = buffer_to_use[i - 1]; }
  buffer_to_use[0] = newValue;
  float output = 0.0f;
  for (int i = 0; i < buffer_len; i++) { output += firCoefficients[i] * buffer_to_use[i]; }
  return output;
}

void readECGStreamingData(int32_t *ecg1, int32_t *ecg2, int32_t *ecg3) {
  int32_t value = 0;
  digitalWrite(csb, LOW); 
  vspi.transfer(0x50 | 0x80); 
  value = vspi.transfer(0x00); value = (value << 8) | vspi.transfer(0x00); value = (value << 8) | vspi.transfer(0x00);
  if (value & 0x800000) { value |= 0xFF000000; } *ecg1 = value;
  value = vspi.transfer(0x00); value = (value << 8) | vspi.transfer(0x00); value = (value << 8) | vspi.transfer(0x00);
  if (value & 0x800000) { value |= 0xFF000000; } *ecg2 = value;
  value = vspi.transfer(0x00); value = (value << 8) | vspi.transfer(0x00); value = (value << 8) | vspi.transfer(0x00);
  if (value & 0x800000) { value |= 0xFF000000; } *ecg3 = value;
  digitalWrite(csb, HIGH); 
}

void setup_ECG() {
  writeRegister(0x01, 0x11); writeRegister(0x02, 0x19); writeRegister(0x03, 0x2E);
  writeRegister(0x0A, 0x07); writeRegister(0x0C, 0x04); writeRegister(0x0D, 0x01);
  writeRegister(0x0E, 0x02); writeRegister(0x0F, 0x03); writeRegister(0x10, 0x01);
  writeRegister(0x12, 0x04); writeRegister(0x21, 0x02); writeRegister(0x22, 0x10);
  writeRegister(0x23, 0x10); writeRegister(0x24, 0x10); writeRegister(0x27, 0x08);
  writeRegister(0x2F, 0x70); writeRegister(0x00, 0x01); 
  Serial.println("ECG configuration completed and conversion started.");
}

void writeRegister(byte reg, byte data) {
  reg &= 0x7F; digitalWrite(csb, LOW); vspi.transfer(reg); vspi.transfer(data); digitalWrite(csb, HIGH);
}

byte readRegister(byte reg) {
  byte data; reg |= 0x80; digitalWrite(csb, LOW); vspi.transfer(reg); data = vspi.transfer(0x00); digitalWrite(csb, HIGH); return data;
}

// ========================================================================
// --- Implementasi Fungsi Logging ---
// ========================================================================

void checkLoggingSwitch() {
  bool currentSwitchState = !digitalRead(LOGGING_SWITCH_PIN);

  if (currentSwitchState != lastLoggingState) {
    vTaskDelay(pdMS_TO_TICKS(50)); // Debounce
    currentSwitchState = !digitalRead(LOGGING_SWITCH_PIN); 

    if (currentSwitchState != lastLoggingState) { 
        lastLoggingState = currentSwitchState;
        if (currentSwitchState) { 
          Serial.println("Switch detected: ON. Calling startLogging().");
          startLogging();
        } else { 
          Serial.println("Switch detected: OFF. Calling stopLogging().");
          stopLogging();
        }
    }
  }
}

void startLogging() {
  if (xSemaphoreTake(loggingMutex, portMAX_DELAY) == pdTRUE) {
    if (!loggingEnabled) { 
      DateTime now = rtc.now(); 
      String yearStr = String(now.year());
      String monthStr = (now.month() < 10 ? "0" : "") + String(now.month());
      String dayStr = (now.day() < 10 ? "0" : "") + String(now.day());
      String hourStr = (now.hour() < 10 ? "0" : "") + String(now.hour());
      String minuteStr = (now.minute() < 10 ? "0" : "") + String(now.minute());
      String secondStr = (now.second() < 10 ? "0" : "") + String(now.second());

      currentLogFileName = "/" + yearStr + monthStr + dayStr + hourStr + minuteStr + secondStr + ".csv";
      
      File file = SD.open(currentLogFileName, FILE_WRITE);
      if (file) {
        file.println("Timestamp,ECG1,ECG2,ECG3,LeadIII,BPM,Classification");
        file.close();
        Serial.println("Logging started and CSV header written to: " + currentLogFileName);
        loggingEnabled = true; 
        sampleCounterGlobal = 0; 
        logBufferIndex = 0; 
        currentLoggingMicros = 0; 
      } else {
        Serial.println("Error: Failed to create/open log file at " + currentLogFileName + " for header!");
      }
    } else {
        Serial.println("Logging already active. Ignoring start request.");
    }
    xSemaphoreGive(loggingMutex); 
  }
}

void stopLogging() {
  if (xSemaphoreTake(loggingMutex, portMAX_DELAY) == pdTRUE) {
    if (loggingEnabled) { 
      loggingEnabled = false; 
      Serial.println("Logging stopped. Writing remaining data to SD Card...");

      if (logBufferIndex > 0) {
        writeBufferToSD();
        logBufferIndex = 0;
      }
      xQueueReset(ecgRawDataQueue); 
      xQueueReset(processedDataQueue); 
      Serial.println("Remaining data written. Queues cleared.");
    } else {
        Serial.println("Logging not active. Ignoring stop request.");
    }
    xSemaphoreGive(loggingMutex); 
  }
}

void writeBufferToSD() {
  File file = SD.open(currentLogFileName, FILE_APPEND);
  if (!file) {
    Serial.println("Error: Failed to open log file for appending in writeBufferToSD! Is SD card removed or corrupted?");
    digitalWrite(LED_INDICATOR_PIN, HIGH); delay(100); digitalWrite(LED_INDICATOR_PIN, LOW); delay(100);
    return;
  }

  for (uint16_t i = 0; i < logBufferIndex; i++) {
    file.printf("%s,%ld,%ld,%ld,%ld,%.2f,%d\n",
                logBuffer[i].timestamp,
                logBuffer[i].ecg1,
                logBuffer[i].ecg2,
                logBuffer[i].ecg3,
                logBuffer[i].leadIII,
                logBuffer[i].bpm, 
                logBuffer[i].classification);
  }
  file.close(); 
  Serial.println("Data batch written to SD Card. " + String(logBufferIndex) + " entries.");
}

// ========================================================================
// --- Setup dan Loop Utama Arduino ---
// ========================================================================

void setup() {
  pinMode(mosi, OUTPUT); pinMode(miso, INPUT); pinMode(sclk, OUTPUT);
  pinMode(csb, OUTPUT); pinMode(dredy, INPUT); 
  pinMode(LED_INDICATOR_PIN, OUTPUT); 
  digitalWrite(LED_INDICATOR_PIN, LOW); 

  pinMode(LOGGING_SWITCH_PIN, INPUT_PULLUP); 

  Serial.begin(115200);
  while (!Serial && millis() < 5000); 
  Serial.println("\nStarting ECG Monitoring System...");

  // Pastikan CS ADS1293 non-aktif saat mencoba inisialisasi SD
  digitalWrite(csb, HIGH); 
  delay(10); // Beri sedikit waktu untuk ADS1293 CS non-aktif sepenuhnya

  if (!rtc.begin()) {
    Serial.println("Error: Couldn't find RTC! Please check wiring. Attempting to set time from compile time.");
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  } else {
    Serial.println("RTC initialized successfully.");
    if (rtc.lostPower()) {
      Serial.println("RTC lost power, setting the time from compile time!");
      rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    }
  }

  // Inisialisasi VSPI untuk ADS1293
  vspi.begin(sclk, miso, mosi, csb);
  vspi.setFrequency(4000000); 
  vspi.setBitOrder(MSBFIRST);
  vspi.setDataMode(SPI_MODE1); 
  digitalWrite(csb, HIGH); // Pastikan CSB non-aktif setelah inisialisasi
  Serial.println("VSPI for ADS1293 initialized.");

  // Inisialisasi HSPI untuk SD card
  hspi.begin(HSPI_SCLK, HSPI_MISO, HSPI_MOSI, HSPI_CS);
  Serial.println("HSPI for SD Card initialized.");

  delay(100); 
  Serial.print("Initializing SD card (CS pin: " + String(HSPI_CS) + ")...");
  // Ini adalah titik kritis. Jika macet di sini, 99% masalah hardware/pin.
  if (!SD.begin(HSPI_CS, hspi)) { 
    Serial.println("FAILED!");
    Serial.println("CRITICAL ERROR: SD Card initialization failed! Check connections, pins, and SD card itself.");
    Serial.println("Make sure HSPI_CS (" + String(HSPI_CS) + ") is correct for your module.");
    while (1) { 
      digitalWrite(LED_INDICATOR_PIN, HIGH); delay(200);
      digitalWrite(LED_INDICATOR_PIN, LOW); delay(200);
    }; 
  }
  Serial.println("SUCCESS. SD Card initialized.");

  ecgRawDataQueue = xQueueCreate(10, sizeof(RawECGData));
  // Perbesar processedDataQueue sedikit jika diperlukan, misal LOG_BUFFER_SIZE * 2
  processedDataQueue = xQueueCreate(LOG_BUFFER_SIZE * 2, sizeof(ProcessedECGData)); 
  loggingMutex = xSemaphoreCreateMutex();

  if (ecgRawDataQueue == NULL || processedDataQueue == NULL || loggingMutex == NULL) {
    Serial.println("FATAL: FreeRTOS object creation failed! Out of memory?");
    while (1) {
      digitalWrite(LED_INDICATOR_PIN, HIGH); delay(100);
      digitalWrite(LED_INDICATOR_PIN, LOW); delay(100);
    };
  }
  Serial.println("FreeRTOS Queues and Mutex created.");

  // --- Inisialisasi TensorFlow Lite ---
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal "
                                         "to supported version %d.",
                                         model->version(), TFLITE_SCHEMA_VERSION);
    while(1); 
  }
  Serial.println("TFLite Model loaded.");

  // BARIS INI YANG DIPERBAIKI! TAMBAHKAN <JUMLAH_OPERASI>
  static tflite::MicroMutableOpResolver<4> op_resolver; // Perbaikan: Tambahkan jumlah operasi

  op_resolver.AddFullyConnected();
  op_resolver.AddLogistic();
  op_resolver.AddRelu();
  op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed!");
    while(1); 
  }
  Serial.println("TFLite Tensors allocated.");

  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) ||
      (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != WINDOW_SIZE) ||
      (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model!");
    while(1); 
  }

  TfLiteTensor* model_output = interpreter->output(0);
  if ((model_output->dims->size != 2) ||
      (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != 3) ||
      (model_output->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad output tensor parameters in model!");
    while(1); 
  }
  Serial.println("TFLite Model input/output verified.");
  // --- End TensorFlow Lite Initialization ---

  DetectionResponder::setup(); 
  setup_ECG(); 

  xTaskCreatePinnedToCore(ecgReadTask, "ECG_Read_Task", 4096, NULL, configMAX_PRIORITIES - 1, NULL, 0);
  xTaskCreatePinnedToCore(processDataTask, "Process_Data_Task", 8192, NULL, configMAX_PRIORITIES - 2, NULL, 1);
  xTaskCreatePinnedToCore(logDataTask, "Log_Data_Task", 8192, NULL, 1, NULL, 0); 

  Serial.println("System initialized successfully. FreeRTOS Tasks created.");
  Serial.println("Ready: Toggle switch on pin " + String(LOGGING_SWITCH_PIN) + " to start/stop logging.");
}

void loop() {
  checkLoggingSwitch(); 
  vTaskDelay(pdMS_TO_TICKS(100)); 
}

// Task Akuisisi Data ECG (Prioritas Tertinggi)
void ecgReadTask(void* pvParameters) {
  (void) pvParameters; 
  RawECGData rawData; 

  for (;;) { 
    while (digitalRead(dredy) == HIGH) {
      vTaskDelay(pdMS_TO_TICKS(1)); 
    }

    readECGStreamingData(&rawData.ch1, &rawData.ch2, &rawData.ch3);

    if (xQueueSend(ecgRawDataQueue, &rawData, pdMS_TO_TICKS(5)) != pdPASS) {
      //Serial.println("ECG Read Task: WARNING! Raw data queue full, data dropped.");
    }
  }
}

// Task Pemrosesan Data (Prioritas Menengah)
void processDataTask(void* pvParameters) {
  (void) pvParameters;
  RawECGData rawData; 
  ProcessedECGData processedData; 
  static unsigned long lastPeakTime = millis(); 

  for (;;) {
    if (xQueueReceive(ecgRawDataQueue, &rawData, portMAX_DELAY) == pdPASS) {
      int32_t currentECG1Filtered = applyFIRFilter((float)rawData.ch1, firBuffer1, firLength);
      int32_t currentECG2Filtered = applyFIRFilter((float)rawData.ch2, firBuffer2, firLength);
      int32_t currentECG3Filtered = applyFIRFilter((float)rawData.ch3, firBuffer3, firLength);
      int32_t currentLeadIII = currentECG2Filtered - currentECG1Filtered;

      Serial.print(currentECG1Filtered); // Lead I
      Serial.print("\t");
      Serial.print(currentECG2Filtered); // Lead II
      Serial.print("\t");
      Serial.print(currentECG3Filtered); // Lead V1
      Serial.print("\t");
      Serial.print(currentLeadIII); // Lead V1
      Serial.print("\t"); 
      Serial.print(bpm); // Lead III
      Serial.print("\t"); 
      Serial.println(classification); // Lead III


      sampleCounterGlobal++;

      peakDetector.addSample(currentECG2Filtered);
      if (peakDetector.isPeakDetected()) {
        unsigned long currTime = millis();
        int peakTimeDiff = currTime - lastPeakTime; 
        lastPeakTime = currTime;

        if (peakTimeDiff > 0 && peakTimeDiff < 2000) { 
          buffer.push(peakTimeDiff);      
          envelope.addValue(peakTimeDiff); 
          bpm = 60000.0f / peakTimeDiff; 
          Serial.printf("Prcs: Peak Detected! RR=%dms, BPM=%.2f\n", peakTimeDiff, bpm);
        } else {
            bpm = 0; 
        }
        peakDetector.clearPeakFlag(); 

        int CLASS_BPM_RULE = 6; 
        if (bpm >= 60.0f && bpm <= 100.0f) { CLASS_BPM_RULE = 0; }
        else if (bpm < 60.0f && bpm >= 30.0f) { CLASS_BPM_RULE = 4; }
        else if (bpm > 100.0f && bpm <= 200.0f) { CLASS_BPM_RULE = 3; }
        else if (bpm < 30.0f || bpm > 200.0f) { CLASS_BPM_RULE = 5; }

        if (buffer.isFilled()) {
          for (int i = 0; i < WINDOW_SIZE; ++i) {
            interpreter->input(0)->data.f[i] = static_cast<float>(buffer[i]) / envelope.getMax();
          }
          TfLiteStatus invoke_status = interpreter->Invoke();
          if (invoke_status == kTfLiteOk) {
            for (int i = 0; i < 3; ++i) { 
              model_output_probabilities[i] = interpreter->output(0)->data.f[i];
            }
            Serial.printf("Prcs: TFLite Output: %.2f, %.2f, %.2f\n", model_output_probabilities[0], model_output_probabilities[1], model_output_probabilities[2]);
          }
        }

        int tempClassification = 6; 
        const float THRESHOLD_PVC = 0.5f; 
        const float THRESHOLD_FLUTTER = 0.5f; 
        int model_predicted_class = -1;
        float max_model_prob = -1.0f;
        for (int i = 0; i < 3; ++i) {
            if (model_output_probabilities[i] > max_model_prob) {
                max_model_prob = model_output_probabilities[i];
                model_predicted_class = i;
            }
        }

        if (CLASS_BPM_RULE == 0) { 
            if (model_predicted_class == 2 && max_model_prob >= THRESHOLD_FLUTTER) { tempClassification = 2; } 
            else if (model_predicted_class == 1 && max_model_prob >= THRESHOLD_PVC) { tempClassification = 1; } 
            else { tempClassification = 0; }
        } else { 
            tempClassification = CLASS_BPM_RULE;
        }
        classification = tempClassification; 
        Serial.printf("Prcs: Final Classification: %d\n", classification);

        DetectionResponder::respondToDetection(classification, loggingEnabled);
      } 

      processedData.sampleCount = sampleCounterGlobal;
      processedData.ecg1Filtered = currentECG1Filtered;
      processedData.ecg2Filtered = currentECG2Filtered;
      processedData.ecg3Filtered = currentECG3Filtered;
      processedData.leadIII = currentLeadIII;
      processedData.bpmValue = bpm;
      processedData.classValue = classification;

      if (xQueueSend(processedDataQueue, &processedData, pdMS_TO_TICKS(100)) != pdPASS) { // Timeout lebih lama
        Serial.println("Process Data Task: WARNING! Processed data queue full, data NOT LOGGED.");
      }
    }
  }
}

// Task Logging Data (Prioritas Rendah)
void logDataTask(void* pvParameters) {
  (void) pvParameters;
  ProcessedECGData dataToLog;

  for (;;) {
    BaseType_t xStatus = xQueueReceive(processedDataQueue, &dataToLog, pdMS_TO_TICKS(100)); // Timeout lebih lama
    if (xStatus == pdPASS) {
      // Serial.println("LogDataTask: Data received from queue."); // Ini bisa membanjiri serial, gunakan hati-hati
      if (xSemaphoreTake(loggingMutex, pdMS_TO_TICKS(500)) == pdTRUE) { // Timeout lebih lama untuk mutex
        if (loggingEnabled) { 
          uint32_t totalSeconds = currentLoggingMicros / 1000000;
          uint16_t totalMillis = (currentLoggingMicros % 1000000) / 1000;
          snprintf(logBuffer[logBufferIndex].timestamp, sizeof(logBuffer[0].timestamp),
                   "%lu.%03u", totalSeconds, totalMillis);

          logBuffer[logBufferIndex].ecg1 = dataToLog.ecg1Filtered;
          logBuffer[logBufferIndex].ecg2 = dataToLog.ecg2Filtered;
          logBuffer[logBufferIndex].ecg3 = dataToLog.ecg3Filtered;
          logBuffer[logBufferIndex].leadIII = dataToLog.leadIII;
          logBuffer[logBufferIndex].bpm = dataToLog.bpmValue;
          logBuffer[logBufferIndex].classification = dataToLog.classValue;

          logBufferIndex++;
          currentLoggingMicros += 3125; 

          if (logBufferIndex >= LOG_BUFFER_SIZE) {
            Serial.println("LogDataTask: Buffer full. Calling writeBufferToSD()...");
            writeBufferToSD();
            logBufferIndex = 0; 
          }
        } else {
          currentLoggingMicros = 0; 
        }
        xSemaphoreGive(loggingMutex); 
      } else {
        Serial.println("LogDataTask: WARNING! Failed to take loggingMutex, data not buffered.");
      }
    } else {
      // Serial.println("LogDataTask: No data in queue or timeout."); // Ini juga bisa membanjiri serial
      vTaskDelay(pdMS_TO_TICKS(10)); // Tingkatkan delay jika tidak ada data untuk mengurangi beban CPU
    }
  }
}