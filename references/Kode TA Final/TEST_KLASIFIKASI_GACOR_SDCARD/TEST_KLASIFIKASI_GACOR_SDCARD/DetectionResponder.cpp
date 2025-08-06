#include "DetectionResponder.h"
#include <Arduino.h>

// Pin untuk RGB LED
const int pinR = 19;
const int pinG = 20;
const int pinB = 21;
const int buzzerPin = 15;  // Pin buzzer

// Variabel untuk kontrol LED dan buzzer
unsigned long previousLedMillis = 0;
bool ledState = false;
const long ledInterval = 500;  // 500ms on/off untuk 1Hz

namespace DetectionResponder {

void setup() {
  // Set semua pin RGB sebagai output
  pinMode(pinR, OUTPUT);
  pinMode(pinG, OUTPUT);
  pinMode(pinB, OUTPUT);
  
  // Set pin buzzer sebagai output
  pinMode(buzzerPin, OUTPUT);

  // Matikan semua warna dan buzzer di awal
  analogWrite(pinR, 0);
  analogWrite(pinG, 0);
  analogWrite(pinB, 0);
  digitalWrite(buzzerPin, LOW);
}

// Fungsi untuk mengatur warna RGB
void setColor(int r, int g, int b) {
  analogWrite(pinR, r);
  analogWrite(pinG, g);
  analogWrite(pinB, b);
}

// Fungsi untuk mengontrol LED berkedip
void controlLed(int classification, bool loggingEnabled) {
  unsigned long currentMillis = millis();
  
  // Hanya aktifkan kedip untuk kondisi tidak normal (classification 1-6) DAN jika loggingEnabled true
  if (classification >= 1 && classification <= 6 && loggingEnabled) {
    if (currentMillis - previousLedMillis >= ledInterval) {
      previousLedMillis = currentMillis;
      ledState = !ledState;
      if (ledState) {
        // Nyalakan LED sesuai warna classification
        switch (classification) {
          case 1: 
            setColor(0, 255, 0); 
            // Serial.println("Case 1 - PVC"); 
            break;   // Hijau - PVC
          case 2: 
            setColor(0, 0, 255); 
            // Serial.println("Case 2 - FLUTTER"); 
            break;   // Biru - FLUTTER
          case 3: 
            setColor(200, 100, 0); 
            // Serial.println("Case 3 - TAKIKARDIA"); 
            break;  // Kuning - TAKIKARDIA
          case 4: 
            setColor(255, 0, 255); 
            // Serial.println("Case 4 - BRADIKARDIA"); 
            break;  // Magenta - BRADIKARDIA
          case 5: 
            setColor(255, 0, 0); 
            // Serial.println("Case 5 - ASISTOL"); 
            break;   // Merah - ASISTOL
          case 6: 
            setColor(255, 125, 125); 
            // Serial.println("Case 6 - UNCLASSIFIED"); 
            break;// Putih - UNCLASSIFIED
        }
      } else {
        // Matikan LED
        setColor(0, 0, 0);
      }
    }
  } else {
    // Matikan LED untuk kondisi normal atau jika logging disabled
    setColor(0, 0, 0);
    ledState = false;
  }
}

// Fungsi untuk mengontrol buzzer (terus menyala)
void controlBuzzer(int classification, bool loggingEnabled) {
  // Hanya aktifkan buzzer untuk kondisi tidak normal (classification 1-6) DAN jika loggingEnabled true
  if (classification >= 1 && classification <= 6 && loggingEnabled) {
    digitalWrite(buzzerPin, HIGH);
  } else {
    // Matikan buzzer untuk kondisi normal atau jika logging disabled
    digitalWrite(buzzerPin, LOW);
  }
}

// Fungsi respon berdasarkan classification dan loggingEnabled
void respondToDetection(int classification, bool loggingEnabled) {
  // Kontrol LED berkedip dengan kondisi loggingEnabled
  controlLed(classification, loggingEnabled);
  
  // Kontrol buzzer (terus menyala) dengan kondisi loggingEnabled
  controlBuzzer(classification, loggingEnabled);
  
}

} // namespace DetectionResponder