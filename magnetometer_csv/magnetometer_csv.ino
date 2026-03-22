/*
 * QMC5883P CSV Output
 * Output format:
 * gx,gy,gz,B
 */

#include <Adafruit_QMC5883P.h>
#include <math.h>

Adafruit_QMC5883P qmc;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  if (!qmc.begin()) {
    Serial.println("ERROR: Failed to find QMC5883P chip");
    while (1) {
      delay(10);
    }
  }

  // Stable configuration for data collection
  qmc.setMode(QMC5883P_MODE_NORMAL);
  qmc.setODR(QMC5883P_ODR_50HZ);
  qmc.setOSR(QMC5883P_OSR_4);
  qmc.setDSR(QMC5883P_DSR_2);
  qmc.setRange(QMC5883P_RANGE_8G);
  qmc.setSetResetMode(QMC5883P_SETRESET_ON);

  // Optional CSV header
  Serial.println("gx,gy,gz,B");
}

void loop() {
  if (qmc.isDataReady()) {
    float gx, gy, gz;

    if (qmc.getGaussField(&gx, &gy, &gz)) {
      float B = sqrt(gx * gx + gy * gy + gz * gz);

      Serial.print(gx, 6);
      Serial.print(",");
      Serial.print(gy, 6);
      Serial.print(",");
      Serial.print(gz, 6);
      Serial.print(",");
      Serial.println(B, 6);
    } else {
      Serial.println("ERROR: Failed to read gauss field");
    }
  }

  delay(20);  // ~50 Hz
}