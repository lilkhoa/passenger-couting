

#include <project-2_inferencing.h>
#include "SimpleTracker.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "Esp.h" 

// CONFIG
#define INPUT_W 160   
#define INPUT_H 160
#define BUFFER_SIZE (INPUT_W * INPUT_H)
#define LED_BUILTIN 33
#define LINE_Y INPUT_H / 2

#define EST_POWER_MW 850.0 

SimpleTracker tracker(5, 30);
uint32_t *image_buffer; 
int total_up = 0;
int total_down = 0;
int frame_counter = 0;

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        uint32_t pixel = image_buffer[offset + i];
        out_ptr[i] = (float)pixel;
    }
    return 0;
}

void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH); 

    Serial.setRxBufferSize(4096);
    Serial.begin(921600); 
    Serial.setTimeout(10000);
    
    if (psramInit()) {
        image_buffer = (uint32_t*)ps_malloc(BUFFER_SIZE * sizeof(uint32_t));
    } else {
        image_buffer = (uint32_t*)malloc(BUFFER_SIZE * sizeof(uint32_t));
    }
    
    if (!image_buffer) {
        while(1) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(100); }
    }
    
    while(Serial.available()) Serial.read();
    
    Serial.println("{\"boot\":\"Ready\"}");
}

void loop() {
    // 1. Heartbeat: Send READY every seconds
    if (Serial.available() < 3) {
        static unsigned long last_time = 0;
        if (millis() - last_time > 1000) { 
            while(Serial.available()) Serial.read(); 
            Serial.println("READY");             
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); 
            last_time = millis();
        }
        return; 
    }

    // 2. Read Heder
    char header[3];
    if (Serial.readBytes(header, 3) != 3) return;

    if (header[0] == 'R' && header[1] == 'S' && header[2] == 'T') {
        Serial.println("{\"status\":\"REBOOTING\"}");
        Serial.flush();
        delay(100);
        ESP.restart();
        return;
    }

    if (header[0] == 'I' && header[1] == 'M' && header[2] == 'G') {
        digitalWrite(LED_BUILTIN, LOW); 
        frame_counter++;

        uint8_t *ptr = (uint8_t*)image_buffer;
        size_t expected_bytes = BUFFER_SIZE * sizeof(uint32_t);
        size_t bytes_read = 0;
        
        unsigned long t_read_start = millis();
        while(bytes_read < expected_bytes && (millis() - t_read_start < 5000)) {
            if(Serial.available()) bytes_read += Serial.readBytes(ptr + bytes_read, expected_bytes - bytes_read);
        }

        if(bytes_read != expected_bytes) { Serial.println("{\"error\":\"incomplete\"}"); return; }

        unsigned long t_start = micros();
        
        ei::signal_t signal;
        signal.total_length = INPUT_W * INPUT_H; 
        signal.get_data = &raw_feature_get_data;
        ei_impulse_result_t result = { 0 };
        run_classifier(&signal, &result, false); 

        // Tracking
        std::vector<Point> points;
        float max_conf = 0.0;
        for (size_t i = 0; i < result.bounding_boxes_count; i++) {
            float val = result.bounding_boxes[i].value;
            if (val > max_conf) max_conf = val;
            if (val > 0.6) points.push_back({(int)result.bounding_boxes[i].x, (int)result.bounding_boxes[i].y});
        }
        std::map<int, TrackedObject> tracked = tracker.update(points);
        
        for (auto const& [id, obj] : tracked) {
             if (!obj.counted && obj.history_y.size() >= 2) {
                int first_y = obj.history_y.front();
                int cur_y = obj.center.y;
                int last_y = obj.history_y.back();

                if ((first_y < LINE_Y && cur_y >= LINE_Y) || (cur_y >= LINE_Y && cur_y < LINE_Y + 10 && last_y < LINE_Y)) {
                    total_up++; 
                    tracker.objects[id].counted = true;
                } 
                else if ((first_y > LINE_Y && cur_y <= LINE_Y) || (cur_y <= LINE_Y && cur_y > LINE_Y - 10 && last_y > LINE_Y)) {
                    total_down++; 
                    tracker.objects[id].counted = true;
                }
            }
        }

        unsigned long t_end = micros();
        
        float process_time_sec = (t_end - t_start) / 1000000.0;
        float fps = 1.0 / process_time_sec;
        
        uint32_t total_ram = (ESP.getHeapSize() - ESP.getFreeHeap()) + (ESP.getPsramSize() - ESP.getFreePsram());
        uint32_t flash_used = ESP.getSketchSize();
        float energy_mj = EST_POWER_MW * process_time_sec;

        Serial.printf("{\"frame\":%d,\"up\":%d,\"down\":%d,\"fps\":%.2f,\"conf\":%.2f,\"ram_used\":%u,\"flash_used\":%u,\"energy_mj\":%.2f,\"time_ms\":%.2f}\n", 
                      frame_counter, total_up, total_down, fps, max_conf, 
                      total_ram, flash_used, energy_mj, process_time_sec * 1000);
        
        digitalWrite(LED_BUILTIN, HIGH); 
    }
}