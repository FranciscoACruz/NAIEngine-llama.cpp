#ifndef NAI_TELEMETRY_HUB_HPP
#define NAI_TELEMETRY_HUB_HPP

#include "nai-telemetry-ring.hpp"
#include "nai-telemetry-event.hpp"
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

namespace nai {

class NaiTelemetryHub {
public:
    using EventCallback = std::function<void(const NaiTelemetryEvent&)>;

    NaiTelemetryHub(uint32_t capacity = 8192) 
        : _ring(capacity), _running(true) 
    {
        _worker = std::thread([this]() { this->_loop(); });
    }

    ~NaiTelemetryHub() {
        _running.store(false, std::memory_order_release);
        if (_worker.joinable()) {
            _worker.join();
        }
    }

    NaiTelemetryHub(const NaiTelemetryHub&) = delete;
    NaiTelemetryHub& operator=(const NaiTelemetryHub&) = delete;

    inline void emit(const NaiTelemetryEvent& ev) noexcept {
        _ring.try_push(ev);
    }

    void set_callback(EventCallback cb) {
        _callback = std::move(cb);
    }

private:
    void _loop() {
        NaiTelemetryEvent ev;
        while (_running.load(std::memory_order_acquire)) {
            bool did_work = false;
            for (int i = 0; i < 64; ++i) {
                if (_ring.try_pop(ev)) {
                    did_work = true;
                    if (_callback) {
                        _callback(ev);
                    }
                } else {
                    break;
                }
            }
            
            if (!did_work) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    NaiTelemetryRing<NaiTelemetryEvent> _ring;
    std::atomic<bool> _running;
    std::thread _worker;
    EventCallback _callback;
};

extern NaiTelemetryHub g_nai_telemetry_hub;

}

#endif