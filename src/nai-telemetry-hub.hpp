#ifndef NAI_TELEMETRY_HUB_HPP
#define NAI_TELEMETRY_HUB_HPP

// =============================================================================
// [NAI] Hub de telemetría asíncrona
//
// - El hilo consumidor NO se crea al cargar la librería: arranca la primera vez
//   que se registra un callback con set_callback() y se detiene con stop().
// - Mientras no haya consumidor, emit() descarta el evento (coste ~1 load atómico).
// - El ring es SPSC; varios hilos productores se serializan con un spinlock
//   corto para que emit() sea seguro desde cualquier hilo.
// - El callback se protege con un mutex: puede cambiarse mientras el hilo corre.
// =============================================================================

#include "nai-telemetry-ring.hpp"
#include "nai-telemetry-event.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace nai {

class NaiTelemetryHub {
public:
    using EventCallback = std::function<void(const NaiTelemetryEvent&)>;

    explicit NaiTelemetryHub(uint32_t capacity = 8192) : _ring(capacity) {}

    ~NaiTelemetryHub() { stop(); }

    NaiTelemetryHub(const NaiTelemetryHub&) = delete;
    NaiTelemetryHub& operator=(const NaiTelemetryHub&) = delete;

    // Devuelve false si el evento se descartó (sin consumidor o ring lleno).
    bool emit(const NaiTelemetryEvent& ev) noexcept {
        if (!_running.load(std::memory_order_acquire)) {
            return false;
        }
        while (_push_lock.test_and_set(std::memory_order_acquire)) { /* spin */ }
        const bool ok = _ring.try_push(ev);
        _push_lock.clear(std::memory_order_release);
        return ok;
    }

    // Registra (o reemplaza) el callback y arranca el hilo consumidor si hace falta.
    void set_callback(EventCallback cb) {
        std::lock_guard<std::mutex> lock(_mtx);
        _callback = std::move(cb);
        if (!_worker.joinable()) {
            _running.store(true, std::memory_order_release);
            _worker = std::thread([this]() { this->_loop(); });
        }
    }

    // Detiene el hilo consumidor tras vaciar los eventos pendientes.
    void stop() {
        {
            std::lock_guard<std::mutex> lock(_mtx);
            if (!_worker.joinable()) {
                return;
            }
            _running.store(false, std::memory_order_release);
        }
        _cv.notify_all();
        _worker.join();
    }

private:
    void _loop() {
        NaiTelemetryEvent ev;
        for (;;) {
            bool did_work = false;
            while (_ring.try_pop(ev)) {
                did_work = true;
                std::lock_guard<std::mutex> lock(_mtx);
                if (_callback) {
                    _callback(ev);
                }
            }
            if (!_running.load(std::memory_order_acquire)) {
                break; // ring vacío y se pidió parar
            }
            if (!did_work) {
                std::unique_lock<std::mutex> lock(_mtx);
                _cv.wait_for(lock, std::chrono::milliseconds(50), [this] {
                    return !_running.load(std::memory_order_acquire);
                });
            }
        }
    }

    NaiTelemetryRing<NaiTelemetryEvent> _ring;
    std::atomic<bool>       _running{false};
    std::atomic_flag        _push_lock = ATOMIC_FLAG_INIT;
    std::mutex              _mtx;
    std::condition_variable _cv;
    std::thread             _worker;
    EventCallback           _callback;
};

extern NaiTelemetryHub g_nai_telemetry_hub;

} // namespace nai

#endif
