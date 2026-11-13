#ifndef NAI_TELEMETRY_RING_HPP
#define NAI_TELEMETRY_RING_HPP

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace nai {

template<typename Event>
class NaiTelemetryRing {
    static_assert(std::is_trivially_copyable<Event>::value,
                  "Event debe ser trivially copyable");
    static_assert(alignof(Event) <= 64,
                  "Event no debe exceder alineacion de linea de cache");

public:
    explicit NaiTelemetryRing(uint32_t capacity)
        : _capacity(capacity)
    {
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument(
                "NaiTelemetryRing: capacity debe ser potencia de 2 > 0");
        }
        _capacity_mask = capacity - 1;
        _buffer = new Event[capacity]();
        _head.store(0, std::memory_order_relaxed);
        _tail.store(0, std::memory_order_relaxed);
    }

    ~NaiTelemetryRing() {
        delete[] _buffer;
    }

    NaiTelemetryRing(const NaiTelemetryRing&) = delete;
    NaiTelemetryRing& operator=(const NaiTelemetryRing&) = delete;

    bool try_push(const Event& ev) noexcept {
        const uint32_t h = _head.load(std::memory_order_relaxed);
        const uint32_t t = _tail.load(std::memory_order_acquire);

        if (((h + 1) & _capacity_mask) == (t & _capacity_mask)) {
            return false;
        }

        _buffer[h & _capacity_mask] = ev;
        _head.store((h + 1) & _capacity_mask, std::memory_order_release);
        return true;
    }

    bool try_pop(Event& out) noexcept {
        const uint32_t t = _tail.load(std::memory_order_relaxed);
        const uint32_t h = _head.load(std::memory_order_acquire);

        if (t == h) {
            return false;
        }

        out = _buffer[t & _capacity_mask];
        _tail.store((t + 1) & _capacity_mask, std::memory_order_release);
        return true;
    }

    uint32_t capacity() const noexcept { return _capacity; }

private:
    Event*   _buffer        = nullptr;
    uint32_t _capacity      = 0;
    uint32_t _capacity_mask = 0;

    alignas(64) std::atomic<uint32_t> _head{0};
    alignas(64) std::atomic<uint32_t> _tail{0};
};

}

#endif