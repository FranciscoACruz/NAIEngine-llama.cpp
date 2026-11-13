#include "src/nai-telemetry-hub.hpp"
#include <cstdio>
#include <cassert>
#include <thread>
#include <chrono>
#include <atomic>

int main() {
    nai::NaiTelemetryHub hub(1024);
    std::atomic<int> count{0};
    
    hub.set_callback([&count](const nai::NaiTelemetryEvent& ev) {
        count.fetch_add(1, std::memory_order_relaxed);
    });

    for (int i = 0; i < 10; ++i) {
        hub.emit({(uint64_t)i, 12345, 0, 0, 1, 1, {0}});
    }

    for (int i = 0; i < 50; ++i) {
        if (count.load() >= 10) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    assert(count.load() == 10);
    std::printf("[test-hub] PASS (10 eventos consumidos por el thread)\n");
    return 0;
}