#include "src/nai-telemetry-ring.hpp"
#include <cstdio>
#include <cassert>

struct TestEvent {
    uint64_t ts;
    uint32_t code;
    uint32_t pad;
};

int main() {
    nai::NaiTelemetryRing<TestEvent> ring(8);

    for (uint32_t i = 0; i < 7; ++i) {
        assert(ring.try_push({1000 + i, i, 0}));
    }

    assert(!ring.try_push({9999, 99, 0}));

    TestEvent out;
    for (uint32_t i = 0; i < 7; ++i) {
        assert(ring.try_pop(out));
        assert(out.code == i);
        assert(out.ts == 1000 + i);
    }

    assert(!ring.try_pop(out));

    std::printf("[test-ring] PASS (7 push, 7 pop FIFO, capacidad respetada)\n");
    return 0;
}