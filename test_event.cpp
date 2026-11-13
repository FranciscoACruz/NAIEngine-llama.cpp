#include "src/nai-telemetry-event.hpp"
#include <cstdio>
#include <cassert>

int main() {
    static_assert(sizeof(nai::NaiTelemetryEvent) == 32, "Debe ser 32 bytes");
    static_assert(alignof(nai::NaiTelemetryEvent) == 32, "Debe estar alineado a 32 bytes");

    nai::NaiTelemetryEvent ev{};
    ev.session_handle = 0xDEADBEEF;
    ev.timestamp_us = 123456789;
    ev.bloque_id = 1;
    ev.action = static_cast<uint8_t>(nai::NaiTelemetryAction::AutopistaActivada);
    ev.detail_1 = 4200;
    ev.detail_2 = 2;

    assert(ev.session_handle == 0xDEADBEEF);
    assert(ev.action == 1);

    std::printf("[test-event] PASS (32 bytes, alineacion correcta, lectura/escritura OK)\n");
    return 0;
}