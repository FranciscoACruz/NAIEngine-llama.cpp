#ifndef NAI_TELEMETRY_EVENT_HPP
#define NAI_TELEMETRY_EVENT_HPP

#include <cstdint>

namespace nai {
enum class NaiTelemetryAction : uint8_t {
    AutopistaActivada = 1,
    NodoConflicto     = 2,
    SlotFail          = 3,
    UbatchApplied     = 4,
    Clear             = 5
};

struct alignas(32) NaiTelemetryEvent {
    uint64_t session_handle;
    uint64_t timestamp_us;
    
    uint32_t detail_1;
    uint32_t detail_2;
    
    uint16_t bloque_id;
    uint8_t  action;
    uint8_t  _padding[5];
};

}

#endif