// nai-session-table.hpp
//
// [NAI] Tabla de sesiones: el "GPS" de la conversación.
//
// Cada sesión identifica quién (usuario / cliente), dónde (sucursal / dispositivo),
// en qué bloque de trabajo está (bloque_id) y cuándo fue su última actividad.
// El servidor usa la tabla para dar persistencia a la conversación durante un TTL
// (24 h por defecto) y eliminarla al expirar.
//
// Implementación: direccionamiento abierto con sondeo lineal, índice por hashing de
// Fibonacci (multiplicar por 2^64/φ y tomar los bits altos) y lápidas (tombstones)
// para borrar sin romper las cadenas de sondeo. Capacidad siempre potencia de 2;
// se redimensiona cuando ocupadas + lápidas superan el 70 %.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

constexpr uint64_t NAI_GOLDEN_RATIO     = 11400714819323198485ULL; // 2^64 / φ
constexpr uint64_t NAI_EMPTY_HANDLE     = 0ULL;
constexpr uint64_t NAI_TOMBSTONE_HANDLE = 0xFFFFFFFFFFFFFFFFULL;

struct alignas(64) NaiSession {
    uint64_t session_handle;   // hash del identificador de sesión (nunca 0 ni TOMBSTONE)
    uint64_t packed_user_id;
    int64_t  cliente_local_id;
    uint64_t name_hash;
    int64_t  last_activity;    // segundos Unix de la última actividad
    uint32_t device_hash;
    int32_t  sucursal_id;
    uint16_t bloque_id;        // bloque de trabajo actual (reservado, aún sin lógica)
    uint16_t permissions_mask;
    uint8_t  _padding[12];     // 52 bytes de datos + 12 = 64 (una línea de caché)

    NaiSession() { std::memset(this, 0, sizeof(NaiSession)); }
};

static_assert(sizeof(NaiSession) == 64, "NaiSession debe ocupar exactamente una línea de caché");

namespace nai {

// Hash de 64 bits (FNV-1a + mezcla final de splitmix64) para identificadores de sesión.
// Nunca devuelve NAI_EMPTY_HANDLE ni NAI_TOMBSTONE_HANDLE.
inline uint64_t hash_sesion(const std::string & id) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : id) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    if (h == NAI_EMPTY_HANDLE || h == NAI_TOMBSTONE_HANDLE) {
        h = 1;
    }
    return h;
}

class NaiSessionTable {
public:
    explicit NaiSessionTable(size_t capacidad_inicial = 1024) {
        size_t cap = 16;
        while (cap < capacidad_inicial) cap <<= 1;
        reservar(cap);
    }

    // Copia de la sesión si existe.
    bool buscar(uint64_t handle, NaiSession & out) const {
        std::lock_guard<std::mutex> lock(_mtx);
        const size_t i = localizar(handle);
        if (i == NO_ENCONTRADO) return false;
        out = _slots[i];
        return true;
    }

    // Inserta o reemplaza la sesión (clave: s.session_handle).
    void guardar(const NaiSession & s) {
        if (!handle_valido(s.session_handle)) return;
        std::lock_guard<std::mutex> lock(_mtx);
        const size_t i = localizar(s.session_handle);
        if (i != NO_ENCONTRADO) {
            _slots[i] = s;
            return;
        }
        if ((_ocupadas + _lapidas + 1) * 10 > _slots.size() * 7) {
            // si sobran lápidas basta con reconstruir; si está realmente llena, duplicar
            const bool llena = (_ocupadas + 1) * 10 > _slots.size() * 5;
            redimensionar(llena ? _slots.size() * 2 : _slots.size());
        }
        insertar_sin_buscar(s);
    }

    bool eliminar(uint64_t handle) {
        std::lock_guard<std::mutex> lock(_mtx);
        const size_t i = localizar(handle);
        if (i == NO_ENCONTRADO) return false;
        marcar_lapida(i);
        return true;
    }

    // Elimina las sesiones con last_activity + ttl <= ahora y llama a al_expirar(sesión)
    // por cada una (p. ej. para borrar su archivo). Devuelve cuántas expiraron.
    size_t expirar(int64_t ahora, int64_t ttl, const std::function<void(const NaiSession &)> & al_expirar) {
        std::vector<NaiSession> expiradas;
        {
            std::lock_guard<std::mutex> lock(_mtx);
            for (size_t i = 0; i < _slots.size(); ++i) {
                const NaiSession & s = _slots[i];
                if (handle_valido(s.session_handle) && s.last_activity + ttl <= ahora) {
                    expiradas.push_back(s);
                    marcar_lapida(i);
                }
            }
        }
        if (al_expirar) {
            for (const auto & s : expiradas) al_expirar(s);
        }
        return expiradas.size();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(_mtx);
        return _ocupadas;
    }

    size_t capacidad() const {
        std::lock_guard<std::mutex> lock(_mtx);
        return _slots.size();
    }

private:
    static constexpr size_t NO_ENCONTRADO = SIZE_MAX;

    static bool handle_valido(uint64_t h) {
        return h != NAI_EMPTY_HANDLE && h != NAI_TOMBSTONE_HANDLE;
    }

    size_t indice(uint64_t handle) const {
        return static_cast<size_t>((handle * NAI_GOLDEN_RATIO) >> _shift);
    }

    size_t localizar(uint64_t handle) const {
        if (!handle_valido(handle)) return NO_ENCONTRADO;
        const size_t mask = _slots.size() - 1;
        size_t i = indice(handle);
        for (size_t n = 0; n < _slots.size(); ++n, i = (i + 1) & mask) {
            const uint64_t h = _slots[i].session_handle;
            if (h == handle)           return i;
            if (h == NAI_EMPTY_HANDLE) return NO_ENCONTRADO;
            // lápida u otra clave: seguir sondeando
        }
        return NO_ENCONTRADO;
    }

    void marcar_lapida(size_t i) {
        _slots[i] = NaiSession();
        _slots[i].session_handle = NAI_TOMBSTONE_HANDLE;
        --_ocupadas;
        ++_lapidas;
    }

    void insertar_sin_buscar(const NaiSession & s) {
        const size_t mask = _slots.size() - 1;
        size_t i = indice(s.session_handle);
        while (handle_valido(_slots[i].session_handle)) {
            i = (i + 1) & mask;
        }
        if (_slots[i].session_handle == NAI_TOMBSTONE_HANDLE) --_lapidas;
        _slots[i] = s;
        ++_ocupadas;
    }

    void reservar(size_t cap) {
        _slots.assign(cap, NaiSession());
        _shift = 64;
        for (size_t c = cap; c > 1; c >>= 1) --_shift;
        _ocupadas = 0;
        _lapidas  = 0;
    }

    void redimensionar(size_t nueva_cap) {
        std::vector<NaiSession> viejas;
        viejas.swap(_slots);
        reservar(nueva_cap);
        for (const auto & s : viejas) {
            if (handle_valido(s.session_handle)) insertar_sin_buscar(s);
        }
    }

    mutable std::mutex      _mtx;
    std::vector<NaiSession> _slots;
    unsigned                _shift    = 64;
    size_t                  _ocupadas = 0;
    size_t                  _lapidas  = 0;
};

} // namespace nai
