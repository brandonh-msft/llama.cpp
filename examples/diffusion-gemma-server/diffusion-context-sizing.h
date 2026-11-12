#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

namespace diffusion_context {

inline uint64_t add(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

inline uint64_t mul(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

inline bool parse_requested_context(const char * text, int & value) {
    if (!text) {
        value = 0;
        return true;
    }
    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    const int64_t max_padded = (int64_t(std::numeric_limits<int32_t>::max()) / 256) * 256;
    if (errno == ERANGE || end == text || *end != '\0' || parsed < 0 || parsed > max_padded) {
        return false;
    }
    value = int(parsed);
    return true;
}

inline bool explicit_cpu_only(int requested, int gpu_layers) {
    return requested > 0 && gpu_layers == 0;
}

inline bool cached_attention_supported(int gpu_count, bool cpu_only) {
    return gpu_count <= 1 || cpu_only;
}

inline uint64_t padded_context(uint64_t requested) {
    return mul(add(requested, 255) / 256, 256);
}

inline bool fits_unsplit_attention(uint64_t requested, uint64_t heads) {
    const uint64_t padded = padded_context(requested);
    return mul(heads, mul(padded, padded)) <= (uint64_t(1) << 30);
}

struct memory_requirements {
    uint64_t padded_context;
    uint64_t chunk;
    uint64_t prompt_kv;
    uint64_t workspace;
    uint64_t host;
    uint64_t device;
};

inline memory_requirements estimate(
        uint64_t requested, uint64_t heads, uint64_t kv_bytes_per_token,
        uint64_t vocab, uint64_t embedding, uint64_t canvas,
        bool cached, bool layer0_on_gpu, bool cpu_layers, bool gpu_layers) {
    const uint64_t n = padded_context(requested);
    const uint64_t chunk = cached
        ? std::min(n, std::max(uint64_t(256), std::min(uint64_t(2048), (uint64_t(1) << 30) / mul(heads, n))))
        : n;
    const uint64_t kv = cached ? mul(kv_bytes_per_token, n) : 0;
    const uint64_t emb = mul(mul(vocab, embedding), 2);
    const uint64_t logits = mul(mul(vocab, canvas), 4);
    const uint64_t persistent = add(add(kv, emb), cached ? logits : 0);
    const uint64_t scores = mul(mul(mul(heads, n), chunk), 4);
    const uint64_t workspace = add(scores, add(scores / 2, scores / 4));
    // Transposing the embedding stages an F32 source and an F16 destination.
    uint64_t host = add(mul(mul(vocab, embedding), 6), mul(logits, 2));
    if (cpu_layers) host = add(host, workspace);
    if (!layer0_on_gpu) host = add(host, persistent);
    uint64_t device = gpu_layers ? workspace : 0;
    if (layer0_on_gpu) device = add(device, persistent);
    return {n, chunk, kv, workspace, host, device};
}

inline bool fits(uint64_t needed, uint64_t free) {
    return free > 0 && needed <= free - (free / 10) * 3;
}

enum class file_status { ok, missing, error };

inline bool has_controller(const std::string & controllers, const std::string & name) {
    return ("," + controllers + ",").find("," + name + ",") != std::string::npos;
}

inline std::string unescape_mount_path(const std::string & value) {
    std::string result;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 3 < value.size() &&
            value[i + 1] >= '0' && value[i + 1] <= '7' &&
            value[i + 2] >= '0' && value[i + 2] <= '7' &&
            value[i + 3] >= '0' && value[i + 3] <= '7') {
            result += char((value[i + 1] - '0') * 64 + (value[i + 2] - '0') * 8 + value[i + 3] - '0');
            i += 3;
        } else {
            result += value[i];
        }
    }
    return result;
}

inline bool parse_memory_value(const std::string & text, uint64_t & value) {
    std::istringstream input(text);
    std::string token, extra;
    if (!(input >> token) || input >> extra || token.empty() ||
        token.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(token.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

// Account for every enclosing memory limit, including limits above the process's own cgroup.
template <typename Reader>
uint64_t cgroup_available(uint64_t available, const std::string & memberships,
                          const std::string & mounts, Reader read) {
    std::string v1, v2, line;
    std::istringstream groups(memberships);
    while (std::getline(groups, line)) {
        const size_t first = line.find(':');
        const size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (second == std::string::npos) continue;
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::string path = line.substr(second + 1);
        if (controllers.empty()) v2 = path;
        else if (has_controller(controllers, "memory")) v1 = path;
    }
    const bool unified = v1.empty();
    const std::string group = unescape_mount_path(unified ? v2 : v1);
    if (group.empty()) return available;
    if (group[0] != '/' || group.find("/../") != std::string::npos ||
        group == "/.." || (group.size() >= 3 && group.compare(group.size() - 3, 3, "/..") == 0)) return 0;

    std::istringstream mount_lines(mounts);
    while (std::getline(mount_lines, line)) {
        const size_t sep = line.find(" - ");
        if (sep == std::string::npos) continue;
        std::istringstream before(line.substr(0, sep)), after(line.substr(sep + 3));
        std::string id, parent, device, root, target, options, type, source, super;
        if (!(before >> id >> parent >> device >> root >> target >> options) ||
            !(after >> type >> source >> super)) continue;
        if ((unified && type != "cgroup2") ||
            (!unified && (type != "cgroup" || !has_controller(super, "memory")))) continue;
        root = unescape_mount_path(root);
        target = unescape_mount_path(target);
        if (root.empty() || target.empty() || root[0] != '/' || target[0] != '/') continue;
        while (root.size() > 1 && root.back() == '/') root.pop_back();
        while (target.size() > 1 && target.back() == '/') target.pop_back();
        if (root != "/" && group != root && group.compare(0, root.size() + 1, root + "/") != 0) continue;
        const std::string relative = root == "/" ? (group == "/" ? "" : group) : group.substr(root.size());
        std::string path = target == "/" ? (relative.empty() ? "/" : relative) : target + relative;
        const std::string limit_file = unified ? "/memory.max" : "/memory.limit_in_bytes";
        const std::string usage_file = unified ? "/memory.current" : "/memory.usage_in_bytes";
        while (true) {
            std::string limit_text;
            const file_status status = read(path + limit_file, limit_text);
            if (status == file_status::error) return 0;
            if (status == file_status::ok) {
                std::istringstream input(limit_text);
                std::string token, extra;
                if (!(input >> token) || input >> extra) return 0;
                if (token != "max") {
                    uint64_t limit = 0, usage = 0;
                    std::string usage_text;
                    if (!parse_memory_value(token, limit) ||
                        read(path + usage_file, usage_text) != file_status::ok ||
                        !parse_memory_value(usage_text, usage)) return 0;
                    available = std::min(available, limit > usage ? limit - usage : uint64_t(0));
                }
            }
            if (path == target) return available;
            const size_t slash = path.find_last_of('/');
            if (slash == std::string::npos || (target != "/" && slash < target.size())) return 0;
            path = slash == 0 ? "/" : path.substr(0, slash);
        }
    }
    return 0;
}

} // namespace diffusion_context
