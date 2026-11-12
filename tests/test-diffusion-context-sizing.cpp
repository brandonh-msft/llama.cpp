#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <map>
#include <string>

#include "../examples/diffusion-gemma-server/diffusion-context-sizing.h"

int main() {
    using diffusion_context::estimate;
    int parsed = -1;
    assert(diffusion_context::parse_requested_context(nullptr, parsed) && parsed == 0);
    assert(diffusion_context::parse_requested_context("0", parsed) && parsed == 0);
    assert(diffusion_context::parse_requested_context("65535", parsed) && parsed == 65535);
    assert(!diffusion_context::parse_requested_context("", parsed));
    assert(!diffusion_context::parse_requested_context("8192garbage", parsed));
    assert(!diffusion_context::parse_requested_context("-1", parsed));
    assert(!diffusion_context::parse_requested_context("999999999999999999999999", parsed));
    assert(!diffusion_context::parse_requested_context("2147483647", parsed));

    assert(diffusion_context::explicit_cpu_only(65535, 0));
    assert(!diffusion_context::explicit_cpu_only(0, 0));
    assert(!diffusion_context::explicit_cpu_only(65535, 1));
    assert(!diffusion_context::explicit_cpu_only(65535, -1));
    assert(diffusion_context::cached_attention_supported(
        2, diffusion_context::explicit_cpu_only(65535, 0)));
    assert(!diffusion_context::cached_attention_supported(
        2, diffusion_context::explicit_cpu_only(0, 0)));
    assert(!diffusion_context::cached_attention_supported(
        2, diffusion_context::explicit_cpu_only(65535, 1)));
    assert(diffusion_context::cached_attention_supported(1, false));

    assert(diffusion_context::fits_unsplit_attention(8448, 15));
    assert(!diffusion_context::fits_unsplit_attention(8449, 15));

    const auto rounded = estimate(65535, 16, 1024, 262144, 3072, 256, true, true, false, true);
    assert(rounded.padded_context == 65536);
    assert(rounded.chunk == 1024);
    assert(rounded.prompt_kv == 65536 * 1024);

    const auto large = estimate(262144, 16, 1024, 262144, 3072, 256, true, true, false, true);
    assert(large.chunk == 256);
    assert(large.workspace == uint64_t(7) * 1024 * 1024 * 1024);
    assert(large.prompt_kv == uint64_t(262144) * 1024);
    assert(large.device > large.workspace + large.prompt_kv);

    const auto cpu = estimate(262144, 16, 1024, 262144, 3072, 256, true, false, true, false);
    assert(cpu.device == 0);
    assert(cpu.host > large.host + large.prompt_kv);
    assert(!diffusion_context::fits(cpu.host, cpu.host));
    assert(diffusion_context::fits(cpu.host, cpu.host * 2));

    const auto split = estimate(8192, 16, 1024, 262144, 3072, 256, false, false, true, true);
    assert(split.chunk == 8192);
    assert(split.prompt_kv == 0);
    assert(split.workspace == uint64_t(7) * 1024 * 1024 * 1024);

    using diffusion_context::file_status;
    std::map<std::string, std::string> files = {
        {"/cg/tenant/job/memory.max", "4000\n"},
        {"/cg/tenant/job/memory.current", "3000\n"},
        {"/cg/tenant/memory.max", "2000\n"},
        {"/cg/tenant/memory.current", "1500\n"},
        {"/cg/memory.max", "max\n"},
    };
    const auto read = [&](const std::string & path, std::string & value) {
        const auto it = files.find(path);
        if (it == files.end()) return file_status::missing;
        value = it->second;
        return file_status::ok;
    };
    const std::string v2_mount = "1 0 0:1 / /cg rw - cgroup2 cgroup rw\n";
    assert(diffusion_context::cgroup_available(10000, "0::/tenant/job\n", v2_mount, read) == 500);
    files["/cg/tenant/memory.current"] = "invalid";
    assert(diffusion_context::cgroup_available(10000, "0::/tenant/job\n", v2_mount, read) == 0);
    files["/cg/tenant/memory.current"] = "1500";
    files.erase("/cg/tenant/job/memory.max");
    assert(diffusion_context::cgroup_available(10000, "0::/tenant/job\n", v2_mount, read) == 500);
    files["/mem/job/memory.limit_in_bytes"] = "3000";
    files["/mem/job/memory.usage_in_bytes"] = "2500";
    files["/mem/memory.limit_in_bytes"] = "9000";
    files["/mem/memory.usage_in_bytes"] = "1000";
    const std::string v1_mount = "1 0 0:1 /host /mem rw - cgroup cgroup rw,memory\n";
    assert(diffusion_context::cgroup_available(10000, "2:cpu,memory:/host/job\n", v1_mount, read) == 500);
    assert(diffusion_context::cgroup_available(10000, "2:memory:/different\n", v1_mount, read) == 0);
}
