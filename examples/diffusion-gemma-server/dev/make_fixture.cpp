#define main architecture_test_main
#include "/src/tests/test-llama-archs.cpp"
#undef main
#include "/src/src/llama-model.h"

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    init_normal_pool(3407, 0.01f);
    auto metadata = get_gguf_ctx(LLM_ARCH_DIFFUSION_GEMMA, true);
    auto model = get_model(metadata.get(), nullptr, 3407, 0.01f, {});
    gguf_context_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), metadata.get());
    for (const auto & named : model->tensors_by_name) {
        gguf_add_tensor(output.get(), named.second);
    }
    return gguf_write_to_file(output.get(), argv[1], false) ? 0 : 1;
}
