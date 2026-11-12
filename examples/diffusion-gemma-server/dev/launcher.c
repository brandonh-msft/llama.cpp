#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

static wchar_t *quote(wchar_t *out, const wchar_t *arg) {
    *out++ = L'"';
    while (*arg) {
        size_t slashes = 0;
        while (*arg == L'\\') {
            ++slashes;
            ++arg;
        }
        size_t count = (*arg == L'"' || !*arg) ? slashes * 2 : slashes;
        while (count--) *out++ = L'\\';
        if (*arg == L'"') *out++ = L'\\';
        if (*arg) *out++ = *arg++;
    }
    *out++ = L'"';
    *out = 0;
    return out;
}

int wmain(int argc, wchar_t **argv) {
    wchar_t python[32768];
    DWORD python_length = GetEnvironmentVariableW(L"DG_BRIDGE_PYTHON", python, 32768);
    if (!python_length || python_length >= 32768) {
        fputs("diffusion bridge: set DG_BRIDGE_PYTHON to the Studio Python executable\n", stderr);
        return 2;
    }
    wchar_t script[32768];
    DWORD length = GetModuleFileNameW(NULL, script, 32768);
    if (!length || length >= 32768 || argc != 2) {
        fputs("diffusion bridge: expected --capabilities or a GGUF path\n", stderr);
        return 2;
    }
    wchar_t *separator = wcsrchr(script, L'\\');
    if (!separator || (size_t)(separator - script) + 11 >= 32768) {
        fputs("diffusion bridge: launcher directory path is too long or invalid\n", stderr);
        return 2;
    }
    wcscpy(separator + 1, L"bridge.py");
    wchar_t pid[32];
    swprintf(pid, 32, L"%lu", (unsigned long)GetCurrentProcessId());
    size_t capacity = 2 * (wcslen(python) + wcslen(script) + wcslen(argv[1])) + 128;
    if (capacity > 32767) {
        fputs("diffusion bridge: command line exceeds the Windows length limit\n", stderr);
        return 2;
    }
    wchar_t *command = calloc(capacity, sizeof(wchar_t));
    if (!command) {
        fputs("diffusion bridge: command line allocation failed\n", stderr);
        return 2;
    }
    wchar_t *cursor = quote(command, python);
    wcscpy(cursor, L" -u ");
    cursor += 4;
    cursor = quote(cursor, script);
    *cursor++ = L' ';
    cursor = quote(cursor, pid);
    *cursor++ = L' ';
    quote(cursor, argv[1]);
    STARTUPINFOW startup = {0};
    PROCESS_INFORMATION process = {0};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE null_input = INVALID_HANDLE_VALUE;
    if (wcscmp(argv[1], L"--capabilities") == 0) {
        SECURITY_ATTRIBUTES security = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
        null_input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (null_input == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "diffusion bridge: opening NUL failed: %lu\n", GetLastError());
            free(command);
            return 2;
        }
        startup.hStdInput = null_input;
    }
    BOOL created = CreateProcessW(python, command, NULL, NULL, TRUE,
                                 CREATE_NO_WINDOW, NULL, NULL, &startup, &process);
    DWORD create_error = created ? 0 : GetLastError();
    if (null_input != INVALID_HANDLE_VALUE) CloseHandle(null_input);
    free(command);
    if (!created) {
        fprintf(stderr, "diffusion bridge: CreateProcess failed: %lu\n", create_error);
        return 2;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD status = 2;
    GetExitCodeProcess(process.hProcess, &status);
    CloseHandle(process.hProcess);
    return (int)status;
}
