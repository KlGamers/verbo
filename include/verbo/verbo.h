#ifndef VERBO_H
#define VERBO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define VERBO_API __declspec(dllexport)
#else
#define VERBO_API __attribute__((visibility("default")))
#endif

typedef struct verbo_model verbo_model;
typedef struct verbo_session verbo_session;
typedef int (*verbo_text_cb)(const char* text, size_t len, void* user);

typedef struct {
  int max_tokens;
  float temp;
  int top_k;
  float top_p;
  float min_p;
  float repeat;
  unsigned long long seed;
} verbo_gen_opts;

VERBO_API const char* verbo_last_error(void);
VERBO_API const char* verbo_version(void);

VERBO_API verbo_model* verbo_model_load(const char* path, const char* dtype);
VERBO_API void verbo_model_free(verbo_model* m);
VERBO_API int verbo_model_save(const verbo_model* m, const char* path);
VERBO_API int verbo_model_vocab(const verbo_model* m);
VERBO_API int verbo_model_is_stop(const verbo_model* m, int id);
VERBO_API verbo_gen_opts verbo_model_defaults(const verbo_model* m);

VERBO_API int verbo_tokenize(const verbo_model* m, const char* text, size_t len, int special, int* out, int cap);
VERBO_API int verbo_detokenize(const verbo_model* m, const int* ids, int n, int skip_special, char* out, int cap);
VERBO_API int verbo_chat_prompt(const verbo_model* m, const char* const* roles, const char* const* texts, int n,
                                int gen, char* out, int cap);

VERBO_API verbo_session* verbo_session_new(verbo_model* m, int ctx, int threads);
VERBO_API void verbo_session_free(verbo_session* s);
VERBO_API const float* verbo_session_eval(verbo_session* s, const int* ids, int n);
VERBO_API int verbo_session_pos(const verbo_session* s);
VERBO_API int verbo_session_ctx(const verbo_session* s);
VERBO_API int verbo_session_rewind(verbo_session* s, int pos);
VERBO_API int verbo_generate(verbo_session* s, const int* ids, int n, const verbo_gen_opts* opts, verbo_text_cb cb,
                             void* user);

#ifdef __cplusplus
}
#endif

#endif
