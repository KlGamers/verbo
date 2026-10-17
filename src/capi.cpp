#include <cstring>
#include <string>

#include "verbo/verbo.h"
#include "verbo/verbo.hpp"

struct verbo_model {
  std::shared_ptr<const verbo::Model> m;
};

struct verbo_session {
  verbo::Session s;
};

namespace {

thread_local std::string g_err;

template <typename F>
auto guard(F&& fn, decltype(fn()) fail) -> decltype(fn()) {
  try {
    g_err.clear();
    return fn();
  } catch (const std::exception& e) {
    g_err = e.what();
  } catch (...) {
    g_err = "verbo: unknown error";
  }
  return fail;
}

int copy_out(const std::string& s, char* out, int cap) {
  const int n = static_cast<int>(s.size());
  if (out && cap > 0) {
    const int k = n < cap - 1 ? n : cap - 1;
    std::memcpy(out, s.data(), static_cast<std::size_t>(k));
    out[k] = '\0';
  }
  return n;
}

}

extern "C" {

const char* verbo_last_error(void) { return g_err.c_str(); }

const char* verbo_version(void) { return "0.1.0"; }

verbo_model* verbo_model_load(const char* path, const char* dtype) {
  return guard(
      [&]() -> verbo_model* {
        const auto dt = verbo::parse_dtype(dtype && *dtype ? dtype : "q8");
        return new verbo_model{verbo::Model::load(path, dt)};
      },
      nullptr);
}

void verbo_model_free(verbo_model* m) { delete m; }

int verbo_model_save(const verbo_model* m, const char* path) {
  return guard(
      [&]() {
        m->m->save(path);
        return 0;
      },
      -1);
}

int verbo_model_vocab(const verbo_model* m) { return m->m->cfg().vocab; }

int verbo_model_is_stop(const verbo_model* m, int id) { return m->m->is_stop(id) ? 1 : 0; }

verbo_gen_opts verbo_model_defaults(const verbo_model* m) {
  const verbo::Sampling s = m->m->sampling();
  return verbo_gen_opts{256, s.temp, s.top_k, s.top_p, s.min_p, s.repeat, 0};
}

int verbo_tokenize(const verbo_model* m, const char* text, size_t len, int special, int* out, int cap) {
  return guard(
      [&]() {
        const auto ids = m->m->tok().encode(std::string_view(text, len), special != 0);
        const int n = static_cast<int>(ids.size());
        if (out) {
          for (int i = 0; i < n && i < cap; ++i) out[i] = ids[static_cast<std::size_t>(i)];
        }
        return n;
      },
      -1);
}

int verbo_detokenize(const verbo_model* m, const int* ids, int n, int skip_special, char* out, int cap) {
  return guard(
      [&]() { return copy_out(m->m->tok().decode(std::vector<int>(ids, ids + n), skip_special != 0), out, cap); }, -1);
}

int verbo_chat_prompt(const verbo_model* m, const char* const* roles, const char* const* texts, int n, int gen,
                      char* out, int cap) {
  return guard(
      [&]() {
        std::vector<verbo::Msg> msgs;
        for (int i = 0; i < n; ++i) msgs.push_back({roles[i], texts[i]});
        return copy_out(m->m->chat(msgs, gen != 0), out, cap);
      },
      -1);
}

verbo_session* verbo_session_new(verbo_model* m, int ctx, int threads) {
  return guard([&]() -> verbo_session* { return new verbo_session{verbo::Session(m->m, ctx, threads)}; }, nullptr);
}

void verbo_session_free(verbo_session* s) { delete s; }

const float* verbo_session_eval(verbo_session* s, const int* ids, int n) {
  return guard([&]() { return s->s.eval(ids, n); }, static_cast<const float*>(nullptr));
}

int verbo_session_pos(const verbo_session* s) { return s->s.pos(); }

int verbo_session_ctx(const verbo_session* s) { return s->s.ctx(); }

int verbo_session_rewind(verbo_session* s, int pos) {
  return guard(
      [&]() {
        s->s.rewind(pos);
        return 0;
      },
      -1);
}

int verbo_generate(verbo_session* s, const int* ids, int n, const verbo_gen_opts* opts, verbo_text_cb cb, void* user) {
  return guard(
      [&]() {
        verbo::GenOpts o;
        o.smp = s->s.model().sampling();
        if (opts) {
          o.max_tokens = opts->max_tokens;
          o.smp.temp = opts->temp;
          o.smp.top_k = opts->top_k;
          o.smp.top_p = opts->top_p;
          o.smp.min_p = opts->min_p;
          o.smp.repeat = opts->repeat;
          o.smp.seed = opts->seed;
        }
        const auto st = verbo::generate(s->s, std::vector<int>(ids, ids + n), o, [&](std::string_view t) {
          return cb ? cb(t.data(), t.size(), user) != 0 : true;
        });
        return st.gen;
      },
      -1);
}

}
