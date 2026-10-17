#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "verbo/verbo.hpp"

namespace {

using cli::Args;
using cli::load;
using cli::opts;
using cli::report;

void usage() {
  std::fprintf(stderr,
               "verbo: local LLM inference\n\n"
               "  verbo run      -m MODEL -p TEXT [-n 256] [--raw] [-s SYSTEM] [--temp T] [--top-k K]\n"
               "                 [--top-p P] [--min-p P] [--repeat R] [--seed S] [-t N] [-q q8] [--stats]\n"
               "  verbo chat     -m MODEL [-s SYSTEM] [options]\n"
               "  verbo pack     -m HF_DIR -q q8|q4|f32 -o FILE.vrb\n"
               "  verbo bench    -m MODEL [-p 512] [-n 128] [-t N] [-q q8] [--reps 3]\n"
               "  verbo tokenize -m MODEL TEXT\n"
               "  verbo serve    -m MODEL [--host 127.0.0.1] [--port 8080] [-t N] [--ctx 4096]\n"
               "  verbo logits   -m MODEL -i IDS.txt -o OUT.bin [-q f32]\n\n"
               "MODEL is a Hugging Face folder (config.json, tokenizer.json, *.safetensors) or a .vrb file.\n");
}

int cmd_run(const Args& a) {
  auto m = load(a);
  verbo::Session ses(m, a.num("--ctx", 0), a.num("--threads", 0));
  std::string text = a.str("--prompt");
  if (text.empty() && !a.pos.empty()) text = a.pos[0];
  if (text == "-") {
    std::stringstream ss;
    ss << std::cin.rdbuf();
    text = ss.str();
  }
  std::string prompt = text;
  if (!a.has("--raw")) {
    std::vector<verbo::Msg> msgs;
    if (a.has("--system")) msgs.push_back({"system", a.str("--system")});
    msgs.push_back({"user", text});
    prompt = m->chat(msgs);
  }
  auto ids = m->tok().encode(prompt);
  if (m->bos() >= 0) ids.insert(ids.begin(), m->bos());
  const auto st = verbo::generate(ses, ids, opts(a, *m), [](std::string_view s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
    return true;
  });
  std::printf("\n");
  if (a.has("--stats")) report(st);
  return 0;
}

int cmd_chat(const Args& a) {
  auto m = load(a);
  verbo::Session ses(m, a.num("--ctx", 0), a.num("--threads", 0));
  std::vector<verbo::Msg> msgs;
  if (a.has("--system")) msgs.push_back({"system", a.str("--system")});
  const auto o = opts(a, *m);
  std::vector<int> done;
  std::string line;
  for (;;) {
    std::fprintf(stderr, "> ");
    if (!std::getline(std::cin, line)) break;
    if (line.empty()) continue;
    if (line == "/reset") {
      msgs.erase(std::remove_if(msgs.begin(), msgs.end(), [](const verbo::Msg& x) { return x.role != "system"; }),
                 msgs.end());
      ses.rewind(0);
      done.clear();
      continue;
    }
    msgs.push_back({"user", line});
    const auto ids = m->tok().encode(m->chat(msgs));
    std::size_t same = 0;
    while (same < done.size() && same < ids.size() && done[same] == ids[same]) ++same;
    if (same == ids.size()) --same;
    ses.rewind(static_cast<int>(same));
    std::vector<int> rest(ids.begin() + static_cast<std::ptrdiff_t>(same), ids.end());
    std::string reply;
    const auto st = verbo::generate(ses, rest, o, [&](std::string_view s) {
      reply += s;
      std::fwrite(s.data(), 1, s.size(), stdout);
      std::fflush(stdout);
      return true;
    });
    std::printf("\n");
    if (a.has("--stats")) report(st);
    msgs.push_back({"assistant", reply});
    done = ids;
    done.insert(done.end(), st.toks.begin(), st.toks.end());
    done.resize(std::min(done.size(), static_cast<std::size_t>(ses.pos())));
  }
  return 0;
}

int cmd_pack(const Args& a) {
  if (!a.has("--out")) throw std::runtime_error("missing -o FILE.vrb");
  auto m = load(a);
  m->save(a.str("--out"));
  std::fprintf(stderr, "wrote %s\n", a.str("--out").c_str());
  return 0;
}

int cmd_bench(const Args& a) {
  auto m = load(a);
  const int np = a.num("--prompt", 512), ng = a.num("--tokens", 128), reps = a.num("--reps", 3);
  verbo::Session ses(m, np + ng + 8, a.num("--threads", 0));
  std::vector<int> ids(static_cast<std::size_t>(np));
  for (int i = 0; i < np; ++i) ids[static_cast<std::size_t>(i)] = 100 + (i * 7919) % 1000;
  double best_p = 0, best_g = 0;
  for (int r = 0; r < reps; ++r) {
    ses.rewind(0);
    auto t0 = std::chrono::steady_clock::now();
    const float* lg = ses.eval(ids);
    const double pm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    int tok = 0;
    for (int i = 1; i < m->cfg().vocab; ++i) {
      if (lg[i] > lg[tok]) tok = i;
    }
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ng; ++i) {
      lg = ses.eval(&tok, 1);
      tok = static_cast<int>(std::max_element(lg, lg + m->cfg().vocab) - lg);
    }
    const double gm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best_p = std::max(best_p, np * 1000.0 / pm);
    best_g = std::max(best_g, ng * 1000.0 / gm);
  }
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  const double rss = static_cast<double>(ru.ru_maxrss) / 1e6;
#else
  const double rss = static_cast<double>(ru.ru_maxrss) / 1e3;
#endif
  std::printf("threads=%d prompt=%d gen=%d  pp %.1f tok/s  tg %.1f tok/s  rss %.0f MB\n", ses.threads(), np, ng, best_p,
              best_g, rss);
  return 0;
}

int cmd_tokenize(const Args& a) {
  auto m = load(a);
  const std::string text = a.pos.empty() ? a.str("--prompt") : a.pos[0];
  for (const int id : m->tok().encode(text)) {
    std::string pc = m->tok().piece(id);
    std::printf("%d\t", id);
    for (const char ch : pc) {
      if (ch == '\n') {
        std::printf("\\n");
      } else {
        std::putchar(ch);
      }
    }
    std::printf("\n");
  }
  return 0;
}

int cmd_logits(const Args& a) {
  auto m = load(a);
  std::ifstream in(a.str("--input"));
  std::ofstream out(a.str("--out"), std::ios::binary);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::vector<int> ids;
    int v;
    while (ss >> v) ids.push_back(v);
    if (ids.empty()) continue;
    verbo::Session ses(m, static_cast<int>(ids.size()) + 1, a.num("--threads", 0));
    const auto lg = ses.eval_all(ids);
    out.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * sizeof(float)));
  }
  return 0;
}

}

int main(int argc, char** argv) {
  const Args a = cli::parse(argc, argv);
  try {
    if (a.cmd == "run") return cmd_run(a);
    if (a.cmd == "chat") return cmd_chat(a);
    if (a.cmd == "pack") return cmd_pack(a);
    if (a.cmd == "bench") return cmd_bench(a);
    if (a.cmd == "tokenize") return cmd_tokenize(a);
    if (a.cmd == "logits") return cmd_logits(a);
    if (a.cmd == "serve") return cli::cmd_serve(a);
    usage();
    return a.cmd.empty() || a.cmd == "help" || a.cmd == "--help" ? 0 : 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
