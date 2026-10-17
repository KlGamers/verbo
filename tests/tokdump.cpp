#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "json.hpp"
#include "verbo/verbo.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: tokdump tokenizer.json corpus.jsonl out.txt\n");
    return 2;
  }
  std::ifstream tf(argv[1], std::ios::binary);
  std::stringstream ss;
  ss << tf.rdbuf();
  const auto tok = verbo::Tokenizer::parse(ss.str());
  std::ifstream in(argv[2]);
  std::ofstream out(argv[3]);
  std::string ln;
  long n = 0, bad = 0;
  while (std::getline(in, ln)) {
    const std::string txt = verbo::Json::parse(ln).as_str();
    const auto ids = tok.encode(txt);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (i) out << ' ';
      out << ids[i];
    }
    out << '\n';
    ++n;
    if (tok.decode(ids).size() < txt.size() / 2) ++bad;
  }
  std::fprintf(stderr, "%ld texts, %ld short decodes\n", n, bad);
  return 0;
}
