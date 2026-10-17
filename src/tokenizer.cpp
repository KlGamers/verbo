#include <algorithm>
#include <cstring>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include "json.hpp"
#include "unicode.hpp"
#include "verbo/verbo.hpp"

namespace verbo {

namespace {

enum class Pre : std::uint8_t { Gpt2, Llama };

struct Added {
  std::string text;
  int id;
  bool special;
  bool lstrip;
  bool rstrip;
};

std::uint64_t pair_key(int a, int b) { return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b); }

std::vector<std::uint32_t> byte_chars() {
  std::vector<std::uint32_t> out(256, 0);
  std::vector<bool> used(256, false);
  for (int b = '!'; b <= '~'; ++b) used[b] = true;
  for (int b = 0xA1; b <= 0xAC; ++b) used[b] = true;
  for (int b = 0xAE; b <= 0xFF; ++b) used[b] = true;
  std::uint32_t nxt = 256;
  for (int b = 0; b < 256; ++b) out[b] = used[b] ? static_cast<std::uint32_t>(b) : nxt++;
  return out;
}

const std::string kGpt2Re = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
const std::string kLlamaRe =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

bool is_other(std::uint32_t cp) { return !uni::is_space(cp) && !uni::is_letter(cp) && !uni::is_number(cp); }

std::uint32_t fold(std::uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return cp + 32;
  if (cp == 0x17F) return 's';
  if (cp == 0x212A) return 'k';
  return cp;
}

}

struct Tokenizer::Impl {
  std::string src;
  Pre pre = Pre::Gpt2;
  int ndig = 1;
  bool digits = false;
  bool nfc = false;
  bool prefix_space = false;
  bool ignore_merges = false;
  std::vector<std::string> pieces;
  std::vector<std::uint8_t> special;
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<std::uint64_t, std::pair<int, int>> merges;
  int byte_id[256];
  std::vector<Added> added;
  mutable std::mutex mu;
  mutable std::unordered_map<std::string, std::vector<int>> cache;

  void parse_pre(const Json& j);
  void split(const std::vector<uni::Cp>& cps, std::size_t b, std::size_t e,
             std::vector<std::pair<std::size_t, std::size_t>>& out) const;
  void split_gpt2(const std::vector<uni::Cp>& cps, std::size_t b, std::size_t e,
                  std::vector<std::pair<std::size_t, std::size_t>>& out) const;
  void split_llama(const std::vector<uni::Cp>& cps, std::size_t b, std::size_t e,
                   std::vector<std::pair<std::size_t, std::size_t>>& out) const;
  void bpe(std::string_view word, std::vector<int>& out) const;
  void encode_text(std::string_view s, std::vector<int>& out) const;
};

void Tokenizer::Impl::parse_pre(const Json& j) {
  if (j.is_null()) throw std::runtime_error("verbo: tokenizer without pre_tokenizer is not supported");
  const std::string ty = j["type"].as_str();
  if (ty == "Sequence") {
    for (const Json& p : j["pretokenizers"].arr()) parse_pre(p);
    return;
  }
  if (ty == "Digits") {
    if (!j.bool_or("individual_digits", false)) throw std::runtime_error("verbo: grouped Digits pre-tokenizer");
    digits = true;
    return;
  }
  if (ty == "ByteLevel") {
    prefix_space = j.bool_or("add_prefix_space", false);
    if (j.bool_or("use_regex", true)) pre = Pre::Gpt2;
    return;
  }
  if (ty == "Split") {
    const std::string re = j["pattern"]["Regex"].as_str();
    if (j.str_or("behavior", "") != "Isolated" || j.bool_or("invert", false)) {
      throw std::runtime_error("verbo: unsupported Split behavior");
    }
    if (re == kGpt2Re) {
      pre = Pre::Gpt2;
      return;
    }
    for (int n = 1; n <= 9; ++n) {
      std::string want = kLlamaRe;
      want.replace(want.find("{1,N}"), 5, n == 1 ? "" : "{1," + std::to_string(n) + "}");
      if (re == want) {
        pre = Pre::Llama;
        ndig = n;
        return;
      }
    }
    throw std::runtime_error("verbo: unsupported pre-tokenizer regex");
  }
  throw std::runtime_error("verbo: unsupported pre-tokenizer " + ty);
}

void Tokenizer::Impl::split_gpt2(const std::vector<uni::Cp>& c, std::size_t b, std::size_t e,
                                 std::vector<std::pair<std::size_t, std::size_t>>& out) const {
  std::size_t i = b;
  auto run = [&](std::size_t k, bool (*pred)(std::uint32_t)) {
    while (k < e && pred(c[k].cp)) ++k;
    return k;
  };
  while (i < e) {
    const std::uint32_t ch = c[i].cp;
    std::size_t j = i;
    if (ch == '\'' && i + 1 < e) {
      const std::uint32_t n1 = c[i + 1].cp;
      const std::uint32_t n2 = i + 2 < e ? c[i + 2].cp : 0;
      if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
        j = i + 2;
      } else if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
        j = i + 3;
      }
    }
    if (j == i) {
      const std::size_t k = ch == ' ' && i + 1 < e ? i + 1 : i;
      const std::uint32_t c2 = c[k].cp;
      if (uni::is_letter(c2)) {
        j = run(k, uni::is_letter);
      } else if (uni::is_number(c2)) {
        j = run(k, uni::is_number);
      } else if (is_other(c2)) {
        j = run(k, is_other);
      }
    }
    if (j == i) {
      const std::size_t r = run(i, uni::is_space);
      if (r == e || r - i == 1) {
        j = r;
      } else {
        j = r - 1;
      }
    }
    out.emplace_back(i, j);
    i = j;
  }
}

void Tokenizer::Impl::split_llama(const std::vector<uni::Cp>& c, std::size_t b, std::size_t e,
                                  std::vector<std::pair<std::size_t, std::size_t>>& out) const {
  auto run = [&](std::size_t k, bool (*pred)(std::uint32_t)) {
    while (k < e && pred(c[k].cp)) ++k;
    return k;
  };
  auto nl = [](std::uint32_t cp) { return cp == '\r' || cp == '\n'; };
  std::size_t i = b;
  while (i < e) {
    const std::uint32_t ch = c[i].cp;
    std::size_t j = i;
    if (ch == '\'' && i + 1 < e) {
      const std::uint32_t n1 = fold(c[i + 1].cp);
      const std::uint32_t n2 = i + 2 < e ? fold(c[i + 2].cp) : 0;
      if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
        j = i + 2;
      } else if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
        j = i + 3;
      }
    }
    if (j == i) {
      if (uni::is_letter(ch)) {
        j = run(i, uni::is_letter);
      } else if (!nl(ch) && !uni::is_number(ch) && i + 1 < e && uni::is_letter(c[i + 1].cp)) {
        j = run(i + 1, uni::is_letter);
      } else if (uni::is_number(ch)) {
        j = i;
        while (j < e && j - i < static_cast<std::size_t>(ndig) && uni::is_number(c[j].cp)) ++j;
      } else {
        const std::size_t k = ch == ' ' && i + 1 < e && is_other(c[i + 1].cp) ? i + 1 : i;
        if (is_other(c[k].cp)) {
          j = run(k, is_other);
          while (j < e && nl(c[j].cp)) ++j;
        }
      }
    }
    if (j == i) {
      const std::size_t r = run(i, uni::is_space);
      std::size_t last = r;
      for (std::size_t k = i; k < r; ++k) {
        if (nl(c[k].cp)) last = k;
      }
      if (last != r) {
        j = last + 1;
      } else if (r == e || r - i == 1) {
        j = r;
      } else {
        j = r - 1;
      }
    }
    out.emplace_back(i, j);
    i = j;
  }
}

void Tokenizer::Impl::split(const std::vector<uni::Cp>& c, std::size_t b, std::size_t e,
                            std::vector<std::pair<std::size_t, std::size_t>>& out) const {
  if (pre == Pre::Gpt2) {
    split_gpt2(c, b, e, out);
  } else {
    split_llama(c, b, e, out);
  }
}

void Tokenizer::Impl::bpe(std::string_view word, std::vector<int>& out) const {
  if (ignore_merges) {
    const auto it = vocab.find(std::string(word));
    if (it != vocab.end()) {
      out.push_back(it->second);
      return;
    }
  }
  const std::size_t n = word.size();
  if (n == 1) {
    const int id = byte_id[static_cast<unsigned char>(word[0])];
    if (id >= 0) out.push_back(id);
    return;
  }
  if (n <= 64) {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(std::string(word));
    if (it != cache.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  std::vector<int> sym(n), prv(n), nxt(n);
  for (std::size_t i = 0; i < n; ++i) {
    sym[i] = byte_id[static_cast<unsigned char>(word[i])];
    prv[i] = static_cast<int>(i) - 1;
    nxt[i] = i + 1 < n ? static_cast<int>(i + 1) : -1;
  }
  struct Cand {
    int rank;
    int pos;
    int a;
    int b;
    bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
  };
  std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> pq;
  auto push = [&](int p) {
    if (p < 0 || nxt[p] < 0) return;
    const auto it = merges.find(pair_key(sym[p], sym[nxt[p]]));
    if (it != merges.end()) pq.push({it->second.first, p, sym[p], sym[nxt[p]]});
  };
  for (std::size_t i = 0; i + 1 < n; ++i) push(static_cast<int>(i));
  while (!pq.empty()) {
    const Cand c = pq.top();
    pq.pop();
    if (sym[c.pos] != c.a || nxt[c.pos] < 0 || sym[nxt[c.pos]] != c.b) continue;
    const int q = nxt[c.pos];
    sym[c.pos] = merges.at(pair_key(c.a, c.b)).second;
    sym[q] = -1;
    nxt[c.pos] = nxt[q];
    if (nxt[q] >= 0) prv[nxt[q]] = c.pos;
    push(prv[c.pos]);
    push(c.pos);
  }
  const std::size_t st = out.size();
  for (int p = 0; p >= 0; p = nxt[p]) {
    if (sym[p] >= 0) out.push_back(sym[p]);
  }
  if (n <= 64) {
    std::lock_guard<std::mutex> lk(mu);
    if (cache.size() > 200000) cache.clear();
    cache.emplace(std::string(word), std::vector<int>(out.begin() + static_cast<std::ptrdiff_t>(st), out.end()));
  }
}

void Tokenizer::Impl::encode_text(std::string_view raw, std::vector<int>& out) const {
  if (raw.empty()) return;
  std::string norm;
  std::string_view s = raw;
  if (nfc) {
    norm = uni::nfc(raw);
    s = norm;
  }
  std::string pfx;
  if (prefix_space && s[0] != ' ') {
    pfx = " ";
    pfx += s;
    s = pfx;
  }
  const auto cps = uni::decode(s);
  auto off = [&](std::size_t k) { return k < cps.size() ? cps[k].off : static_cast<std::uint32_t>(s.size()); };
  std::vector<std::pair<std::size_t, std::size_t>> parts;
  if (digits) {
    std::size_t b = 0;
    for (std::size_t k = 0; k < cps.size(); ++k) {
      if (uni::is_number(cps[k].cp)) {
        if (k > b) split(cps, b, k, parts);
        parts.emplace_back(k, k + 1);
        b = k + 1;
      }
    }
    if (b < cps.size()) split(cps, b, cps.size(), parts);
  } else {
    split(cps, 0, cps.size(), parts);
  }
  for (const auto& p : parts) bpe(s.substr(off(p.first), off(p.second) - off(p.first)), out);
}

Tokenizer Tokenizer::parse(std::string_view json) {
  auto im = std::make_shared<Impl>();
  im->src = std::string(json);
  const Json j = Json::parse(json);
  const Json& m = j["model"];
  if (m.str_or("type", "BPE") != "BPE") throw std::runtime_error("verbo: only BPE tokenizers are supported");
  if (m.bool_or("byte_fallback", false)) throw std::runtime_error("verbo: byte_fallback BPE is not supported");
  im->ignore_merges = m.bool_or("ignore_merges", false);
  const Json& nm = j["normalizer"];
  if (!nm.is_null()) {
    const std::string ty = nm["type"].as_str();
    if (ty == "NFC") {
      im->nfc = true;
    } else if (!(ty == "Sequence" && nm["normalizers"].size() == 0)) {
      throw std::runtime_error("verbo: unsupported normalizer " + ty);
    }
  }
  im->parse_pre(j["pre_tokenizer"]);
  const Json& dec = j["decoder"];
  if (!dec.is_null() && dec["type"].as_str() != "ByteLevel") throw std::runtime_error("verbo: unsupported decoder");

  const auto chars = byte_chars();
  std::unordered_map<std::uint32_t, unsigned char> inv;
  for (int b = 0; b < 256; ++b) inv[chars[b]] = static_cast<unsigned char>(b);
  auto unbyte = [&](const std::string& t) {
    std::string r;
    for (const auto& c : uni::decode(t)) {
      const auto it = inv.find(c.cp);
      if (it != inv.end()) {
        r += static_cast<char>(it->second);
      } else {
        uni::encode(r, c.cp);
      }
    }
    return r;
  };

  int top = -1;
  for (const auto& kv : m["vocab"].obj()) top = std::max(top, static_cast<int>(kv.second.as_int()));
  for (const Json& a : j["added_tokens"].arr()) top = std::max(top, static_cast<int>(a["id"].as_int()));
  im->pieces.assign(static_cast<std::size_t>(top + 1), std::string());
  im->special.assign(static_cast<std::size_t>(top + 1), 0);
  im->vocab.reserve(m["vocab"].size() * 2);
  for (const auto& kv : m["vocab"].obj()) {
    const int id = static_cast<int>(kv.second.as_int());
    std::string b = unbyte(kv.first);
    im->pieces[static_cast<std::size_t>(id)] = b;
    im->vocab.emplace(std::move(b), id);
  }
  for (int b = 0; b < 256; ++b) {
    const auto it = im->vocab.find(std::string(1, static_cast<char>(b)));
    im->byte_id[b] = it == im->vocab.end() ? -1 : it->second;
  }
  const auto& mg = m["merges"].arr();
  im->merges.reserve(mg.size() * 2);
  std::string ma, mb;
  for (std::size_t r = 0; r < mg.size(); ++r) {
    if (mg[r].is_str()) {
      const std::string& s = mg[r].as_str();
      const auto sp = s.find(' ', 1);
      if (sp == std::string::npos) throw std::runtime_error("verbo: malformed merge");
      ma = s.substr(0, sp);
      mb = s.substr(sp + 1);
    } else {
      ma = mg[r].arr().at(0).as_str();
      mb = mg[r].arr().at(1).as_str();
    }
    const std::string ua = unbyte(ma), ub = unbyte(mb);
    const auto ia = im->vocab.find(ua), ib = im->vocab.find(ub), ic = im->vocab.find(ua + ub);
    if (ia == im->vocab.end() || ib == im->vocab.end() || ic == im->vocab.end()) continue;
    im->merges.emplace(pair_key(ia->second, ib->second), std::make_pair(static_cast<int>(r), ic->second));
  }
  for (const Json& a : j["added_tokens"].arr()) {
    Added ad{a["content"].as_str(), static_cast<int>(a["id"].as_int()), a.bool_or("special", false),
             a.bool_or("lstrip", false), a.bool_or("rstrip", false)};
    im->pieces[static_cast<std::size_t>(ad.id)] = ad.text;
    if (ad.special) im->special[static_cast<std::size_t>(ad.id)] = 1;
    if (!ad.text.empty()) im->added.push_back(std::move(ad));
  }
  std::stable_sort(im->added.begin(), im->added.end(),
                   [](const Added& x, const Added& y) { return x.text.size() > y.text.size(); });
  Tokenizer t;
  t.impl_ = std::move(im);
  return t;
}

std::vector<int> Tokenizer::encode(std::string_view text, bool special) const {
  const Impl& im = *impl_;
  std::vector<int> out;
  std::size_t st = 0, i = 0;
  const std::size_t n = text.size();
  while (i < n) {
    const Added* hit = nullptr;
    for (const Added& a : im.added) {
      if (!special && a.special) continue;
      if (a.text[0] == text[i] && text.compare(i, a.text.size(), a.text) == 0) {
        hit = &a;
        break;
      }
    }
    if (!hit) {
      ++i;
      continue;
    }
    std::size_t end = i;
    if (hit->lstrip) {
      while (end > st && uni::is_space(static_cast<unsigned char>(text[end - 1]))) --end;
    }
    im.encode_text(text.substr(st, end - st), out);
    out.push_back(hit->id);
    i += hit->text.size();
    if (hit->rstrip) {
      while (i < n && uni::is_space(static_cast<unsigned char>(text[i]))) ++i;
    }
    st = i;
  }
  im.encode_text(text.substr(st), out);
  return out;
}

std::string Tokenizer::piece(int id) const {
  if (id < 0 || id >= size()) return {};
  return impl_->pieces[static_cast<std::size_t>(id)];
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string out;
  for (const int id : ids) {
    if (id < 0 || id >= size()) continue;
    if (skip_special && impl_->special[static_cast<std::size_t>(id)]) continue;
    out += impl_->pieces[static_cast<std::size_t>(id)];
  }
  return out;
}

int Tokenizer::find(std::string_view tok) const {
  for (const Added& a : impl_->added) {
    if (a.text == tok) return a.id;
  }
  const auto it = impl_->vocab.find(std::string(tok));
  return it == impl_->vocab.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int id) const {
  return id >= 0 && id < size() && impl_->special[static_cast<std::size_t>(id)];
}

int Tokenizer::size() const { return impl_ ? static_cast<int>(impl_->pieces.size()) : 0; }

const std::string& Tokenizer::source() const { return impl_->src; }

std::string Detok::push(int id) {
  pend_ += tok_->piece(id);
  const std::size_t k = uni::valid_prefix(pend_);
  std::string out = pend_.substr(0, k);
  pend_.erase(0, k);
  return out;
}

std::string Detok::flush() {
  std::string out;
  out.swap(pend_);
  return out;
}

}
