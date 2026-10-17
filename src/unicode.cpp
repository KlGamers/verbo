#include "unicode.hpp"

#include <algorithm>
#include <iterator>

namespace verbo::uni {

namespace {

struct Range {
  std::uint32_t lo;
  std::uint32_t hi;
};

struct Ccc {
  std::uint32_t cp;
  std::uint8_t cls;
};

struct Dec {
  std::uint32_t cp;
  std::uint32_t a;
  std::uint32_t b;
};

struct Comp {
  std::uint32_t a;
  std::uint32_t b;
  std::uint32_t c;
};

#include "unicode_tables.inc"

template <std::size_t N>
bool in(const Range (&tbl)[N], std::uint32_t cp) {
  const auto* it = std::upper_bound(std::begin(tbl), std::end(tbl), cp,
                                    [](std::uint32_t v, const Range& r) { return v < r.lo; });
  if (it == std::begin(tbl)) return false;
  --it;
  return cp <= it->hi;
}

constexpr std::uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr std::uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount,
                        kSCount = kLCount * kNCount;

std::uint8_t ccc(std::uint32_t cp) {
  if (cp < 0x300) return 0;
  const auto* it = std::lower_bound(std::begin(kCcc), std::end(kCcc), cp,
                                    [](const Ccc& c, std::uint32_t v) { return c.cp < v; });
  return it != std::end(kCcc) && it->cp == cp ? it->cls : 0;
}

void decompose(std::uint32_t cp, std::vector<std::uint32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const std::uint32_t si = cp - kSBase;
    out.push_back(kLBase + si / kNCount);
    out.push_back(kVBase + (si % kNCount) / kTCount);
    if (si % kTCount) out.push_back(kTBase + si % kTCount);
    return;
  }
  if (cp >= 0xC0) {
    const auto* it = std::lower_bound(std::begin(kDec), std::end(kDec), cp,
                                      [](const Dec& d, std::uint32_t v) { return d.cp < v; });
    if (it != std::end(kDec) && it->cp == cp) {
      decompose(it->a, out);
      if (it->b) decompose(it->b, out);
      return;
    }
  }
  out.push_back(cp);
}

std::uint32_t compose(std::uint32_t a, std::uint32_t b) {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  }
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount) {
    return a + (b - kTBase);
  }
  const auto* it = std::lower_bound(std::begin(kComp), std::end(kComp), std::make_pair(a, b),
                                    [](const Comp& c, const std::pair<std::uint32_t, std::uint32_t>& v) {
                                      return c.a < v.first || (c.a == v.first && c.b < v.second);
                                    });
  if (it != std::end(kComp) && it->a == a && it->b == b) return it->c;
  return 0;
}

}

std::vector<Cp> decode(std::string_view s) {
  std::vector<Cp> out;
  out.reserve(s.size());
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::uint32_t cp = kBad;
    std::size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if (c >= 0xC2 && c < 0xE0 && i + 1 < n) {
      const auto c1 = static_cast<unsigned char>(s[i + 1]);
      if ((c1 & 0xC0) == 0x80) {
        cp = ((c & 0x1Fu) << 6) | (c1 & 0x3Fu);
        len = 2;
      }
    } else if (c >= 0xE0 && c < 0xF0 && i + 2 < n) {
      const auto c1 = static_cast<unsigned char>(s[i + 1]);
      const auto c2 = static_cast<unsigned char>(s[i + 2]);
      if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
        const std::uint32_t v = ((c & 0x0Fu) << 12) | ((c1 & 0x3Fu) << 6) | (c2 & 0x3Fu);
        if (v >= 0x800 && (v < 0xD800 || v > 0xDFFF)) {
          cp = v;
          len = 3;
        }
      }
    } else if (c >= 0xF0 && c < 0xF5 && i + 3 < n) {
      const auto c1 = static_cast<unsigned char>(s[i + 1]);
      const auto c2 = static_cast<unsigned char>(s[i + 2]);
      const auto c3 = static_cast<unsigned char>(s[i + 3]);
      if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
        const std::uint32_t v = ((c & 0x07u) << 18) | ((c1 & 0x3Fu) << 12) | ((c2 & 0x3Fu) << 6) | (c3 & 0x3Fu);
        if (v >= 0x10000 && v < 0x110000) {
          cp = v;
          len = 4;
        }
      }
    }
    out.push_back({cp, static_cast<std::uint32_t>(i)});
    i += len;
  }
  return out;
}

void encode(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

std::size_t valid_prefix(std::string_view s) {
  std::size_t n = s.size();
  std::size_t back = 0;
  while (back < 4 && back < n && (static_cast<unsigned char>(s[n - 1 - back]) & 0xC0) == 0x80) ++back;
  if (back == n) return n;
  const auto lead = static_cast<unsigned char>(s[n - 1 - back]);
  std::size_t need = 1;
  if (lead >= 0xF0 && lead < 0xF8) {
    need = 4;
  } else if (lead >= 0xE0) {
    need = lead < 0xF0 ? 3 : 1;
  } else if (lead >= 0xC0) {
    need = 2;
  }
  if (need > back + 1) return n - back - 1;
  return n;
}

bool is_letter(std::uint32_t cp) {
  if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
  return cp != kBad && in(kLetter, cp);
}

bool is_number(std::uint32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return cp != kBad && in(kNumber, cp);
}

bool is_space(std::uint32_t cp) {
  if (cp < 0x80) return cp == ' ' || (cp >= 9 && cp <= 13);
  return cp != kBad && in(kSpace, cp);
}

bool is_digit(std::uint32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return cp != kBad && in(kDigit, cp);
}

std::string nfc(std::string_view s) {
  bool ascii = true;
  for (const char c : s) {
    if (static_cast<unsigned char>(c) >= 0x80) {
      ascii = false;
      break;
    }
  }
  if (ascii) return std::string(s);
  const auto cps = decode(s);
  std::string out;
  out.reserve(s.size());
  std::vector<std::uint32_t> d;
  std::size_t i = 0;
  while (i < cps.size()) {
    if (cps[i].cp == kBad) {
      const std::size_t end = i + 1 < cps.size() ? cps[i + 1].off : s.size();
      out.append(s.substr(cps[i].off, end - cps[i].off));
      ++i;
      continue;
    }
    d.clear();
    std::size_t j = i;
    while (j < cps.size() && cps[j].cp != kBad) {
      if (j > i && ccc(cps[j].cp) == 0 && cps[j].cp < 0x300) break;
      decompose(cps[j].cp, d);
      ++j;
    }
    for (std::size_t a = 1; a < d.size(); ++a) {
      const std::uint8_t ca = ccc(d[a]);
      if (!ca) continue;
      std::size_t b = a;
      while (b > 0 && ccc(d[b - 1]) > ca) {
        std::swap(d[b], d[b - 1]);
        --b;
      }
    }
    std::size_t st = 0;
    bool have = !d.empty() && ccc(d[0]) == 0;
    int last = have ? 0 : -1;
    std::size_t w = d.empty() ? 0 : 1;
    for (std::size_t k = 1; k < d.size(); ++k) {
      const std::uint32_t c = d[k];
      const std::uint8_t cc = ccc(c);
      const bool blocked = last != -1 && !(last < cc || (last == 0 && w == st + 1));
      if (have && !blocked) {
        const std::uint32_t m = compose(d[st], c);
        if (m) {
          d[st] = m;
          continue;
        }
      }
      if (cc == 0) {
        st = w;
        have = true;
        last = 0;
      } else {
        last = cc;
      }
      d[w++] = c;
    }
    d.resize(w);
    for (const std::uint32_t c : d) encode(out, c);
    i = j;
  }
  return out;
}

const char* version() { return kUnicodeVersion; }

}
