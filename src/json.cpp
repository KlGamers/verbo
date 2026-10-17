#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace verbo {

namespace {

const Json kNull;

void put_utf8(std::string& out, std::uint32_t cp) {
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

}

class JsonParser {
 public:
  explicit JsonParser(std::string_view s) : s_(s) {}

  Json run() {
    Json v = value(0);
    ws();
    if (p_ != s_.size()) fail("trailing data");
    return v;
  }

 private:
  std::string_view s_;
  std::size_t p_ = 0;

  [[noreturn]] void fail(const char* why) const {
    throw std::runtime_error("verbo: invalid json at byte " + std::to_string(p_) + ": " + why);
  }

  void ws() {
    while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\r' || s_[p_] == '\t')) ++p_;
  }

  char peek() {
    ws();
    if (p_ >= s_.size()) fail("unexpected end");
    return s_[p_];
  }

  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++p_;
  }

  bool lit(std::string_view w) {
    if (s_.substr(p_, w.size()) != w) return false;
    p_ += w.size();
    return true;
  }

  Json value(int depth) {
    if (depth > 512) fail("nesting too deep");
    Json v;
    const char c = peek();
    if (c == '{') {
      ++p_;
      v.kind_ = Json::Kind::Obj;
      if (peek() == '}') {
        ++p_;
        return v;
      }
      for (;;) {
        if (peek() != '"') fail("expected key");
        std::string key = str();
        expect(':');
        v.o_.emplace_back(std::move(key), value(depth + 1));
        const char d = peek();
        ++p_;
        if (d == '}') break;
        if (d != ',') fail("expected , or }");
      }
    } else if (c == '[') {
      ++p_;
      v.kind_ = Json::Kind::Arr;
      if (peek() == ']') {
        ++p_;
        return v;
      }
      for (;;) {
        v.a_.push_back(value(depth + 1));
        const char d = peek();
        ++p_;
        if (d == ']') break;
        if (d != ',') fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind_ = Json::Kind::Str;
      v.s_ = str();
    } else if (lit("true")) {
      v.kind_ = Json::Kind::Bool;
      v.b_ = true;
    } else if (lit("false")) {
      v.kind_ = Json::Kind::Bool;
    } else if (lit("null")) {
    } else if (lit("NaN")) {
      v.kind_ = Json::Kind::Num;
      v.n_ = NAN;
    } else if (lit("Infinity")) {
      v.kind_ = Json::Kind::Num;
      v.n_ = INFINITY;
    } else if (lit("-Infinity")) {
      v.kind_ = Json::Kind::Num;
      v.n_ = -INFINITY;
    } else {
      v.kind_ = Json::Kind::Num;
      v.n_ = num();
    }
    return v;
  }

  double num() {
    const std::size_t st = p_;
    if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) ++p_;
    while (p_ < s_.size()) {
      const char c = s_[p_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
        ++p_;
      } else {
        break;
      }
    }
    if (p_ == st) fail("expected value");
    const std::string tmp(s_.substr(st, p_ - st));
    char* end = nullptr;
    const double d = std::strtod(tmp.c_str(), &end);
    if (end != tmp.c_str() + tmp.size()) fail("bad number");
    return d;
  }

  std::uint32_t hex4() {
    if (p_ + 4 > s_.size()) fail("bad escape");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<std::uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<std::uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<std::uint32_t>(c - 'A' + 10);
      } else {
        fail("bad escape");
      }
    }
    return v;
  }

  std::string str() {
    ++p_;
    std::string out;
    for (;;) {
      if (p_ >= s_.size()) fail("unterminated string");
      const char c = s_[p_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p_ >= s_.size()) fail("bad escape");
      const char e = s_[p_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          std::uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && s_.substr(p_, 2) == "\\u") {
            const std::size_t save = p_;
            p_ += 2;
            const std::uint32_t lo = hex4();
            if (lo >= 0xDC00 && lo < 0xE000) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              p_ = save;
            }
          }
          if (cp >= 0xD800 && cp < 0xE000) cp = 0xFFFD;
          put_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }
};

Json Json::parse(std::string_view src) { return JsonParser(src).run(); }

bool Json::as_bool() const {
  if (kind_ != Kind::Bool) throw std::runtime_error("verbo: json value is not a bool");
  return b_;
}

double Json::as_num() const {
  if (kind_ != Kind::Num) throw std::runtime_error("verbo: json value is not a number");
  return n_;
}

std::int64_t Json::as_int() const { return static_cast<std::int64_t>(as_num()); }

const std::string& Json::as_str() const {
  if (kind_ != Kind::Str) throw std::runtime_error("verbo: json value is not a string");
  return s_;
}

const std::vector<Json>& Json::arr() const {
  if (kind_ != Kind::Arr) throw std::runtime_error("verbo: json value is not an array");
  return a_;
}

const std::vector<std::pair<std::string, Json>>& Json::obj() const {
  if (kind_ != Kind::Obj) throw std::runtime_error("verbo: json value is not an object");
  return o_;
}

const Json* Json::find(std::string_view key) const {
  if (kind_ != Kind::Obj) return nullptr;
  for (const auto& kv : o_) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

const Json& Json::operator[](std::string_view key) const {
  const Json* v = find(key);
  return v ? *v : kNull;
}

std::size_t Json::size() const {
  if (kind_ == Kind::Arr) return a_.size();
  if (kind_ == Kind::Obj) return o_.size();
  return 0;
}

double Json::num_or(std::string_view key, double dflt) const {
  const Json* v = find(key);
  return v && v->is_num() ? v->n_ : dflt;
}

std::string Json::str_or(std::string_view key, std::string dflt) const {
  const Json* v = find(key);
  return v && v->is_str() ? v->s_ : dflt;
}

bool Json::bool_or(std::string_view key, bool dflt) const {
  const Json* v = find(key);
  return v && v->kind_ == Kind::Bool ? v->b_ : dflt;
}

void json_escape(std::string& out, std::string_view s) {
  out += '"';
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

std::string Json::dump() const {
  std::string out;
  switch (kind_) {
    case Kind::Null: out = "null"; break;
    case Kind::Bool: out = b_ ? "true" : "false"; break;
    case Kind::Num: {
      char buf[32];
      if (std::isfinite(n_) && n_ == std::floor(n_) && std::fabs(n_) < 1e15) {
        std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(n_));
      } else {
        std::snprintf(buf, sizeof buf, "%.17g", n_);
      }
      out = buf;
      break;
    }
    case Kind::Str: json_escape(out, s_); break;
    case Kind::Arr:
      out = "[";
      for (std::size_t i = 0; i < a_.size(); ++i) {
        if (i) out += ',';
        out += a_[i].dump();
      }
      out += ']';
      break;
    case Kind::Obj:
      out = "{";
      for (std::size_t i = 0; i < o_.size(); ++i) {
        if (i) out += ',';
        json_escape(out, o_[i].first);
        out += ':';
        out += o_[i].second.dump();
      }
      out += '}';
      break;
  }
  return out;
}

}
