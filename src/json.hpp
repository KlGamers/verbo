#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace verbo {

class Json {
 public:
  enum class Kind : std::uint8_t { Null, Bool, Num, Str, Arr, Obj };

  static Json parse(std::string_view src);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool is_num() const { return kind_ == Kind::Num; }
  bool is_str() const { return kind_ == Kind::Str; }
  bool is_arr() const { return kind_ == Kind::Arr; }
  bool is_obj() const { return kind_ == Kind::Obj; }

  bool as_bool() const;
  double as_num() const;
  std::int64_t as_int() const;
  const std::string& as_str() const;
  const std::vector<Json>& arr() const;
  const std::vector<std::pair<std::string, Json>>& obj() const;

  const Json* find(std::string_view key) const;
  const Json& operator[](std::string_view key) const;
  std::size_t size() const;

  double num_or(std::string_view key, double dflt) const;
  std::string str_or(std::string_view key, std::string dflt) const;
  bool bool_or(std::string_view key, bool dflt) const;

  std::string dump() const;

 private:
  friend class JsonParser;
  Kind kind_ = Kind::Null;
  bool b_ = false;
  double n_ = 0.0;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;
};

void json_escape(std::string& out, std::string_view s);

}
