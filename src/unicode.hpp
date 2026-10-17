#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace verbo::uni {

constexpr std::uint32_t kBad = 0xFFFFFFFFu;

struct Cp {
  std::uint32_t cp;
  std::uint32_t off;
};

std::vector<Cp> decode(std::string_view s);
void encode(std::string& out, std::uint32_t cp);
std::size_t valid_prefix(std::string_view s);

bool is_letter(std::uint32_t cp);
bool is_number(std::uint32_t cp);
bool is_space(std::uint32_t cp);
bool is_digit(std::uint32_t cp);

std::string nfc(std::string_view s);
const char* version();

}
