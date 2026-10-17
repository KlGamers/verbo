#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "json.hpp"

namespace cli {

namespace {

struct Req {
  std::string method;
  std::string path;
  std::string body;
};

struct Engine {
  std::shared_ptr<const verbo::Model> mdl;
  std::unique_ptr<verbo::Session> ses;
  std::vector<int> cached;
  std::string name;
  int max_tokens = 512;
  std::mutex mu;
  std::atomic<long> seq{0};
};

bool send_all(int fd, std::string_view s) {
  while (!s.empty()) {
    const ssize_t n = ::send(fd, s.data(), s.size(), 0);
    if (n <= 0) return false;
    s.remove_prefix(static_cast<std::size_t>(n));
  }
  return true;
}

bool read_req(int fd, Req& r) {
  std::string buf;
  char tmp[8192];
  std::size_t hend = std::string::npos;
  while (hend == std::string::npos) {
    const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
    if (n <= 0) return false;
    buf.append(tmp, static_cast<std::size_t>(n));
    hend = buf.find("\r\n\r\n");
    if (buf.size() > (1 << 20) && hend == std::string::npos) return false;
  }
  const std::string head = buf.substr(0, hend);
  const std::size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
  r.method = head.substr(0, sp1);
  r.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
  const std::size_t q = r.path.find('?');
  if (q != std::string::npos) r.path.resize(q);
  std::size_t len = 0;
  std::size_t at = head.find("\r\n");
  while (at != std::string::npos && at < head.size()) {
    const std::size_t nx = head.find("\r\n", at + 2);
    std::string ln = head.substr(at + 2, (nx == std::string::npos ? head.size() : nx) - at - 2);
    std::string low = ln;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return std::tolower(c); });
    if (low.rfind("content-length:", 0) == 0) len = static_cast<std::size_t>(std::strtoull(ln.c_str() + 15, nullptr, 10));
    at = nx;
  }
  if (len > (64u << 20)) return false;
  r.body = buf.substr(hend + 4);
  while (r.body.size() < len) {
    const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
    if (n <= 0) return false;
    r.body.append(tmp, static_cast<std::size_t>(n));
  }
  r.body.resize(len);
  return true;
}

void respond(int fd, int code, const std::string& body, const char* type = "application/json") {
  const char* msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Internal Server Error";
  std::string h = "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + type +
                  "\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + std::to_string(body.size()) +
                  "\r\nConnection: close\r\n\r\n";
  send_all(fd, h + body);
}

std::string err_json(const std::string& m) {
  std::string out = "{\"error\":{\"message\":";
  verbo::json_escape(out, m);
  return out + ",\"type\":\"invalid_request_error\"}}";
}

std::string content_of(const verbo::Json& c) {
  if (c.is_str()) return c.as_str();
  std::string out;
  if (c.is_arr()) {
    for (const verbo::Json& part : c.arr()) {
      if (part.str_or("type", "") == "text") out += part.str_or("text", "");
    }
  }
  return out;
}

std::string chunk(const std::string& id, const std::string& model, long created, const std::string& delta,
                  const char* finish, bool chat, bool role) {
  std::string out = "{\"id\":\"" + id + "\",\"object\":\"" + (chat ? "chat.completion.chunk" : "text_completion") +
                    "\",\"created\":" + std::to_string(created) + ",\"model\":";
  verbo::json_escape(out, model);
  out += ",\"choices\":[{\"index\":0,";
  if (chat) {
    out += "\"delta\":{";
    if (role) out += "\"role\":\"assistant\"";
    if (!delta.empty()) {
      if (role) out += ",";
      out += "\"content\":";
      verbo::json_escape(out, delta);
    }
    out += "}";
  } else {
    out += "\"text\":";
    verbo::json_escape(out, delta);
  }
  out += ",\"finish_reason\":";
  out += finish ? "\"" + std::string(finish) + "\"" : "null";
  return out + "}]}";
}

void complete(Engine& e, int fd, const verbo::Json& j, bool chat) {
  std::string prompt;
  if (chat) {
    std::vector<verbo::Msg> msgs;
    for (const verbo::Json& m : j["messages"].arr()) msgs.push_back({m.str_or("role", "user"), content_of(m["content"])});
    if (msgs.empty()) throw std::runtime_error("messages must not be empty");
    prompt = e.mdl->chat(msgs);
  } else {
    prompt = content_of(j["prompt"]);
  }
  verbo::GenOpts o;
  o.smp = e.mdl->sampling();
  o.max_tokens = static_cast<int>(j.num_or("max_completion_tokens", j.num_or("max_tokens", e.max_tokens)));
  o.smp.temp = static_cast<float>(j.num_or("temperature", o.smp.temp));
  o.smp.top_p = static_cast<float>(j.num_or("top_p", o.smp.top_p));
  o.smp.top_k = static_cast<int>(j.num_or("top_k", o.smp.top_k));
  o.smp.min_p = static_cast<float>(j.num_or("min_p", o.smp.min_p));
  o.smp.repeat = static_cast<float>(j.num_or("repetition_penalty", o.smp.repeat));
  o.smp.seed = static_cast<std::uint64_t>(j.num_or("seed", 0));
  const verbo::Json& stop = j["stop"];
  if (stop.is_str()) o.stop_text.push_back(stop.as_str());
  if (stop.is_arr()) {
    for (const verbo::Json& s : stop.arr()) o.stop_text.push_back(s.as_str());
  }
  const bool stream = j.bool_or("stream", false);
  const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(++e.seq);
  const long created = static_cast<long>(std::time(nullptr));
  const std::string model = j.str_or("model", e.name);

  std::lock_guard<std::mutex> lk(e.mu);
  std::vector<int> ids = e.mdl->tok().encode(prompt);
  if (e.mdl->bos() >= 0) ids.insert(ids.begin(), e.mdl->bos());
  if (static_cast<int>(ids.size()) + 1 > e.ses->ctx()) throw std::runtime_error("prompt exceeds the context window");
  o.max_tokens = std::max(0, std::min(o.max_tokens, e.ses->ctx() - static_cast<int>(ids.size())));
  std::size_t same = 0;
  while (same < e.cached.size() && same < ids.size() && e.cached[same] == ids[same]) ++same;
  if (same == ids.size()) --same;
  if (same > static_cast<std::size_t>(e.ses->pos())) same = static_cast<std::size_t>(e.ses->pos());
  e.ses->rewind(static_cast<int>(same));
  const std::vector<int> rest(ids.begin() + static_cast<std::ptrdiff_t>(same), ids.end());
  std::string text;
  bool alive = true;
  if (stream) {
    send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                 "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (chat) alive = send_all(fd, "data: " + chunk(id, model, created, "", nullptr, true, true) + "\n\n");
  }
  verbo::Stats st;
  if (o.max_tokens > 0) {
    st = verbo::generate(*e.ses, rest, o, [&](std::string_view t) {
      text += t;
      if (stream && alive) alive = send_all(fd, "data: " + chunk(id, model, created, std::string(t), nullptr, chat, false) + "\n\n");
      return alive;
    });
  }
  e.cached = ids;
  e.cached.insert(e.cached.end(), st.toks.begin(), st.toks.end());
  e.cached.resize(std::min(e.cached.size(), static_cast<std::size_t>(e.ses->pos())));
  const char* fin = st.stopped ? "stop" : "length";
  if (stream) {
    if (alive) send_all(fd, "data: " + chunk(id, model, created, "", fin, chat, false) + "\n\ndata: [DONE]\n\n");
    return;
  }
  std::string out = "{\"id\":\"" + id + "\",\"object\":\"" + (chat ? "chat.completion" : "text_completion") +
                    "\",\"created\":" + std::to_string(created) + ",\"model\":";
  verbo::json_escape(out, model);
  out += ",\"choices\":[{\"index\":0,";
  if (chat) {
    out += "\"message\":{\"role\":\"assistant\",\"content\":";
    verbo::json_escape(out, text);
    out += "}";
  } else {
    out += "\"text\":";
    verbo::json_escape(out, text);
  }
  out += ",\"finish_reason\":\"" + std::string(fin) + "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(ids.size()) +
         ",\"completion_tokens\":" + std::to_string(st.gen) +
         ",\"total_tokens\":" + std::to_string(ids.size() + static_cast<std::size_t>(st.gen)) + "}}";
  respond(fd, 200, out);
}

void handle(Engine& e, int fd) {
  Req r;
  if (!read_req(fd, r)) {
    ::close(fd);
    return;
  }
  try {
    if (r.method == "OPTIONS") {
      send_all(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, "
                   "OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    } else if (r.method == "GET" && r.path == "/health") {
      respond(fd, 200, "{\"status\":\"ok\"}");
    } else if (r.method == "GET" && r.path == "/v1/models") {
      std::string out = "{\"object\":\"list\",\"data\":[{\"id\":";
      verbo::json_escape(out, e.name);
      respond(fd, 200, out + ",\"object\":\"model\",\"owned_by\":\"verbo\"}]}");
    } else if (r.method == "POST" && (r.path == "/v1/chat/completions" || r.path == "/v1/completions")) {
      verbo::Json j;
      try {
        j = verbo::Json::parse(r.body);
      } catch (const std::exception& ex) {
        respond(fd, 400, err_json(ex.what()));
        ::close(fd);
        return;
      }
      complete(e, fd, j, r.path == "/v1/chat/completions");
    } else {
      respond(fd, 404, err_json("unknown route " + r.path));
    }
  } catch (const std::exception& ex) {
    respond(fd, 400, err_json(ex.what()));
  }
  ::close(fd);
}

}

int cmd_serve(const Args& a) {
  ::signal(SIGPIPE, SIG_IGN);
  Engine e;
  e.mdl = load(a);
  e.ses = std::make_unique<verbo::Session>(e.mdl, a.num("--ctx", 0), a.num("--threads", 0));
  e.max_tokens = a.num("--tokens", 512);
  e.name = a.str("--name", std::filesystem::path(a.str("--model")).filename().string());
  const std::string host = a.str("--host", "127.0.0.1");
  const int port = a.num("--port", 8080);
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("cannot create socket");
  const int yes = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) throw std::runtime_error("invalid host " + host);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
    throw std::runtime_error("cannot bind " + host + ":" + std::to_string(port));
  }
  if (::listen(fd, 64) != 0) throw std::runtime_error("cannot listen");
  std::fprintf(stderr, "verbo serving %s on http://%s:%d/v1\n", e.name.c_str(), host.c_str(), port);
  for (;;) {
    const int c = ::accept(fd, nullptr, nullptr);
    if (c < 0) continue;
    std::thread([&e, c] { handle(e, c); }).detach();
  }
}

}
