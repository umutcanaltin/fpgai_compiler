#pragma once
#include <queue>

template<typename T>
class hls_stream {
public:
  void write(const T& x) { q.push(x); }
  T read() { T x = q.front(); q.pop(); return x; }
  bool empty() const { return q.empty(); }
  size_t size() const { return q.size(); }
private:
  std::queue<T> q;
};
