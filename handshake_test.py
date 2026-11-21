import socket, ssl, time, statistics as st

host = "animated-guacamole-v6rpvvg5v9943w4wq-8000.app.github.dev"
ctx = ssl.create_default_context()
dns, tcp, tls = [], [], []
for _ in range(20):
    t0 = time.perf_counter()
    ip = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    t1 = time.perf_counter()
    s = socket.create_connection((ip, 443), timeout=10)
    t2 = time.perf_counter()
    w = ctx.wrap_socket(s, server_hostname=host)
    t3 = time.perf_counter()
    w.close()
    dns.append((t1 - t0) * 1000); tcp.append((t2 - t1) * 1000); tls.append((t3 - t2) * 1000)

for name, v in (("DNS lookup", dns), ("TCP connect", tcp), ("TLS handshake", tls)):
    print(f"{name:14s} median {st.median(v):7.1f} ms   max {max(v):7.1f} ms")