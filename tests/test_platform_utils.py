"""Unit tests for platform_utils._listening_pids (pure parser)."""

from platform_utils import _listening_pids

SAMPLE = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:7860           client.example.invalid:0           LISTENING       1234
  TCP    127.0.0.1:7860         localhost:54321        LISTENING       4321
  TCP    127.0.0.1:54321        localhost:7860         ESTABLISHED     9999
  TCP    192.0.2.30:54322         203.0.113.20:7860       ESTABLISHED     8888
  TCP    0.0.0.0:78601          client.example.invalid:0           LISTENING       7777
  TCP    [::]:7860              client.example.invalid:0           LISTENING       5555
  UDP    0.0.0.0:1900           *:*                                    3333
"""


def test_only_exact_port_listeners():
    assert _listening_pids(SAMPLE, 7860, own_pid=0) == ["1234", "4321", "5555"]


def test_excludes_own_pid():
    assert "4321" not in _listening_pids(SAMPLE, 7860, own_pid=4321)


def test_ignores_established_connections():
    # browser (9999) connected TO the port must never be killed
    pids = _listening_pids(SAMPLE, 7860, own_pid=0)
    assert "9999" not in pids
    assert "8888" not in pids


def test_ignores_other_ports_with_same_prefix():
    assert "7777" not in _listening_pids(SAMPLE, 7860, own_pid=0)


def test_ignores_udp_and_headers():
    pids = _listening_pids(SAMPLE, 1900, own_pid=0)
    assert pids == []


def test_empty_output():
    assert _listening_pids("", 7860, own_pid=0) == []


def test_occupied_port_is_reported_without_terminating_owner():
    import socket
    import pytest
    from platform_utils import cleanup_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="unavailable"):
            cleanup_port(listener.getsockname()[1])
        assert listener.fileno() != -1
