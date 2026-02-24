import base64
import concurrent.futures
import json
import os.path
import subprocess
import time
from urllib.parse import quote_plus

import requests

_API_KEY = None

log = print


def vastai_api_key():
    global _API_KEY  # noqa
    if _API_KEY is None:
        api_key_file = os.path.expanduser("~/.vast_api_key")
        if os.path.exists(api_key_file):
            with open(api_key_file, encoding="utf-8") as reader:
                _API_KEY = reader.read().strip()
    return _API_KEY


def weird_querystring_encode(query_dict):
    parts = []
    for k, v in query_dict.items():
        v = quote_plus(v) if isinstance(v, str) else json.dumps(v)
        parts.append(f"{k}={v}")
    return "&".join(parts)


_BASE_URL = "https://console.vast.ai/api/v0"


def vastai_api_request(api_path, query_dict=None, retries=3):
    if query_dict is None:
        query_dict = {}
    query_dict["api_key"] = vastai_api_key()
    encoded_qstring = weird_querystring_encode(query_dict)
    url = f"{_BASE_URL}/{api_path}?{encoded_qstring}"
    headers = {"Accept": "application/json"}
    for attempt in range(retries):
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 429:
            wait = 2 ** (attempt + 1)
            log(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


def vastai_api_put(api_path, json_data):
    url = f"{_BASE_URL}/{api_path}?api_key={vastai_api_key()}"
    response = requests.put(url, json=json_data, timeout=10)
    response.raise_for_status()
    return response.json()


def vastai_api_delete(api_path):
    url = f"{_BASE_URL}/{api_path}?api_key={vastai_api_key()}"
    response = requests.delete(url, json={}, timeout=60)
    response.raise_for_status()
    return response.json()


def get_offers(query):
    api_path = "bundles"
    default_query = {
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
    }
    combined_query = {**default_query, **query}

    return vastai_api_request(api_path, {"q": combined_query})["offers"]


def get_existing_instances():
    api_path = "instances"
    data = vastai_api_request(api_path, {"owner": "me"})
    return data["instances"]


def rent_instance(
    instance_id,
    docker_image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
    docker_args="",
    disk_gb=10,
):
    machine_config = {
        "client_id": "me",
        "image": docker_image,
        "args": docker_args,
        "env": {},
        "price": None,
        "disk": disk_gb,
        "onstart": "touch ~/.no_auto_tmux; sed -i 's/ClientAliveInterval.*/ClientAliveInterval 120/' /etc/ssh/sshd_config; sed -i 's/ClientAliveCountMax.*/ClientAliveCountMax 10/' /etc/ssh/sshd_config; service ssh restart",
        "runtype": "ssh_direct ssh_proxy",
        "image_login": None,
        "use_jupyter_lab": False,
    }
    api_path = f"asks/{instance_id}/"
    log(f"Renting instance {instance_id}")
    data = vastai_api_put(api_path, machine_config)
    log(data)
    return data


def destroy_instance(instance_id):
    api_path = f"instances/{instance_id}/"
    log(f"Destroying instance {instance_id}")
    data = vastai_api_delete(api_path)
    log(data)
    return data


def get_direct_ssh(instance):
    """Get direct SSH host:port, bypassing the vast.ai proxy."""
    ip = instance.get("public_ipaddr")
    ports = instance.get("ports", {})
    tcp_ports = ports.get("22/tcp", [])
    if ip and tcp_ports:
        return ip, int(tcp_ports[0]["HostPort"])
    return instance["ssh_host"], instance["ssh_port"]


_SSH_OPTS = (
    "-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
)


def _ssh_run(host, port, cmd):
    """Run a command on a remote host via SSH."""
    full_cmd = f"ssh {_SSH_OPTS} -p {port} root@{host} '{cmd}'"
    subprocess.run(full_cmd, shell=True, check=True)


def push_to_instance(local_path, remote_path, remote_host, remote_port, excluded=None):
    """Push code to instance via rsync."""
    t0 = time.time()
    exclude_args = ""
    if excluded:
        exclude_args = " ".join(f"--exclude={e}" for e in excluded)

    ssh_cmd = f"ssh {_SSH_OPTS} -p {remote_port}"
    cmd = f"rsync -az {exclude_args} -e '{ssh_cmd}' {local_path}/ root@{remote_host}:{remote_path}/"
    subprocess.run(cmd, shell=True, check=True)

    elapsed = time.time() - t0
    log(f"push to {remote_host}:{remote_port} completed in {elapsed:.1f}s")


def pull_from_instance(local_path, remote_path, remote_host, remote_port):
    """Pull remote directory from instance via scp tarball."""
    t0 = time.time()

    pack_cmd = (
        f"ssh {_SSH_OPTS} -p {remote_port} root@{remote_host} "
        f"'tar czf /tmp/_pull.tar.gz -C {remote_path} . 2>/dev/null || true'"
    )
    subprocess.run(pack_cmd, shell=True, check=True)

    tarball = f"/tmp/_vast_pull_{remote_host}_{remote_port}.tar.gz"
    scp_cmd = f"scp {_SSH_OPTS} -P {remote_port} root@{remote_host}:/tmp/_pull.tar.gz {tarball}"
    max_attempts = 3
    for attempt in range(max_attempts):
        result = subprocess.run(scp_cmd, shell=True)
        if result.returncode == 0:
            break
        log(
            f"scp pull attempt {attempt + 1} failed (exit {result.returncode}), retrying..."
        )
        time.sleep(10)
    else:
        msg = (
            f"scp pull failed after {max_attempts} attempts (exit {result.returncode})"
        )
        raise RuntimeError(msg)

    os.makedirs(local_path, exist_ok=True)
    subprocess.run(f"tar xzf {tarball} -C {local_path}", shell=True, check=True)

    elapsed = time.time() - t0
    log(f"pull from {remote_host}:{remote_port} completed in {elapsed:.1f}s")
    os.remove(tarball)


def wait_for_ssh(instance, timeout=120, interval=5):
    """Wait until SSH is reachable on the instance (tries both proxy and direct)."""
    candidates = [
        (instance["ssh_host"], instance["ssh_port"]),
    ]
    direct = get_direct_ssh(instance)
    if direct != candidates[0]:
        candidates.append(direct)

    deadline = time.time() + timeout
    while time.time() < deadline:
        for host, port in candidates:
            result = subprocess.run(
                f"ssh {_SSH_OPTS} -o ConnectTimeout=5 -p {port} root@{host} echo ok",
                shell=True,
                capture_output=True,
            )
            if result.returncode == 0:
                log(f"SSH ready on {instance['id']}")
                return
        time.sleep(interval)
    msg = f"SSH not reachable on {instance['id']} after {timeout}s"
    raise TimeoutError(msg)


_BENCH_SCRIPT = """\
import torch, time
if not torch.cuda.is_available():
    print("FAIL:NO_CUDA"); exit(1)
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
torch.mm(a, b); torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    torch.mm(a, b)
torch.cuda.synchronize()
tflops = 50 * 2 * 4096**3 / (time.time() - t0) / 1e12
dl_mbps = 0
try:
    import urllib.request
    t0 = time.time()
    urllib.request.urlretrieve("https://speed.cloudflare.com/__down?bytes=10000000", "/dev/null")
    dl_mbps = 10 * 8 / (time.time() - t0)
except Exception:
    pass
print(f"BENCH:GPU={tflops:.1f}:DL={dl_mbps:.0f}")
"""

_BENCH_B64 = base64.b64encode(_BENCH_SCRIPT.encode()).decode()


def benchmark_instance(instance):
    """Run a quick GPU + download benchmark. Returns (tflops, dl_mbps) or (0, 0) on failure."""
    host, port = get_direct_ssh(instance)
    iid = instance["id"]
    try:
        result = subprocess.run(
            f"ssh {_SSH_OPTS} -o ConnectTimeout=10 -p {port} root@{host} "
            f"'echo {_BENCH_B64} | base64 -d | python3'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in result.stdout.strip().split("\n"):
            if line.startswith("BENCH:"):
                parts = dict(p.split("=") for p in line[6:].split(":"))
                tflops = float(parts["GPU"])
                dl_mbps = float(parts["DL"])
                log(f"  {iid}: GPU={tflops:.1f}TFLOPS  DL={dl_mbps:.0f}Mbps")
                return tflops, dl_mbps
    except subprocess.TimeoutExpired:
        log(f"  {iid}: FAIL  benchmark timed out")
        return 0, 0
    except (OSError, subprocess.SubprocessError) as e:
        log(f"  {iid}: FAIL  {e}")
        return 0, 0
    else:
        log(f"  {iid}: FAIL  no benchmark output")
        if result.stderr.strip():
            log(f"    stderr: {result.stderr.strip()[:200]}")
        return 0, 0


def run_command_on_instances(instances, cmd):
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for instance in instances:
            task = executor.submit(run_command_on_instance, instance, cmd)
            tasks.append(task)
        for future in concurrent.futures.as_completed(tasks):
            future.result()


def run_command_on_instance(instance, cmd, check=True):
    host, port = get_direct_ssh(instance)
    cmd_str = f"ssh {_SSH_OPTS} -p {port} root@{host} '{cmd}'"
    log(f"CMD: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True, check=check)
    return result
