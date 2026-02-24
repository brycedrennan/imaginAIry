import concurrent.futures
import os
import time

from .vastai import (
    benchmark_instance,
    destroy_instance,
    get_direct_ssh,
    get_existing_instances,
    get_offers,
    pull_from_instance,
    push_to_instance,
    rent_instance,
    run_command_on_instance,
    run_command_on_instances,
    wait_for_ssh,
)

log = print


def find_best_offers(n=10, blocked_machine_ids=None):
    if n == 0:
        return []
    dlperf_min = 70
    query = {
        "reliability": {"gt": 0.9},
        "inet_down": {"gt": 200},
        "inet_up": {"gt": 200},
        "dph_total": {"lt": 1},
        "cuda_max_good": {"gte": 12.4},
        "gpu_ram": {"gte": 24000},
        "dlperf": {"gte": dlperf_min},
        "allocated_storage": 19.5,
        "disk_space": {"gte": 50},
        "direct_port_count": {"gte": 1},
    }
    offers = get_offers(query)
    offers = [
        o
        for o in offers
        if o.get("geolocation", "")
        and (o["geolocation"].endswith("US") or o["geolocation"].endswith("CA"))
    ]
    if blocked_machine_ids:
        offers = [o for o in offers if o["machine_id"] not in blocked_machine_ids]
    offers = [o for o in offers if o["dlperf"] / o["num_gpus"] > dlperf_min]
    offers.sort(key=lambda o: o["dph_total"])
    print(f"Found {len(offers)} offers")
    return offers[:n]


def get_testing_cluster(
    n=1, extra=2, startup_timeout=60 * 15, blocked_machine_ids=None
):
    """Rent n+extra instances, benchmark all, keep the best n, destroy the rest."""
    rent_count = n + extra

    # Rent n+extra, but only block until n are running
    _rent_instances(rent_count, startup_timeout, blocked_machine_ids)
    _wait_for_running(n, startup_timeout)

    # Give stragglers 60s more, then proceed with whatever we have
    deadline = time.time() + 60
    while time.time() < deadline:
        instances = get_existing_instances()
        running = [i for i in instances if i["actual_status"] == "running"]
        if len(running) >= rent_count:
            break
        time.sleep(10)
    else:
        running = [
            i for i in get_existing_instances() if i["actual_status"] == "running"
        ]
        log(f"Proceeding with {len(running)}/{rent_count} instances")

    # Kill any still not running
    for i in get_existing_instances():
        if i["actual_status"] != "running":
            destroy_instance(i["id"])

    # Wait for SSH + benchmark in parallel
    log(f"Waiting for SSH on {len(running)} instances...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(running)) as pool:
        list(pool.map(wait_for_ssh, running))

    log("Benchmarking instances...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(running)) as pool:
        scores = list(pool.map(benchmark_instance, running))

    # Rank by GPU TFLOPS, keep best n
    ranked = sorted(zip(running, scores), key=lambda x: x[1][0], reverse=True)
    keep = [inst for inst, _score in ranked[:n]]
    trash = [inst for inst, _score in ranked[n:]]

    for inst, (tflops, dl_mbps) in ranked:
        tag = "KEEP" if inst in keep else "DROP"
        host, port = get_direct_ssh(inst)
        log(
            f"  {inst['id']}: {tag}  GPU={tflops:.1f}  DL={dl_mbps:.0f}  ssh -p {port} root@{host}"
        )

    for inst in trash:
        destroy_instance(inst["id"])

    return keep


def _rent_instances(n, startup_timeout, blocked_machine_ids):
    """Rent instances until we have n live (running or loading)."""
    instances = get_existing_instances()
    live = [
        i for i in instances if i["actual_status"] in ("running", "loading", "created")
    ]
    needed = max(n - len(live), 0)
    if needed > 0:
        offers = find_best_offers(needed, blocked_machine_ids=blocked_machine_ids)
        for o in offers:
            log(
                f"{o['num_gpus']}x {o['gpu_name']} - "
                f"id:{o['id']}, "
                f"${o['dph_total']:.3f}/hr, "
                f"gpu_ram:{o.get('gpu_ram', 0) / 1024:.0f}GB, "
                f"inet_down:{o.get('inet_down', '?')}Mbps, "
                f"inet_up:{o.get('inet_up', '?')}Mbps, "
                f"dlperf:{o.get('dlperf', '?')}"
            )
        for o in offers:
            rent_instance(o["id"], disk_gb=50)


def _wait_for_running(n, startup_timeout):
    """Block until at least n instances are running."""
    while True:
        instances = get_existing_instances()
        running = [i for i in instances if i["actual_status"] == "running"]
        if len(running) >= n:
            return

        # Kill instances stuck too long
        for instance in instances:
            if (
                instance["actual_status"] != "running"
                and time.time() - instance["start_date"] > startup_timeout
            ):
                log(f"Killing stale instance {instance['id']}")
                destroy_instance(instance["id"])

        for i in instances:
            log(f"  {i['id']}: {i['actual_status']}")
        time.sleep(30)


def get_ready_cluster(n, extra=2, blocked_machine_ids=None):
    t0 = time.time()
    instances = get_testing_cluster(
        n, extra=extra, blocked_machine_ids=blocked_machine_ids
    )
    t_provision = time.time() - t0

    t_rsync_start = time.time()
    excluded = [
        "tests/test_output",
        "tests/test_cluster_output",
        "tests/test_output_local_cuda",
        "*.pyc",
        "*__pycache__",
        "outputs",
        "build",
        ".venv",
        ".git",
        ".github",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".coveragerc",
        ".dockerignore",
        ".gitignore",
        ".python-version",
        "*DS_Store",
        "*.egg-info",
        "*.safetensors",
        "*.ckpt",
        "*.bin",
        "*.ipynb",
        "assets",
        "downloads",
        "docs",
        "other",
        "prolly_delete",
    ]
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for instance in instances:
            host, port = get_direct_ssh(instance)
            task = executor.submit(
                push_to_instance,
                local_path="./",
                remote_path="./project",
                remote_host=host,
                remote_port=port,
                excluded=excluded,
            )
            tasks.append(task)
        for future in concurrent.futures.as_completed(tasks):
            future.result()
    t_rsync = time.time() - t_rsync_start

    t_setup_start = time.time()
    run_command_on_instances(
        instances,
        "cd project && chmod +x ./tests/vast/worker_setup.sh && ./tests/vast/worker_setup.sh",
    )
    t_setup = time.time() - t_setup_start

    pull_outputs(instances)

    t_total = time.time() - t0

    for i in instances:
        host, port = get_direct_ssh(i)
        summary = (
            f"{i['id']}: {i['actual_status']}\n"
            f'ssh -o "StrictHostKeyChecking=no" -p {port} root@{host}'
        )
        print(summary)

    total_dph = sum(i.get("dph_total", 0) for i in instances)
    cost = total_dph * (t_total / 3600)

    log(f"\n{'=' * 44}")
    log("  Cluster Ready — Performance Summary")
    log(f"{'=' * 44}")
    log(f"  provision instances: {t_provision:.1f}s")
    log(f"  rsync to {len(instances)} instances: {t_rsync:.1f}s")
    log(f"  worker setup:       {t_setup:.1f}s")
    log("  ---")
    log(f"  total:              {t_total:.1f}s")
    log(f"  cost:               ${cost:.4f} ({len(instances)} x ${total_dph:.2f}/hr)")
    log(f"{'=' * 44}\n")

    return instances


def pull_outputs(instances):
    """Pull generated outputs back from each instance."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        tasks = []
        for instance in instances:
            host, port = get_direct_ssh(instance)
            task = executor.submit(
                pull_from_instance,
                local_path="./outputs/",
                remote_path="./project/outputs/",
                remote_host=host,
                remote_port=port,
            )
            tasks.append(task)
        for future in concurrent.futures.as_completed(tasks):
            future.result()


def run_distributed_tests(instances):
    t0 = time.time()
    os.makedirs("./tests/test_cluster_output", exist_ok=True)

    def run_test(instance, instance_num, instance_count):
        iid = instance["id"]
        logfile = f"test_{instance_num}of{instance_count}_{iid}.log"

        # pipefail ensures tee doesn't mask pytest's exit code.
        cmd = (
            f"cd project && set -o pipefail && "
            f"pytest -ra --subset {instance_num}/{instance_count} "
            f"2>&1 | tee tests/test_output/{logfile}"
        )
        t_start = time.time()
        result = run_command_on_instance(instance, cmd, check=False)
        elapsed = time.time() - t_start

        host, port = get_direct_ssh(instance)
        pull_from_instance(
            local_path="./tests/test_cluster_output/",
            remote_path="./project/tests/test_output/",
            remote_host=host,
            remote_port=port,
        )

        passed = result.returncode == 0
        return {
            "instance_id": iid,
            "instance_num": instance_num,
            "passed": passed,
            "returncode": result.returncode,
            "elapsed": elapsed,
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as executor:
        futures = {}
        for i, instance in enumerate(instances, 1):
            fut = executor.submit(run_test, instance, i, len(instances))
            futures[fut] = instance

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except (OSError, RuntimeError) as e:
                inst = futures[future]
                log(f"  {inst['id']}: ERROR — {e}")
                results.append(
                    {
                        "instance_id": inst["id"],
                        "instance_num": 0,
                        "passed": False,
                        "returncode": -1,
                        "elapsed": 0,
                    }
                )

    elapsed = time.time() - t0
    total_dph = sum(i.get("dph_total", 0) for i in instances)
    cost = total_dph * (elapsed / 3600)

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    log(f"\n{'=' * 44}")
    log("  Distributed Tests — Results")
    log(f"{'=' * 44}")
    for r in sorted(results, key=lambda x: x["instance_num"]):
        status = "PASS" if r["passed"] else f"FAIL (exit {r['returncode']})"
        log(
            f"  shard {r['instance_num']}/{len(instances)}: {status}  {r['elapsed']:.0f}s  [{r['instance_id']}]"
        )
    log("  ---")
    log(f"  {passed} passed, {failed} failed across {len(instances)} instances")
    log(f"  wall time:  {elapsed:.1f}s")
    log(f"  cost:       ${cost:.4f} ({len(instances)} x ${total_dph:.2f}/hr)")
    log(f"{'=' * 44}\n")

    if failed > 0:
        log("Logs in ./tests/test_cluster_output/")

    return results


def run_full_pipeline(n=3, extra=2, blocked_machine_ids=None):
    """Provision cluster, run distributed tests, report total cost."""
    t0 = time.time()
    instances = get_ready_cluster(
        n, extra=extra, blocked_machine_ids=blocked_machine_ids
    )
    results = run_distributed_tests(instances)

    elapsed = time.time() - t0
    total_dph = sum(i.get("dph_total", 0) for i in instances)
    cost = total_dph * (elapsed / 3600)
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    log(f"\n{'=' * 44}")
    log("  Full Pipeline — Total")
    log(f"{'=' * 44}")
    log(f"  {passed} passed, {failed} failed")
    log(f"  wall time:  {elapsed:.1f}s")
    log(f"  cost:       ${cost:.4f}")
    log(f"{'=' * 44}\n")

    return results


if __name__ == "__main__":
    run_full_pipeline(n=11)
