#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python으로 Kubernetes와 통신하여 Helm으로 패키지를 설치/업그레이드한 뒤,
해당 릴리스의 Pod 안에 진입(원격 exec)하여 지정된 명령을 실행하는 자동화 스크립트.

필요 사항
- kubectl 컨텍스트가 미리 설정되어 있어야 함 (~/.kube/config)
- Helm CLI가 설치되어 있어야 함 (v3)
- Python 패키지: kubernetes, pyyaml (선택)  설치 필요
    pip install kubernetes pyyaml

예시
1) Helm repo 추가 + 설치 + pod 준비 대기 + pod 내부에서 명령 실행
   python k8s_helm_exec.py \
     --repo-name grafana --repo-url https://grafana.github.io/helm-charts \
     --chart grafana/grafana --release grafana --namespace monitoring \
     --set adminPassword=admin \
     --command "sh -lc 'ls -al /'"

2) values 파일 사용
   python k8s_helm_exec.py \
     --repo-name bitnami --repo-url https://charts.bitnami.com/bitnami \
     --chart bitnami/nginx --release web --namespace demo \
     --values values.yaml \
     --command "sh -lc 'printenv | sort'"
"""
import argparse
import json
import logging
import yaml
import shlex
import subprocess
import sys
import time
from typing import List, Optional, Tuple
from pathlib import Path

from kubernetes import client, config
from kubernetes.stream import stream

from kubernetes import client, config
from kubernetes.stream import stream
from Prome_helper import *
from secret import loki_ip, error_alert_info
import requests
import random

helm_repo_path='ueransim-ue-k6'
# ------------------------------
# 유틸: 쉘 명령 실행
# ------------------------------
def run_cmd(cmd: List[str], check: bool = True, capture_output: bool = True, env: Optional[dict] = None) -> Tuple[int, str, str]:
    """
    서브프로세스로 외부 명령(helm 등)을 실행.
    - check=True: 비정상 종료 시 예외 발생
    - capture_output=True: stdout/stderr 캡처 후 반환
    """
    logging.debug("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, text=True, capture_output=capture_output, env=env)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr


# ------------------------------
# Helm 관련
# ------------------------------

# values.yaml 수정
def set_by_path(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def load_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(data: dict, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_style="'")

def convert_IMSI(IMSI: int) -> str:
    # e.g. 100 -> 0000000100
    return f"{IMSI:010d}"

def modify_ue_values_file(IMSI: int):
    base_values = load_yaml(Path(helm_repo_path) / 'values.yaml')
    set_by_path(base_values, 'initialMSISDN', convert_IMSI(IMSI))
    dump_yaml(base_values, Path(helm_repo_path) / 'values.yaml')

def helm_ue_install(ue_num:int):
    IMSI = 99+ue_num
    modify_ue_values_file(IMSI)
    cmd = ["helm", "install", f"ueransim-ue{ue_num}", helm_repo_path, "-n", "ue"]
    run_cmd(cmd)

def helm_ue_uninstall(ue_num:int):
    cmd = ["helm", "uninstall", f"ueransim-ue{ue_num}", "-n", "ue"]
    run_cmd(cmd)
    #print(f'Deleted {ue_num}-ue.')

def helm_ue_uninstall_multiple(last_ue_num:int, start_num:int=1):
    if last_ue_num < start_num:
        print('last_ue_num must be bigger than start_num!')
        return
    for i in range(start_num,last_ue_num+1):
        helm_ue_uninstall(i)

def helm_repo_add_and_update(repo_name: Optional[str], repo_url: Optional[str]):
    """필요 시 helm repo add & helm repo update 실행"""
    if repo_name and repo_url:
        # 이미 추가돼 있어도 add는 실패하므로, 우선 repo list로 확인
        code, out, _ = run_cmd(["helm", "repo", "list"])
        if repo_name not in out:
            logging.info("Helm repo 추가: %s -> %s", repo_name, repo_url)
            run_cmd(["helm", "repo", "add", repo_name, repo_url])
        logging.info("Helm repo 업데이트")
        run_cmd(["helm", "repo", "update"])

def helm_upgrade_install(
    release: str,
    chart: str,
    namespace: str,
    values_files: List[str],
    set_kvs: List[str],
    version: Optional[str],
    create_namespace: bool = True
):
    """helm upgrade --install 실행"""
    cmd = ["helm", "upgrade", "--install", release, chart, "--namespace", namespace]
    if create_namespace:
        cmd += ["--create-namespace"]
    for vf in values_files:
        cmd += ["-f", vf]
    for kv in set_kvs:
        cmd += ["--set", kv]
    if version:
        cmd += ["--version", version]

    logging.info("Helm 설치/업그레이드 시작: release=%s chart=%s ns=%s", release, chart, namespace)
    run_cmd(cmd)
    logging.info("Helm 설치/업그레이드 완료")


def check_core_function_alive(window: int=5):
    loki = LokiClient(loki_ip)
    recent_logs = loki.get_recent_logs('oai', window=window)
    error_word = ['[error]', '[critical]', '[fatal]']

    with open('tmp/known_error.txt') as f:
        data = f.read()
    known_error_logs = data.split('\n')    
    for single_log in recent_logs:
        for word in error_word:
            if word in single_log['log']:
                is_known_error=False
                if '[error] SMF Selection, no SMF candidate is available' in single_log['log']:
                    # SMF failed. need to restart SMF
                    run_cmd(["kubectl", "rollout", "restart", '-n', 'oai', 'deploy/oai-smf'])
                    run_cmd(["kubectl", "rollout", "restart", '-n', 'oai', 'deploy/oai-nrf'])
                    print('SMF error occured. Rolling SMF.')
                    return 'smf'
                for known_error in known_error_logs:
                    if known_error in single_log['log']:
                        is_known_error=True
                        break
                if not is_known_error:
                    print(f"error occured in {single_log['container']}! :{single_log['log']}")
                    # Send telegram message to me if error occur!
                    requests.post(error_alert_info['url'], json=error_alert_info['body'],headers=error_alert_info['headers'])
                    return single_log['container']
    return False

# ------------------------------
# K8s Pod 대기 & Exec
# ------------------------------
def load_kube_config(context: Optional[str] = None):
    """로컬 kubeconfig를 로드. context가 주어지면 해당 컨텍스트 사용."""
    try:
        if context:
            logging.info("kubeconfig 로드 (context=%s)", context)
            config.load_kube_config(context=context)
        else:
            logging.info("kubeconfig 로드 (기본 컨텍스트)")
            config.load_kube_config()
    except Exception as e:
        logging.exception("kubeconfig 로드 실패")
        raise e


def list_pods_with_selector(namespace: str, selector: str) -> List[client.V1Pod]:
    """라벨 셀렉터로 Pod 목록 조회"""
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=selector)
    return pods.items


def is_pod_ready(pod: client.V1Pod) -> bool:
    """Pod Ready 상태 판단 (모든 컨테이너 준비 완료)"""
    if not pod.status or not pod.status.conditions:
        return False
    conds = {c.type: c.status for c in pod.status.conditions if c.type == "Ready"}
    return conds.get("Ready") == "True"


def wait_for_pods_ready(namespace: str, selector: str, timeout: int = 600, poll_interval: int = 5) -> List[str]:
    """
    특정 라벨 셀렉터의 모든 Pod가 Ready가 될 때까지 대기.
    - timeout: 초
    - poll_interval: 초
    반환: 준비 완료된 Pod 이름 리스트
    """
    logging.info("Pod 준비 대기 시작 (ns=%s, selector=%s, timeout=%ss)", namespace, selector, timeout)
    deadline = time.time() + timeout
    ready_names: List[str] = []

    while time.time() < deadline:
        pods = list_pods_with_selector(namespace, selector)
        if not pods:
            logging.debug("대상 Pod가 아직 생성되지 않음. 잠시 대기...")
            time.sleep(poll_interval)
            continue

        ready_names = [p.metadata.name for p in pods if is_pod_ready(p)]
        not_ready = [p.metadata.name for p in pods if p.metadata.name not in ready_names]

        logging.debug("Ready: %s | NotReady: %s", ready_names, not_ready)
        if len(ready_names) == len(pods):
            logging.info("모든 Pod가 Ready 상태입니다: %s", ready_names)
            return ready_names

        time.sleep(poll_interval)

    raise TimeoutError(f"일부 Pod가 Ready가 아님: ready={ready_names}")


def exec_in_pod(namespace: str, pod: str, command: str, container: Optional[str] = None, tty: bool = False) -> int:
    """
    Pod 내부에서 명령 실행 (kubectl exec 유사).
    - command는 쉘 전체 문자열로 입력 가능 (sh -lc '...')
    - 반환: 0(정상) 또는 1(오류) 정도의 종료 코드. (원격 종료 코드를 직접 알 수 없어 best-effort)
    """
    core = client.CoreV1Api()
    # 문자열을 그대로 전달하려면 sh -lc 사용 권장
    cmd = ["/bin/sh", "-lc", command]

    logging.info("Pod 내부 명령 실행: pod=%s container=%s cmd=%s", pod, container or "<default>", command)
    try:
        resp = stream(
            core.connect_get_namespaced_pod_exec,
            pod,
            namespace,
            command=cmd,
            container=container,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=tty,
        )
        if resp is not None:
            # stream은 표준 출력/에러를 합쳐 문자열로 반환하는 경우가 많음
            sys.stdout.write(resp if isinstance(resp, str) else str(resp))
        return 0
    except client.exceptions.ApiException as e:
        logging.error("API 예외: %s", e)
        return 1
    except Exception as e:
        logging.exception("exec 실패")
        return 1


# ------------------------------
# 메인
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Helm 설치 후 Pod exec 자동화 도구")
    parser.add_argument("--timeout", type=int, default=600, help="Pod Ready 대기 타임아웃(초) 기본 600")
    parser.add_argument("--soft", action="store_true", help = 'UE making softly')
    # 기타
    parser.add_argument("--log-level", default="INFO", help="로그 레벨 (DEBUG/INFO/WARNING/ERROR)")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    # If use this file as main, 
    # It will randomly create or delete UEs
    with open ('tmp/ue_num.json') as f:
        data=json.load(f)
        ue_num = data['num']
    
    # In AMF, IMSI is registered only for 225 UEs.
    # But, in my case, the node can only run only 57 UEs.
    ue_threshold=60 
    while True:
        if random.randrange(1,4)==1:
            # Randomly delete 1/3 of UEs
            delete_num=int(ue_num/3)
            if ue_num-delete_num<4:
                continue
            helm_ue_uninstall_multiple(last_ue_num=ue_num, start_num=ue_num-delete_num)
            ue_num = ue_num-delete_num-1
            with open ('tmp/ue_num.json', 'w') as f:
                data = {'num': ue_num}
                json.dump(data,f)
            #wait for little seconds
            time.sleep(40)
        if args.soft:
            new_ue_num = random.randrange(-2,5)
        else:
            new_ue_num = random.randrange(-8,15)
        if new_ue_num+ue_num<3:
            new_ue_num = 2
        elif new_ue_num+ue_num>ue_threshold:
            new_ue_num = -20
        if new_ue_num>0:
            for i in range(ue_num+1, ue_num+new_ue_num+1):
                helm_ue_install(i)
        elif new_ue_num<0:
            helm_ue_uninstall_multiple(last_ue_num=ue_num, start_num=ue_num+new_ue_num+1)
        ue_num += new_ue_num
        print(f'now we have {ue_num} nums of UEs')
        with open ('tmp/ue_num.json', 'w') as f:
            data = {'num': ue_num}
            json.dump(data,f)
        if args.soft:
            time.sleep(random.randrange(30*6,30*8))
        else:
            time.sleep(random.randrange(5,20))
def run_cmd(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    logging.debug("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
    return p.returncode, p.stdout, p.stderr


def helm_repo_add_update(repo_name: Optional[str], repo_url: Optional[str]):
    if repo_name and repo_url:
        code, out, _ = run_cmd(["helm", "repo", "list"], check=False)
        if repo_name not in out:
            logging.info("helm repo add %s %s", repo_name, repo_url)
            run_cmd(["helm", "repo", "add", repo_name, repo_url])
        logging.info("helm repo update")
        run_cmd(["helm", "repo", "update"])


def helm_upgrade_install(release: str, chart: str, namespace: str,
                         values: List[str], set_kvs: List[str], version: Optional[str]):
    cmd = ["helm", "upgrade", "--install", release, chart,
           "--namespace", namespace, "--create-namespace"]
    for v in values:
        cmd += ["-f", v]
    for kv in set_kvs:
        cmd += ["--set", kv]
    if version:
        cmd += ["--version", version]
    logging.info("helm upgrade --install: %s", " ".join(cmd))
    run_cmd(cmd)


def load_kube(context: Optional[str] = None):
    if context:
        config.load_kube_config(context=context)
    else:
        config.load_kube_config()


def list_pods(ns: str, selector: str):
    v1 = client.CoreV1Api()
    return v1.list_namespaced_pod(namespace=ns, label_selector=selector).items


def is_ready(pod) -> bool:
    conds = {c.type: c.status for c in (pod.status.conditions or [])}
    return conds.get("Ready") == "True"


def wait_ready(ns: str, selector: str, timeout: int = 600, interval: int = 5) -> List[str]:
    logging.info("Waiting pods Ready: ns=%s selector=%s", ns, selector)
    deadline = time.time() + timeout
    ready = []
    while time.time() < deadline:
        pods = list_pods(ns, selector)
        if not pods:
            time.sleep(interval)
            continue
        ready = [p.metadata.name for p in pods if is_ready(p)]
        if len(ready) == len(pods):
            logging.info("All pods Ready: %s", ready)
            return ready
        time.sleep(interval)
    raise TimeoutError(f"Pods not Ready within {timeout}s; ready={ready}")


def build_load_command(tool: str, url: str, duration: str, concurrency: int,
                       qps: int, requests: int) -> str:
    """
    tool:
      - hey: hey -z <duration> -c <concurrency> [-q <qps>] <url>
      - wrk: wrk -t <threads=concurrency//4 or 1> -c <concurrency> -d <duration> <url>
      - ab : ab -n <requests> -c <concurrency> <url>  (duration 미지원)
      - curl: duration 동안 concurrency 병렬로 반복호출
    """
    if tool == "hey":
        parts = ["hey", "-z", duration, "-c", str(concurrency)]
        if qps > 0:
            parts += ["-q", str(qps)]
        parts += [shlex.quote(url)]
        return " ".join(parts)
    if tool == "wrk":
        threads = max(1, concurrency // 4)
        return f"wrk -t{threads} -c{concurrency} -d{duration} {shlex.quote(url)}"
    if tool == "ab":
        n = requests if requests > 0 else concurrency * 1000
        return f"ab -n {n} -c {concurrency} {shlex.quote(url)}"
    # curl fallback
    # duration(초) 동안 concurrency 만큼 병렬로 루프, 각 요청 10초 타임아웃
    return (
        'DUR_S=$(python - <<PY\n'
        'import re,sys\n'
        'd=sys.argv[1]\n'
        'm=re.match(r"^(\\d+)([smh]?)$", d)\n'
        'n=int(m.group(1)); u=m.group(2)\n'
        'print(n if u in ("","s") else n*60 if u=="m" else n*3600)\n'
        'PY\n' + shlex.quote(duration) + '); '
        f'CONC={concurrency}; URL={shlex.quote(url)}; '
        'END=$(( $(date +%s) + DUR_S )); '
        'while [ $(date +%s) -lt $END ]; do '
        '  i=0; while [ $i -lt $CONC ]; do '
        '    (curl -fsS --max-time 10 "$URL" >/dev/null 2>&1 || true) & i=$((i+1)); '
        '  done; wait; '
        'done'
    )


def exec_in_pod(ns: str, pod: str, cmd: str, container: Optional[str] = None):
    core = client.CoreV1Api()
    full = ["/bin/sh", "-lc", cmd]
    logging.info("Exec in pod=%s container=%s cmd=%s", pod, container or "<default>", cmd)
    resp = stream(core.connect_get_namespaced_pod_exec,
                  name=pod, namespace=ns,
                  command=full, container=container,
                  stderr=True, stdin=False, stdout=True, tty=False)
    if resp:
        print(resp)



if __name__ == "__main__":
    main()
