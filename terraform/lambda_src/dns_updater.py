"""Event-driven DNS updater.

There's no ALB (see terraform/ecs.tf), so the Fargate task's public IP
changes on every redeploy and on every Spot interruption. This Lambda fires
on "ECS Task State Change" events for the API service and upserts a low-TTL
Route53 A record pointing at the service's current task. CloudFront
(terraform/dns.tf) uses that record as its origin, so the public domain
stays live without anyone re-running the old "find the public IP" README
steps by hand.

Deliberately resolves the target IP by re-querying the service for its
current running task rather than trusting the triggering event's taskArn:
during a rolling deploy the old task (still up until the new one is healthy)
and the new task can each emit their own RUNNING event, and if the old
task's event happens to be *processed* second, trusting it would point DNS
at the task about to be stopped. Always picking the most-recently-started
running task self-corrects regardless of event arrival order.
"""
import os
import time

import boto3

ROUTE53_ZONE_ID = os.environ["ROUTE53_ZONE_ID"]
ORIGIN_RECORD_NAME = os.environ["ORIGIN_RECORD_NAME"]
ECS_CLUSTER = os.environ["ECS_CLUSTER"]
ECS_SERVICE = os.environ["ECS_SERVICE"]
RECORD_TTL = int(os.environ.get("RECORD_TTL", "30"))


def _resolve_current_public_ip() -> str | None:
    ecs = boto3.client("ecs")
    task_arns = ecs.list_tasks(
        cluster=ECS_CLUSTER, serviceName=ECS_SERVICE, desiredStatus="RUNNING"
    ).get("taskArns", [])
    if not task_arns:
        return None

    tasks = ecs.describe_tasks(cluster=ECS_CLUSTER, tasks=task_arns)["tasks"]
    running = [t for t in tasks if t.get("lastStatus") == "RUNNING" and t.get("startedAt")]
    if not running:
        return None
    newest = max(running, key=lambda t: t["startedAt"])

    try:
        eni_id = next(
            d["value"] for d in newest["attachments"][0]["details"] if d["name"] == "networkInterfaceId"
        )
    except (IndexError, KeyError, StopIteration):
        return None
    eni = boto3.client("ec2").describe_network_interfaces(NetworkInterfaceIds=[eni_id])
    return eni["NetworkInterfaces"][0].get("Association", {}).get("PublicIp")


def handler(event, context) -> dict:
    detail = event.get("detail", {})
    if detail.get("lastStatus") != "RUNNING":
        return {"skipped": "not_running"}

    # The ENI's public IP association -- or, right after a deploy starts, the
    # new task itself -- can lag a few seconds behind this event, so retry
    # within this invocation rather than dropping it.
    public_ip = None
    for _ in range(5):
        public_ip = _resolve_current_public_ip()
        if public_ip:
            break
        time.sleep(3)

    if not public_ip:
        print("no running task with a public IP found after retries -- giving up for this event")
        return {"skipped": "no_public_ip"}

    boto3.client("route53").change_resource_record_sets(
        HostedZoneId=ROUTE53_ZONE_ID,
        ChangeBatch={
            "Comment": "auto-updated by pathml-dns-updater",
            "Changes": [
                {
                    "Action": "UPSERT",
                    "ResourceRecordSet": {
                        "Name": ORIGIN_RECORD_NAME,
                        "Type": "A",
                        "TTL": RECORD_TTL,
                        "ResourceRecords": [{"Value": public_ip}],
                    },
                }
            ],
        },
    )
    print(f"updated {ORIGIN_RECORD_NAME} -> {public_ip}")
    return {"updated_ip": public_ip}
