from __future__ import annotations

from dataclasses import dataclass

from app.models import Difficulty


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: Difficulty
    title: str
    initial_status: str
    initial_alerts: tuple[str, ...]
    initial_metrics: dict[str, float | int | str]
    inspections: dict[str, str]
    diagnosis_target: str
    diagnosis_aliases: tuple[str, ...]
    mitigation_targets: tuple[str, ...]
    validation_target: str
    communication_target: str
    required_facts: tuple[str, ...]
    max_turns: int
    baseline_plan: tuple[tuple[str, str], ...]


TASKS: dict[str, TaskDefinition] = {
    "tls-certificate-expiry": TaskDefinition(
        task_id="tls-certificate-expiry",
        difficulty="easy",
        title="Checkout traffic failing after TLS certificate expiry",
        initial_status="Checkout is degraded and customers are hitting browser certificate warnings.",
        initial_alerts=(
            "checkout-web 502 rate is above threshold",
            "synthetic HTTPS probe failed certificate validation",
        ),
        initial_metrics={
            "checkout_502_rate": 61.0,
            "tls_handshake_failures": 482,
            "successful_checkouts_per_min": 3,
        },
        inspections={
            "service_status": "The checkout ingress certificate storefront-prod expired at 01:00 UTC and handshake failures started at 01:14 UTC.",
            "runbook_tls": "Mitigation is to renew the certificate, reload the ingress, and validate HTTPS externally.",
            "customer_tickets": "Customers report a browser security warning before payment submission.",
            "metrics_dashboard": "Cart and catalog are healthy. Only checkout-web is failing.",
        },
        diagnosis_target="expired_tls_certificate",
        diagnosis_aliases=("tls_certificate_expiry", "certificate_expired", "expired_certificate"),
        mitigation_targets=("renew_certificate", "reload_ingress"),
        validation_target="validate_https",
        communication_target="customer_update",
        required_facts=("service_status", "runbook_tls"),
        max_turns=8,
        baseline_plan=(
            ("inspect", "service_status"),
            ("inspect", "runbook_tls"),
            ("diagnose", "expired_tls_certificate"),
            ("mitigate", "renew_certificate"),
            ("mitigate", "reload_ingress"),
            ("validate", "validate_https"),
            ("communicate", "customer_update"),
        ),
    ),
    "db-pool-exhaustion": TaskDefinition(
        task_id="db-pool-exhaustion",
        difficulty="medium",
        title="Payment API latency caused by database pool exhaustion",
        initial_status="Payments are slow, request queues are backing up, and the writer database is connection-starved.",
        initial_alerts=(
            "payments p95 latency breached SLO",
            "database connection wait events are elevated",
            "queue depth is rising after the last deployment",
        ),
        initial_metrics={
            "payments_p95_ms": 3900,
            "payments_error_rate": 8.0,
            "db_active_connections": 100,
        },
        inspections={
            "deployment_log": "payments-service v2026.03.31.4 enabled a new ORM batch writer and raised db_pool_max from 40 to 120.",
            "db_dashboard": "Aurora is pinned at max connections and waits are dominated by connection acquisition blocking.",
            "runbook_rollback": "Recommended response is to freeze rollout, rollback the service, and reduce pool pressure before retrying.",
            "incident_notes": "A staging test previously reproduced the same issue when pool ceilings were raised.",
        },
        diagnosis_target="db_pool_exhaustion",
        diagnosis_aliases=("connection_pool_exhaustion", "database_pool_exhaustion", "db_connections_exhausted"),
        mitigation_targets=("rollback_release", "reduce_pool_limit"),
        validation_target="validate_payments_latency",
        communication_target="payments_update",
        required_facts=("deployment_log", "db_dashboard", "runbook_rollback"),
        max_turns=9,
        baseline_plan=(
            ("inspect", "deployment_log"),
            ("inspect", "db_dashboard"),
            ("inspect", "runbook_rollback"),
            ("diagnose", "db_pool_exhaustion"),
            ("mitigate", "rollback_release"),
            ("mitigate", "reduce_pool_limit"),
            ("validate", "validate_payments_latency"),
            ("communicate", "payments_update"),
        ),
    ),
    "idp-outage-vs-security": TaskDefinition(
        task_id="idp-outage-vs-security",
        difficulty="hard",
        title="Differentiate identity vendor outage from true security compromise",
        initial_status="Global SSO is failing and security detections are noisy, but the evidence is ambiguous at first glance.",
        initial_alerts=(
            "enterprise SSO failures are spiking globally",
            "foreign login attempts triggered security alarms",
            "VPN and admin console logins are timing out",
        ),
        initial_metrics={
            "sso_success_rate": 12.0,
            "token_mint_failures": 1840,
            "security_alert_volume": 18,
        },
        inspections={
            "vendor_status": "The identity vendor declared a SEV-1 incident for token issuance errors with global spillover starting at 03:05 UTC.",
            "security_log": "Failed logins increased, but the source IPs match approved travel exceptions and health checks. No successful privilege escalation exists.",
            "runbook_identity": "Route to identity engineering, engage vendor support, suppress outage-driven detections, and share a workaround.",
            "geo_allowlist": "Flagged IPs match approved sales travel and vendor probes from Frankfurt and Singapore.",
            "vpn_dashboard": "VPN auth failures correlate exactly with token minting failures and there is no packet loss anomaly.",
        },
        diagnosis_target="idp_vendor_outage_not_compromise",
        diagnosis_aliases=("identity_vendor_outage", "idp_outage_not_security", "vendor_outage_not_compromise"),
        mitigation_targets=("engage_vendor", "suppress_noisy_detections", "publish_workaround"),
        validation_target="validate_sso_recovery",
        communication_target="sso_update",
        required_facts=("vendor_status", "security_log", "runbook_identity", "geo_allowlist"),
        max_turns=10,
        baseline_plan=(
            ("inspect", "vendor_status"),
            ("inspect", "security_log"),
            ("inspect", "runbook_identity"),
            ("inspect", "geo_allowlist"),
            ("diagnose", "idp_vendor_outage_not_compromise"),
            ("mitigate", "engage_vendor"),
            ("mitigate", "suppress_noisy_detections"),
            ("mitigate", "publish_workaround"),
            ("validate", "validate_sso_recovery"),
            ("communicate", "sso_update"),
        ),
    ),
}

TASK_SEQUENCE = tuple(TASKS)
