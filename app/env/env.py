from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle

from app.graders.graders import MAX_SCORE, grade_task
from app.models import Action, Observation, Reward, StateSnapshot
from app.tasks.tasks import TASKS, TASK_SEQUENCE, TaskDefinition


@dataclass
class EpisodeMemory:
    task: TaskDefinition
    turn: int = 0
    done: bool = False
    reward_total: float = 0.0
    diagnosis: str | None = None
    known_facts: list[str] = field(default_factory=list)
    facts_detail: dict[str, str] = field(default_factory=dict)
    fixes_applied: list[str] = field(default_factory=list)
    validations_passed: list[str] = field(default_factory=list)
    communications_sent: list[str] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)
    repeated_useless_actions: int = 0
    wrong_actions: int = 0
    resolution_confirmed: bool = False


class IncidentCommanderEnvironment:
    def __init__(self) -> None:
        self._task_cycle = cycle(TASK_SEQUENCE)
        self._episode: EpisodeMemory | None = None

    def reset(self, task_id: str | None = None) -> Observation:
        selected_id = task_id or next(self._task_cycle)
        if selected_id not in TASKS:
            raise ValueError(f"Unknown task_id '{selected_id}'.")
        self._episode = EpisodeMemory(task=TASKS[selected_id])
        return self._observation()

    def state(self) -> dict:
        episode = self._require_episode()
        state = StateSnapshot(
            task_id=episode.task.task_id,
            difficulty=episode.task.difficulty,
            turn=episode.turn,
            max_turns=episode.task.max_turns,
            done=episode.done,
            score=grade_task(self._raw_state()),
            reward_total=round(episode.reward_total, 4),
            diagnosis=episode.diagnosis,
            fixes_applied=list(episode.fixes_applied),
            validations_passed=list(episode.validations_passed),
            communications_sent=list(episode.communications_sent),
            known_facts=list(episode.known_facts),
            action_history=list(episode.action_history),
            repeated_useless_actions=episode.repeated_useless_actions,
            wrong_actions=episode.wrong_actions,
        )
        return state.model_dump(mode="json")

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        episode = self._require_episode()
        if episode.done:
            raise ValueError("Episode already completed. Call reset() before stepping again.")

        normalized_target = self._canonicalize_target(action.action_type, action.target)
        action = Action(
            action_type=action.action_type,
            target=normalized_target,
            note=action.note,
        )

        episode.turn += 1
        reward = self._dispatch(action)
        episode.action_history.append(f"{action.action_type}:{action.target}")
        episode.reward_total = round(episode.reward_total + reward.score, 4)

        current_score = grade_task(self._raw_state())
        if self._is_resolved():
            episode.done = True
            episode.resolution_confirmed = True
            current_score = MAX_SCORE
        elif episode.turn >= episode.task.max_turns:
            episode.done = True

        info = {
            "task_title": episode.task.title,
            "turn": episode.turn,
            "max_turns": episode.task.max_turns,
            "score": current_score,
            "reward_total": episode.reward_total,
        }
        return self._observation(), reward, episode.done, info

    def _dispatch(self, action: Action) -> Reward:
        if action.action_type == "inspect":
            return self._handle_inspect(action.target)

        handlers = {
            "diagnose": self._handle_diagnose,
            "mitigate": self._handle_mitigate,
            "validate": self._handle_validate,
            "communicate": self._handle_communicate,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            raise ValueError(f"Unsupported action_type '{action.action_type}'.")
        return handler(action.target)

    def _canonicalize_target(self, action_type: str, target: str) -> str:
        episode = self._require_episode()
        normalized = target.lower().strip()

        if action_type == "inspect" and normalized in episode.task.inspections:
            return normalized

        mapping = {
            "tls": "certificate",
            "ssl": "certificate",
            "cert": "certificate",
            "certificate": "certificate",
        }

        if action_type == "inspect":
            task_mapping = {
                "tls-certificate-expiry": {
                    "tls": "service_status",
                    "ssl": "service_status",
                    "cert": "service_status",
                    "certificate": "service_status",
                    "runbook": "runbook_tls",
                    "tickets": "customer_tickets",
                    "customers": "customer_tickets",
                    "metrics": "metrics_dashboard",
                    "dashboard": "metrics_dashboard",
                },
                "db-pool-exhaustion": {
                    "db": "db_dashboard",
                    "database": "db_dashboard",
                    "deploy": "deployment_log",
                    "deployment": "deployment_log",
                    "release": "deployment_log",
                    "runbook": "runbook_rollback",
                    "rollback": "runbook_rollback",
                    "notes": "incident_notes",
                },
                "idp-outage-vs-security": {
                    "idp": "vendor_status",
                    "vendor": "vendor_status",
                    "status": "vendor_status",
                    "security": "security_log",
                    "log": "security_log",
                    "geo": "geo_allowlist",
                    "allowlist": "geo_allowlist",
                    "vpn": "vpn_dashboard",
                    "runbook": "runbook_identity",
                    "identity": "runbook_identity",
                },
            }
            canonical_target = mapping.get(normalized, normalized)
            return task_mapping.get(episode.task.task_id, {}).get(canonical_target, canonical_target)

        action_mappings = {
            "diagnose": {
                "tls-certificate-expiry": {
                    "tls": "expired_tls_certificate",
                    "ssl": "expired_tls_certificate",
                    "cert": "expired_tls_certificate",
                    "certificate": "expired_tls_certificate",
                },
                "db-pool-exhaustion": {
                    "db": "db_pool_exhaustion",
                    "database": "db_pool_exhaustion",
                    "pool": "db_pool_exhaustion",
                },
                "idp-outage-vs-security": {
                    "idp": "idp_vendor_outage_not_compromise",
                    "vendor": "idp_vendor_outage_not_compromise",
                    "sso": "idp_vendor_outage_not_compromise",
                    "security": "idp_vendor_outage_not_compromise",
                },
            },
            "mitigate": {
                "tls-certificate-expiry": {
                    "certificate": "renew_certificate",
                    "renew": "renew_certificate",
                    "reload": "reload_ingress",
                    "ingress": "reload_ingress",
                },
                "db-pool-exhaustion": {
                    "rollback": "rollback_release",
                    "release": "rollback_release",
                    "reduce": "reduce_pool_limit",
                    "pool": "reduce_pool_limit",
                    "database": "reduce_pool_limit",
                },
                "idp-outage-vs-security": {
                    "vendor": "engage_vendor",
                    "engage": "engage_vendor",
                    "suppress": "suppress_noisy_detections",
                    "detections": "suppress_noisy_detections",
                    "workaround": "publish_workaround",
                },
            },
            "validate": {
                "tls-certificate-expiry": {
                    "https": "validate_https",
                    "certificate": "validate_https",
                },
                "db-pool-exhaustion": {
                    "payments": "validate_payments_latency",
                    "latency": "validate_payments_latency",
                    "db": "validate_payments_latency",
                },
                "idp-outage-vs-security": {
                    "sso": "validate_sso_recovery",
                    "idp": "validate_sso_recovery",
                    "recovery": "validate_sso_recovery",
                },
            },
            "communicate": {
                "tls-certificate-expiry": {
                    "update": "customer_update",
                    "customer": "customer_update",
                    "customers": "customer_update",
                },
                "db-pool-exhaustion": {
                    "update": "payments_update",
                    "payments": "payments_update",
                    "customer": "payments_update",
                },
                "idp-outage-vs-security": {
                    "update": "sso_update",
                    "sso": "sso_update",
                    "customer": "sso_update",
                },
            },
        }

        return action_mappings.get(action_type, {}).get(episode.task.task_id, {}).get(mapping.get(normalized, normalized), mapping.get(normalized, normalized))

    def _normalize_token(self, value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def _handle_inspect(self, target: str) -> Reward:
        episode = self._require_episode()
        detail = episode.task.inspections.get(target)
        if detail is None:
            return self._penalty(f"Unknown inspection target '{target}'.")
        if target in episode.known_facts:
            episode.repeated_useless_actions += 1
            return Reward(score=-0.2, reason=f"Inspection '{target}' already completed.")
        episode.known_facts.append(target)
        episode.facts_detail[target] = detail
        return Reward(score=0.1, reason=detail)

    def _handle_diagnose(self, target: str) -> Reward:
        episode = self._require_episode()
        valid_targets = (episode.task.diagnosis_target, *episode.task.diagnosis_aliases)
        if target not in valid_targets:
            return self._penalty(f"Diagnosis '{target}' does not fit the observed system behavior.")
        if episode.diagnosis == episode.task.diagnosis_target:
            episode.repeated_useless_actions += 1
            return Reward(score=-0.2, reason="Diagnosis already established.")
        episode.diagnosis = episode.task.diagnosis_target
        if self._facts_ready():
            return Reward(score=0.3, reason="Correct diagnosis established from the available evidence.")
        return Reward(score=0.1, reason="Diagnosis is plausible, but it is not yet backed by enough inspected evidence.")

    def _handle_mitigate(self, target: str) -> Reward:
        episode = self._require_episode()
        if target not in episode.task.mitigation_targets:
            return self._penalty(f"Mitigation '{target}' is not appropriate for this incident.")
        if target in episode.fixes_applied:
            episode.repeated_useless_actions += 1
            return Reward(score=-0.2, reason=f"Mitigation '{target}' already applied.")
        if episode.diagnosis != episode.task.diagnosis_target:
            return self._penalty("Mitigation attempted before the incident was diagnosed correctly.")
        episode.fixes_applied.append(target)
        if self._all_fixes_applied():
            return Reward(score=0.5, reason="Correct remediation plan fully applied.")
        return Reward(score=0.2, reason=f"Mitigation '{target}' applied and system pressure is reducing.")

    def _handle_validate(self, target: str) -> Reward:
        episode = self._require_episode()
        if target != episode.task.validation_target:
            return self._penalty(f"Validation '{target}' is not defined for this task.")
        if target in episode.validations_passed:
            episode.repeated_useless_actions += 1
            return Reward(score=-0.2, reason="Validation already performed.")
        if episode.task.task_id == "tls-certificate-expiry":
            diagnosis_completed = episode.diagnosis == episode.task.diagnosis_target
            mitigation_completed = (
                "renew_certificate" in episode.fixes_applied
                or "mitigate:renew_certificate" in episode.action_history
            )
            if not diagnosis_completed:
                return self._penalty("Validation attempted before the TLS issue was diagnosed.")
            if not mitigation_completed:
                return self._penalty("Validation attempted before the TLS certificate was renewed.")
            episode.validations_passed.append(target)
            return Reward(
                score=MAX_SCORE,
                reason="Service restored successfully: TLS certificate renewed and service recovering.",
            )
        if not self._all_fixes_applied():
            return self._penalty("Validation attempted before all required mitigations were applied.")
        episode.validations_passed.append(target)
        if episode.task.task_id == "db-pool-exhaustion":
            return Reward(score=0.1, reason="Payments latency and database waits are recovering after rollback and pool reduction.")
        return Reward(score=0.1, reason="SSO token issuance is improving and outage-specific detections are quieting down.")

    def _handle_communicate(self, target: str) -> Reward:
        episode = self._require_episode()
        if target != episode.task.communication_target:
            return self._penalty(f"Communication action '{target}' is not recognized.")
        if target in episode.communications_sent:
            episode.repeated_useless_actions += 1
            return Reward(score=-0.2, reason="The customer update was already sent.")
        if episode.diagnosis != episode.task.diagnosis_target:
            return self._penalty("Communication attempted before a correct diagnosis was available.")
        episode.communications_sent.append(target)
        if self._is_resolved():
            return Reward(score=MAX_SCORE, reason="Full resolution achieved and communicated clearly.")
        return Reward(score=0.1, reason="Customer communication sent with the current incident status.")

    def _facts_ready(self) -> bool:
        episode = self._require_episode()
        return all(fact in episode.known_facts for fact in episode.task.required_facts)

    def _all_fixes_applied(self) -> bool:
        episode = self._require_episode()
        return all(fix in episode.fixes_applied for fix in episode.task.mitigation_targets)

    def _is_resolved(self) -> bool:
        episode = self._require_episode()
        return (
            episode.diagnosis == episode.task.diagnosis_target
            and self._all_fixes_applied()
            and episode.task.validation_target in episode.validations_passed
            and episode.task.communication_target in episode.communications_sent
        )

    def _penalty(self, reason: str) -> Reward:
        episode = self._require_episode()
        episode.wrong_actions += 1
        return Reward(score=-0.2, reason=reason)

    def _observation(self) -> Observation:
        episode = self._require_episode()
        status = episode.task.initial_status
        metrics = dict(episode.task.initial_metrics)

        if episode.task.task_id == "tls-certificate-expiry" and "validate_https" in episode.validations_passed:
            status = "Checkout HTTPS is restored and synthetic probes are passing."
            metrics["checkout_502_rate"] = 1.2
            metrics["tls_handshake_failures"] = 0
            metrics["successful_checkouts_per_min"] = 48
        elif episode.task.task_id == "db-pool-exhaustion" and "validate_payments_latency" in episode.validations_passed:
            status = "Payments recovered after rollback and database connection pressure is back within limits."
            metrics["payments_p95_ms"] = 280
            metrics["payments_error_rate"] = 0.5
            metrics["db_active_connections"] = 41
        elif episode.task.task_id == "idp-outage-vs-security" and "validate_sso_recovery" in episode.validations_passed:
            status = "SSO is stabilizing and the incident is confirmed as an upstream vendor outage, not compromise."
            metrics["sso_success_rate"] = 96.0
            metrics["token_mint_failures"] = 12
            metrics["security_alert_volume"] = 2

        alerts = list(episode.task.initial_alerts)
        if episode.diagnosis == episode.task.diagnosis_target:
            alerts.append(f"Diagnosis confirmed: {episode.task.diagnosis_target}")
        if episode.fixes_applied:
            alerts.append(f"Mitigations applied: {', '.join(episode.fixes_applied)}")
        if episode.validations_passed:
            alerts.append(f"Validations passed: {', '.join(episode.validations_passed)}")
        if episode.communications_sent:
            alerts.append("Stakeholder communication has been sent.")

        completed_actions = list(episode.action_history)
        for fact in episode.known_facts:
            completed_actions.append(f"fact:{fact}")

        return Observation(
            task_id=episode.task.task_id,
            difficulty=episode.task.difficulty,
            system_status=status,
            alerts=alerts,
            metrics=metrics,
            available_actions=["inspect", "diagnose", "mitigate", "validate", "communicate"],
            known_facts=[episode.facts_detail[key] for key in episode.known_facts],
            completed_actions=completed_actions,
            turns_remaining=max(episode.task.max_turns - episode.turn, 0),
        )

    def _raw_state(self) -> dict:
        episode = self._require_episode()
        return {
            "task_id": episode.task.task_id,
            "known_facts": list(episode.known_facts),
            "diagnosis": episode.diagnosis,
            "fixes_applied": list(episode.fixes_applied),
            "validations_passed": list(episode.validations_passed),
            "communications_sent": list(episode.communications_sent),
            "wrong_actions": episode.wrong_actions,
            "repeated_useless_actions": episode.repeated_useless_actions,
        }

    def _require_episode(self) -> EpisodeMemory:
        if self._episode is None:
            raise ValueError("No active episode. Call reset() first.")
        return self._episode
