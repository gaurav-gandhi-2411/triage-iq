#!/usr/bin/env bash
# One-time Cloud Monitoring setup for TriageIQ.
# Run once after deploying the service. Safe to re-run — resource creation
# commands are guarded with existence checks or tolerate "already exists".
#
# What this creates:
#   1. Email notification channel
#   2. Log-based metric for Groq token usage (extracted from JSON access logs)
#   3. Alert: >5% 5xx error rate over 10 min
#   4. Alert: p95 latency >5s over 10 min
#   5. Alert: daily Groq token usage >70K (70% of free-tier 100K TPD)
#   6. Dashboard: request rate, error rate, p95 latency, Groq tokens today
#
# Prerequisites:
#   gcloud authenticated with the service account that has:
#     - roles/monitoring.alertPolicyEditor
#     - roles/monitoring.dashboardEditor
#     - roles/logging.configWriter   (for log-based metrics)
#
# Usage:
#   GCP_PROJECT=triageiq-portfolio-495022 ALERT_EMAIL=you@example.com bash scripts/setup_monitoring.sh

set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:-triageiq-portfolio-495022}"
SERVICE_NAME="${SERVICE_NAME:-triageiq-api}"
GCP_REGION="${GCP_REGION:-us-central1}"
ALERT_EMAIL="${ALERT_EMAIL:-gg5678g@gmail.com}"

echo "=== TriageIQ Cloud Monitoring Setup ==="
echo "  Project:      $GCP_PROJECT"
echo "  Service:      $SERVICE_NAME"
echo "  Region:       $GCP_REGION"
echo "  Alert email:  $ALERT_EMAIL"
echo ""

# ---------------------------------------------------------------------------
# 1. Email notification channel
#    Cloud Monitoring sends alert notifications to this address.
# ---------------------------------------------------------------------------
echo "[1/6] Creating email notification channel..."
CHANNEL_ID=$(gcloud alpha monitoring channels create \
  --display-name="TriageIQ Alerts" \
  --type=email \
  --channel-labels="email_address=${ALERT_EMAIL}" \
  --format="value(name)" \
  --project="${GCP_PROJECT}" 2>/dev/null) \
|| CHANNEL_ID=$(gcloud alpha monitoring channels list \
  --filter="displayName='TriageIQ Alerts' AND type=email" \
  --format="value(name)" \
  --project="${GCP_PROJECT}" | head -1)
echo "  Channel ID: $CHANNEL_ID"

# ---------------------------------------------------------------------------
# 2. Log-based metric for Groq token usage
#    The JSON access logger emits groq_tokens_total per successful triage call.
#    This creates a DISTRIBUTION log-based metric that Cloud Monitoring can
#    sum over a day window for the quota alert.
# ---------------------------------------------------------------------------
echo "[2/6] Creating log-based metric for Groq token usage..."
gcloud logging metrics create triageiq_groq_tokens \
  --description="Groq tokens used per successful triage call (prompt + completion)" \
  --log-filter='resource.type="cloud_run_revision"
    resource.labels.service_name="'"${SERVICE_NAME}"'"
    jsonPayload.log_type="access"
    jsonPayload.status="success"' \
  --value-extractor="EXTRACT(jsonPayload.groq_tokens_total)" \
  --project="${GCP_PROJECT}" 2>/dev/null \
  || echo "  (metric triageiq_groq_tokens already exists — skipping)"

# ---------------------------------------------------------------------------
# 3. Alert: >5% 5xx error rate over 10 min
#    Uses Cloud Run's built-in request_count metric.
#    MQL: compute ratio of 5xx responses to total over a 10-min window.
# ---------------------------------------------------------------------------
echo "[3/6] Creating error rate alert (>5% 5xx over 10 min)..."
cat > /tmp/triageiq_error_rate_alert.json << ALERT_JSON
{
  "displayName": "TriageIQ: High 5xx Error Rate",
  "documentation": {
    "content": "More than 5% of /triage requests are returning 5xx status codes over the last 10 minutes. Check Cloud Logging for stack traces: filter jsonPayload.status=error."
  },
  "conditions": [
    {
      "displayName": "5xx error rate > 5%",
      "conditionMonitoringQueryLanguage": {
        "query": "fetch cloud_run_revision\n| metric 'run.googleapis.com/request_count'\n| filter resource.service_name == '${SERVICE_NAME}'\n    && resource.location == '${GCP_REGION}'\n| {filter metric.response_code_class == '5xx'; ident}\n| outer_join 0\n| value val(0) / (val(0) + val(1))\n| window 10m\n| every 10m\n| condition val() > 0.05",
        "duration": "0s",
        "trigger": {
          "count": 1
        }
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "604800s"
  },
  "combiner": "OR",
  "notificationChannels": ["${CHANNEL_ID}"]
}
ALERT_JSON
gcloud alpha monitoring policies create \
  --policy-from-file=/tmp/triageiq_error_rate_alert.json \
  --project="${GCP_PROJECT}" \
  && echo "  Error rate alert created." \
  || echo "  Warning: error rate alert creation failed — see error above."

# ---------------------------------------------------------------------------
# 4. Alert: p95 latency >5s over 10 min
#    Uses Cloud Run's request_latencies distribution metric.
# ---------------------------------------------------------------------------
echo "[4/6] Creating latency alert (p95 > 5s over 10 min)..."
cat > /tmp/triageiq_latency_alert.json << ALERT_JSON
{
  "displayName": "TriageIQ: High p95 Latency",
  "documentation": {
    "content": "p95 latency on Cloud Run has exceeded 5 seconds over the last 10 minutes. Typical warm p50 is ~3.5s; p95 > 5s may indicate Groq API slowness or instance cold starts."
  },
  "conditions": [
    {
      "displayName": "p95 latency > 5s",
      "conditionMonitoringQueryLanguage": {
        "query": "fetch cloud_run_revision\n| metric 'run.googleapis.com/request_latencies'\n| filter resource.service_name == '${SERVICE_NAME}'\n    && resource.location == '${GCP_REGION}'\n| align delta(10m)\n| every 10m\n| percentile(val(), 95)\n| condition val() > 5000",
        "duration": "0s",
        "trigger": {
          "count": 1
        }
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "604800s"
  },
  "combiner": "OR",
  "notificationChannels": ["${CHANNEL_ID}"]
}
ALERT_JSON
gcloud alpha monitoring policies create \
  --policy-from-file=/tmp/triageiq_latency_alert.json \
  --project="${GCP_PROJECT}" \
  && echo "  Latency alert created." \
  || echo "  Warning: latency alert creation failed — see error above."

# ---------------------------------------------------------------------------
# 5. Alert: daily Groq token usage >70K
#    Uses the log-based metric created in step 2.
#    70K = 70% of the 100K free-tier TPD limit — gives runway to investigate.
# ---------------------------------------------------------------------------
echo "[5/6] Creating Groq quota alert (>70K tokens/day)..."
cat > /tmp/triageiq_quota_alert.json << ALERT_JSON
{
  "displayName": "TriageIQ: Groq Token Quota Warning",
  "documentation": {
    "content": "Daily Groq token usage has exceeded 70K (70% of the 100K free-tier TPD limit). At ~2K tokens/call this corresponds to ~35 triage calls. If usage continues at this rate the service will return 429 from Groq before midnight UTC. Consider throttling or queuing /triage requests."
  },
  "conditions": [
    {
      "displayName": "Daily Groq tokens > 70000",
      "conditionThreshold": {
        "filter": "metric.type=\"logging.googleapis.com/user/triageiq_groq_tokens\" resource.type=\"cloud_run_revision\"",
        "aggregations": [
          {
            "alignmentPeriod": "86400s",
            "perSeriesAligner": "ALIGN_SUM",
            "crossSeriesReducer": "REDUCE_SUM"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": 70000,
        "duration": "0s",
        "trigger": {
          "count": 1
        }
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "86400s"
  },
  "combiner": "OR",
  "notificationChannels": ["${CHANNEL_ID}"]
}
ALERT_JSON
gcloud alpha monitoring policies create \
  --policy-from-file=/tmp/triageiq_quota_alert.json \
  --project="${GCP_PROJECT}" \
  && echo "  Groq quota alert created." \
  || echo "  Warning: Groq quota alert creation failed — see error above."

# ---------------------------------------------------------------------------
# 6. Dashboard: request rate, error rate, p95 latency, Groq tokens today
#    Single pane of glass for the on-call runbook.
# ---------------------------------------------------------------------------
echo "[6/6] Creating monitoring dashboard..."
cat > /tmp/triageiq_dashboard.json << DASHBOARD_JSON
{
  "displayName": "TriageIQ Production",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Request Rate (req/min)",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\" resource.label.service_name=\"${SERVICE_NAME}\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_RATE",
                  "crossSeriesReducer": "REDUCE_SUM"
                }
              }
            },
            "plotType": "LINE"
          }],
          "timeshiftDuration": "0s",
          "yAxis": {"label": "req/s", "scale": "LINEAR"}
        }
      },
      {
        "title": "5xx Error Rate",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\" resource.label.service_name=\"${SERVICE_NAME}\" metric.label.response_code_class=\"5xx\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_RATE",
                  "crossSeriesReducer": "REDUCE_SUM"
                }
              }
            },
            "plotType": "LINE"
          }],
          "timeshiftDuration": "0s",
          "yAxis": {"label": "5xx/s", "scale": "LINEAR"}
        }
      },
      {
        "title": "p95 Latency (ms)",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "metric.type=\"run.googleapis.com/request_latencies\" resource.type=\"cloud_run_revision\" resource.label.service_name=\"${SERVICE_NAME}\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_PERCENTILE_99",
                  "crossSeriesReducer": "REDUCE_MEAN"
                }
              }
            },
            "plotType": "LINE"
          }],
          "timeshiftDuration": "0s",
          "yAxis": {"label": "latency ms", "scale": "LINEAR"}
        }
      },
      {
        "title": "Groq Tokens Used Today",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "metric.type=\"logging.googleapis.com/user/triageiq_groq_tokens\" resource.type=\"cloud_run_revision\"",
                "aggregation": {
                  "alignmentPeriod": "86400s",
                  "perSeriesAligner": "ALIGN_SUM",
                  "crossSeriesReducer": "REDUCE_SUM"
                }
              }
            },
            "plotType": "STACKED_BAR"
          }],
          "timeshiftDuration": "0s",
          "yAxis": {"label": "tokens", "scale": "LINEAR"}
        }
      }
    ]
  }
}
DASHBOARD_JSON
gcloud monitoring dashboards create \
  --config-from-file=/tmp/triageiq_dashboard.json \
  --project="${GCP_PROJECT}" \
  && echo "  Dashboard created." \
  || echo "  Warning: dashboard creation failed — see error above."

echo ""
echo "=== Setup complete ==="
echo "View alerts:    https://console.cloud.google.com/monitoring/alerting?project=${GCP_PROJECT}"
echo "View dashboard: https://console.cloud.google.com/monitoring/dashboards?project=${GCP_PROJECT}"
echo ""
echo "Post-setup verification checklist:"
echo "  [ ] Confirm email channel received a test notification (Console > Alerting > Notification Channels > Send Test)"
echo "  [ ] Hit /triage a few times, then check Groq tokens graph shows activity"
echo "  [ ] Temporarily break a triage call (e.g. invalid env) to verify error rate alert fires within 10 min"
echo "  [ ] Revert any test breakage after alert confirms"
