import json
import os
from threading import Thread

from flask import Flask, jsonify, redirect, render_template, request, url_for

from action_policy import stake_for_action
from collection_repository import (
    DatabaseWriteUnavailableError,
    compute_issue_top_picks,
    expire_pending_manual_reviews,
    get_database_status,
    get_issue_top_picks,
    has_issue_top_picks,
)
from collector_store import (
    add_selectable_matches,
    build_sections,
    collect_all_matches,
    collect_match,
    get_canonical_prediction_run,
    get_collection_failure_reason,
    get_collection_stats,
    get_feedback_log,
    get_feedback_summary,
    get_match_analysis,
    init_db,
    list_selectable_matches,
    list_issues,
    list_matches_by_issue,
    list_pending_manual_review_runs,
    list_prediction_runs,
    predict_issue,
    predict_match,
    record_feedback,
    remove_selectable_match,
    resolve_manual_review,
    settle_match_result,
    settle_issue_results,
    sync_issue_matches,
    sync_matches,
)
from collection_repository import list_backtest_rows as list_backtest_rows
from prediction_engine import summarize_backtest as summarize_backtest
from config_service import (
    LEGACY_MODEL_KEYS,
    load_env_config,
    reload_env_to_os,
    save_env_config,
    test_collection_api,
    test_openai_api,
)
from learning_engine import (
    activate_learning_profile,
    deactivate_learning_profile,
    get_learning_overview,
    train_learning_profile,
)
from historical_import import import_historical_learning_feedback
from replay_backfill import replay_backfill_learning_feedback
from progress_service import complete_task, create_task, fail_task, get_task, update_task
from source_500_client import LIVE_SELECTABLE_URL


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
APP_VERSION = "source-handicap-top3-20260618"


@app.after_request
def _disable_browser_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


_DB_INITIALIZED = False
CONFIG_FORM_KEYS = {
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL_RESEARCH",
    "COLLECTION_BASE_URL",
    "COLLECTION_APIKEY",
    "COLLECTION_MODEL",
    "COLLECTION_REVIEW_MAX_TOKENS",
    "LLM_REVIEW_ENABLED",
    "TAVILY_API_KEY",
    "FOOTBALL_API_KEY",
    "NETWORK_SEARCH_URL",
    "MODEL_OPTIONS",
    "AGENT_MODEL_MAP",
}
# Checkbox-style fields submit nothing when unchecked. We need to know which
# CONFIG_FORM_KEYS are checkboxes so /config/save can write the explicit
# "false" instead of falling back to the previously stored value.
CONFIG_BOOLEAN_KEYS = {"LLM_REVIEW_ENABLED"}


def _ensure_db_initialized() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    try:
        init_db()
    except DatabaseWriteUnavailableError:
        pass
    _DB_INITIALIZED = True


def _index_redirect(
    *,
    match_id: str = "",
    issue: str = "",
    message: str = "",
    level: str = "info",
):
    params = {}
    if match_id:
        params["match_id"] = match_id
    if issue:
        params["issue"] = issue
    if message:
        params["message"] = message
        if level:
            params["level"] = level
    return redirect(url_for("index", **params))


def _run_action(action, *, match_id: str = "", issue: str = ""):
    try:
        result = action()
    except DatabaseWriteUnavailableError as exc:
        return _index_redirect(
            match_id=match_id,
            issue=issue,
            message=str(exc),
            level="error",
        )
    except Exception as exc:  # noqa: BLE001
        return _index_redirect(
            match_id=match_id,
            issue=issue,
            message=f"操作失败：{exc}",
            level="error",
        )
    if isinstance(result, dict) and result.get("status_message"):
        return _index_redirect(
            match_id=match_id,
            issue=issue,
            message=str(result["status_message"]),
            level=str(result.get("status_level", "info") or "info"),
        )
    return _index_redirect(match_id=match_id, issue=issue)


def _selected_match_ids_from_form() -> list[str]:
    selected_match_ids: list[str] = []
    seen: set[str] = set()
    for raw_match_id in request.form.getlist("selected_match_ids"):
        match_id = str(raw_match_id or "").strip()
        if match_id and match_id not in seen:
            selected_match_ids.append(match_id)
            seen.add(match_id)
    return selected_match_ids


def _decorate_collection_row(row):
    if row is None:
        return None
    data = dict(row)
    status = str(data.get("collection_status", "") or "").strip()
    if status not in {"success", "failed", "uncollected"}:
        status = "success" if str(data.get("collected_at", "") or "").strip() else "uncollected"
        if status == "success" and get_collection_failure_reason(data):
            status = "failed"
    data["collection_status"] = status
    data["collection_failure_reason"] = (
        get_collection_failure_reason(data) if status == "failed" else ""
    )
    data["is_selectable_match"] = (
        str(data.get("source_match_url", "") or "").strip() == LIVE_SELECTABLE_URL
    )
    return data


def _decorate_collection_rows(rows):
    return [_decorate_collection_row(row) for row in rows]


def _normalize_score_probability(value) -> float:
    try:
        probability = float(str(value or "0").replace("%", ""))
    except ValueError:
        return 0.0
    if probability > 1:
        probability /= 100.0
    return max(0.0, min(probability, 1.0))


def _score_entry(entry, label: str):
    if not isinstance(entry, dict):
        return None
    score = str(entry.get("score", "") or "").strip()
    if not score:
        return None
    return {
        "label": label,
        "score": score,
        "probability": _normalize_score_probability(
            entry.get("probability", entry.get("prob", 0))
        ),
    }


def _parse_json_object(value: str):
    text = str(value or "").strip()
    if not text.startswith("{") or not text.endswith("}"):
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _handicap_magnitude_label(value) -> str:
    magnitude = abs(_safe_float(value))
    quarter = round(magnitude * 4) / 4
    mapping = {
        0.0: "平手",
        0.25: "平手/半球",
        0.5: "半球",
        0.75: "半球/一球",
        1.0: "一球",
        1.25: "一球/球半",
        1.5: "球半",
        1.75: "球半/两球",
        2.0: "两球",
        2.25: "两球/两球半",
        2.5: "两球半",
        2.75: "两球半/三球",
        3.0: "三球",
        3.25: "三球/三球半",
        3.5: "三球半",
        3.75: "三球半/四球",
        4.0: "四球",
    }
    return mapping.get(quarter, f"{quarter:g}球")


def _handicap_line_label(value) -> str:
    line = _safe_float(value)
    if abs(line) < 0.001:
        return "平手"
    magnitude = _handicap_magnitude_label(line)
    if line < 0:
        return f"主队让{magnitude}"
    return f"主队受让{magnitude}"


def _handicap_side_label(side: str, line_value) -> str:
    side = str(side or "").strip()
    line = _safe_float(line_value)
    if side not in {"home", "away"}:
        return "-"
    if abs(line) < 0.001:
        return "主队平手" if side == "home" else "客队平手"
    magnitude = _handicap_magnitude_label(line)
    if line < 0:
        return f"主队让{magnitude}" if side == "home" else f"客队受让{magnitude}"
    return f"主队受让{magnitude}" if side == "home" else f"客队让{magnitude}"


def _handicap_pick_label(action: str, side: str) -> str:
    action_text = str(action or "").strip() or "观望"
    side_text = {"home": "主队", "away": "客队"}.get(str(side or "").strip(), "")
    if action_text == "观望" or not side_text:
        return action_text
    return f"{action_text}{side_text}"


def _handicap_suggested_stake_pct(data) -> float:
    action = str(data.get("handicap_recommendation", "") or "").strip()
    side = str(data.get("handicap_recommended_side", "") or "").strip()
    if action == "观望" or side not in {"home", "away"}:
        return 0.0

    raw_odds = _safe_float(data.get(f"handicap_{side}_odds"))
    decimal_odds = raw_odds + 1.0 if raw_odds > 0 and raw_odds < 1.5 else raw_odds
    cover_prob = _safe_float(data.get(f"handicap_{side}_cover_prob"))
    risk = {
        "recommended_outcome": side,
        "probabilities": {side: cover_prob},
        "market_odds": {side: decimal_odds},
        "expected_values": {side: _safe_float(data.get("handicap_expected_value"))},
        "confidence": _safe_float(data.get("handicap_confidence")),
        "quality_score": _safe_float(data.get("quality_score"), 0.70),
        "model_agreement": 0.70,
        "action_score": 0.82 if action == "主推" else 0.64,
    }
    return stake_for_action(action, risk)


def _decorate_prediction_run(row):
    if row is None:
        return None
    data = dict(row)
    items = []
    raw_payload = _parse_json_object(data.get("predicted_score_raw", ""))
    for key, label in (
        ("most_likely", "最可能"),
        ("second_1", "次选一"),
        ("second_2", "次选二"),
        ("upset", "冷门"),
    ):
        item = _score_entry(raw_payload.get(key), label)
        if item:
            items.append(item)

    if not items:
        primary_score = str(data.get("predicted_score", "") or "").strip()
        if primary_score:
            items.append(
                {
                    "label": "最可能",
                    "score": primary_score,
                    "probability": _normalize_score_probability(
                        data.get("predicted_score_confidence", 0)
                    ),
                }
            )
        try:
            candidates = json.loads(str(data.get("quant_score_candidates", "") or "[]"))
        except json.JSONDecodeError:
            candidates = []
        if isinstance(candidates, list):
            labels = ("次选一", "次选二", "冷门")
            seen_scores = {item["score"] for item in items}
            for candidate in candidates:
                item = _score_entry(candidate, labels[min(len(items) - 1, 2)])
                if not item or item["score"] in seen_scores:
                    continue
                items.append(item)
                seen_scores.add(item["score"])
                if len(items) >= 4:
                    break

    data["score_display_items"] = items[:4]
    data["handicap_line_label"] = _handicap_line_label(data.get("handicap_line"))
    data["handicap_initial_line_label"] = _handicap_line_label(data.get("handicap_initial_line"))
    data["handicap_recommended_side_label"] = _handicap_side_label(
        data.get("handicap_recommended_side"),
        data.get("handicap_line"),
    )
    data["handicap_pick_label"] = _handicap_pick_label(
        data.get("handicap_recommendation"),
        data.get("handicap_recommended_side"),
    )
    data["handicap_suggested_stake_pct"] = _handicap_suggested_stake_pct(data)
    handicap_reason = str(data.get("handicap_reason", "") or "")
    if handicap_reason:
        raw_current = f"当前盘口 {_safe_float(data.get('handicap_line')):+.3f}"
        raw_initial = f"初盘 {_safe_float(data.get('handicap_initial_line')):+.3f}"
        handicap_reason = handicap_reason.replace(raw_current, f"当前盘口 {data['handicap_line_label']}")
        handicap_reason = handicap_reason.replace(raw_initial, f"初盘 {data['handicap_initial_line_label']}")
    data["handicap_reason_display"] = handicap_reason
    return data


def _summarize_top_picks(top_picks):
    settled = [pick for pick in top_picks if pick.get("is_settled")]
    hit_count = sum(1 for pick in settled if int(pick.get("handicap_hit") or pick.get("hit_recommendation") or 0))
    total = len(settled)
    return {
        "settled_count": total,
        "hit_count": hit_count,
        "miss_count": max(total - hit_count, 0),
        "hit_rate": (hit_count / total) if total else 0.0,
    }


def _historical_issues_before(issue: str = "") -> list[str]:
    issues = list_issues()
    if not issues:
        return []
    current_issue = str(issue or issues[0]).strip()
    if current_issue in issues:
        return issues[issues.index(current_issue) + 1 :]
    return issues[1:]


def _run_historical_predictions(issue: str = "", progress_callback=None) -> dict:
    issues = _historical_issues_before(issue)
    total_issues = len(issues)
    if not issues:
        message = "没有可回测的历史期号。"
        return {
            "issue_count": 0,
            "predicted_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "status_message": message,
            "task_message": message,
            "status_level": "info",
        }

    predicted_count = 0
    skipped_count = 0
    failed_count = 0
    for index, historical_issue in enumerate(issues, start=1):
        if progress_callback is not None:
            progress_callback(
                total_items=total_issues,
                completed_items=index - 1,
                current_item_index=index,
                current_item_label=f"期号 {historical_issue}",
                current_step="历史预测回测",
                message=f"正在回测第 {index}/{total_issues} 个历史期号：{historical_issue}",
            )
        result = predict_issue(historical_issue, ensure_collected=False, use_llm=False)
        predicted_count += int(result.get("predicted_count", 0) or 0)
        skipped_count += int(result.get("skipped_count", 0) or 0)
        failed_count += int(result.get("prediction_failed_count", 0) or 0)

    message = (
        f"历史回测完成：共 {total_issues} 期，生成预测 {predicted_count} 场，"
        f"跳过 {skipped_count} 场，失败 {failed_count} 场。"
    )
    status_level = "success" if failed_count == 0 else "warning"
    if progress_callback is not None:
        progress_callback(
            total_items=total_issues,
            completed_items=total_issues,
            current_item_index=total_issues,
            current_item_label=f"期号 {issues[-1]}",
            current_step="历史回测完成",
            message=message,
            level=status_level,
        )
    return {
        "issue_count": total_issues,
        "predicted_count": predicted_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "status_message": message,
        "task_message": message,
        "status_level": status_level,
    }


def _run_historical_settlement(issue: str = "", progress_callback=None) -> dict:
    issues = _historical_issues_before(issue)
    total_issues = len(issues)
    if not issues:
        message = "没有可结算的历史期号。"
        return {
            "issue_count": 0,
            "result_synced_count": 0,
            "settled_count": 0,
            "skipped_count": 0,
            "status_message": message,
            "task_message": message,
            "status_level": "info",
        }

    result_synced_count = 0
    settled_count = 0
    skipped_count = 0
    for index, historical_issue in enumerate(issues, start=1):
        if progress_callback is not None:
            progress_callback(
                total_items=total_issues,
                completed_items=index - 1,
                current_item_index=index,
                current_item_label=f"期号 {historical_issue}",
                current_step="历史赛果结算",
                message=f"正在结算第 {index}/{total_issues} 个历史期号：{historical_issue}",
            )
        result = settle_issue_results(historical_issue)
        result_synced_count += int(result.get("result_synced_count", 0) or 0)
        settled_count += int(result.get("settled_count", 0) or 0)
        skipped_count += int(result.get("skipped_count", 0) or 0)

    message = (
        f"历史结算完成：共 {total_issues} 期，同步赛果 {result_synced_count} 场，"
        f"结算反馈 {settled_count} 场，跳过 {skipped_count} 场。"
    )
    status_level = "success" if skipped_count == 0 else "warning"
    if progress_callback is not None:
        progress_callback(
            total_items=total_issues,
            completed_items=total_issues,
            current_item_index=total_issues,
            current_item_label=f"期号 {issues[-1]}",
            current_step="历史结算完成",
            message=message,
            level=status_level,
        )
    return {
        "issue_count": total_issues,
        "result_synced_count": result_synced_count,
        "settled_count": settled_count,
        "skipped_count": skipped_count,
        "status_message": message,
        "task_message": message,
        "status_level": status_level,
    }


def _wants_async_json() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _task_progress_callback(
    task_id: str,
    *,
    total_items: int = 0,
    current_item_index: int = 0,
    current_item_label: str = "",
):
    def _callback(**payload):
        update_payload = {"status": "running", "level": "info"}
        if total_items > 0:
            update_payload["total_items"] = total_items
        if current_item_index > 0:
            update_payload["current_item_index"] = current_item_index
        if current_item_label:
            update_payload["current_item_label"] = current_item_label
        update_payload.update(payload)
        update_task(task_id, **update_payload)

    return _callback


def _start_background_task(
    *,
    kind: str,
    title: str,
    action,
    match_id: str = "",
    issue: str = "",
):
    task_id = create_task(kind, title, match_id=match_id, issue=issue)

    def _result_completion(result) -> tuple[str, str]:
        if not isinstance(result, dict):
            return "", "success"
        message = str(result.get("task_message") or result.get("status_message") or "")
        level = str(result.get("status_level", "success") or "success")
        return message, level

    def _runner():
        try:
            update_task(task_id, status="running", message="任务已开始", level="info")
            result = action(task_id)
            message, level = _result_completion(result)
            complete_task(task_id, message, level=level)
        except DatabaseWriteUnavailableError as exc:
            fail_task(task_id, str(exc))
        except Exception as exc:  # noqa: BLE001
            fail_task(task_id, f"操作失败：{exc}")

    Thread(target=_runner, daemon=True).start()
    return {
        "task_id": task_id,
        "status_url": url_for("task_status", task_id=task_id),
        "view_url": url_for("index", match_id=match_id, issue=issue, task_id=task_id),
        "complete_url": url_for("index", match_id=match_id, issue=issue),
    }


@app.route("/")
def index():
    _ensure_db_initialized()
    selected_match_id = request.args.get("match_id", "")
    selected_issue = request.args.get("issue", "")
    show_select_matches = request.args.get("select_matches", "").strip() == "1"
    active_task_id = request.args.get("task_id", "").strip()
    status_message = request.args.get("message", "").strip()
    status_level = request.args.get("level", "info").strip() or "info"
    db_status = get_database_status()

    matches = _decorate_collection_rows(list_matches_by_issue(selected_issue or None))
    if not matches:
        if db_status["read_only"]:
            if not status_message:
                status_message = db_status["message"]
                status_level = "warning"
        else:
            try:
                sync_matches()
                matches = _decorate_collection_rows(list_matches_by_issue(selected_issue or None))
            except DatabaseWriteUnavailableError as exc:
                db_status = get_database_status()
                status_message = status_message or str(exc)
                status_level = "error"
            except Exception as exc:  # noqa: BLE001
                status_message = status_message or f"同步当前对赛失败：{exc}"
                status_level = "error"

    issues = list_issues()

    if selected_issue and selected_issue not in issues:
        selected_issue = issues[0] if issues else ""
        matches = _decorate_collection_rows(list_matches_by_issue(selected_issue or None))

    if not selected_issue and issues:
        selected_issue = issues[0]
        matches = _decorate_collection_rows(list_matches_by_issue(selected_issue))

    available_match_ids = {match["match_id"] for match in matches}
    if matches and selected_match_id not in available_match_ids:
        selected_match_id = matches[0]["match_id"]

    if not db_status["read_only"]:
        try:
            expire_pending_manual_reviews()
        except DatabaseWriteUnavailableError:
            db_status = get_database_status()

    current = (
        _decorate_collection_row(get_match_analysis(selected_match_id))
        if selected_match_id
        else None
    )
    prediction_runs = list_prediction_runs(selected_match_id, limit=1) if selected_match_id else []
    prediction_runs = [_decorate_prediction_run(row) for row in prediction_runs]
    latest_prediction = prediction_runs[0] if prediction_runs else None
    canonical_prediction = _decorate_prediction_run(
        get_canonical_prediction_run(selected_match_id)
    ) if selected_match_id else None
    canonical_feedback = (
        get_feedback_log(int(canonical_prediction["run_id"]))
        if canonical_prediction is not None
        else None
    )
    pending_manual_reviews = list_pending_manual_review_runs(selected_issue or None, limit=12)
    selectable_matches = []
    selectable_error = ""
    if show_select_matches and not db_status["read_only"]:
        try:
            selectable_matches = list_selectable_matches(selected_issue)
        except Exception as exc:  # noqa: BLE001
            selectable_error = f"读取可选对赛失败：{exc}"

    stats = get_collection_stats(selected_issue or None)
    feedback_summary = get_feedback_summary()
    prediction_result_summary = get_feedback_summary(selected_issue or None)
    learning_overview = get_learning_overview(request.args.get("learning_window_issue_count", ""))

    # TOP3 picks for current issue
    topic_issue = selected_issue or (issues[0] if issues else '')
    top_picks = get_issue_top_picks(topic_issue) if topic_issue else []
    if not top_picks and topic_issue:
        try:
            # Auto-compute if not yet done
            compute_issue_top_picks(topic_issue)
            top_picks = get_issue_top_picks(topic_issue)
        except Exception:
            top_picks = []
    top_picks_summary = _summarize_top_picks(top_picks)
    if db_status["message"] and not status_message:
        status_message = db_status["message"]
        status_level = db_status.get("level", "info")

    return render_template(
        "index.html",
        matches=matches,
        current=current,
        sections=build_sections(current),
        latest_prediction=latest_prediction,
        canonical_prediction=canonical_prediction,
        canonical_feedback=canonical_feedback,
        prediction_runs=prediction_runs,
        pending_manual_reviews=pending_manual_reviews,
        selected_match_id=selected_match_id,
        selected_issue=selected_issue,
        available_match_ids=available_match_ids,
        issues=issues,
        stats=stats,
        feedback_summary=feedback_summary,
        prediction_result_summary=prediction_result_summary,
        learning_overview=learning_overview,
        db_status=db_status,
        status_message=status_message,
        status_level=status_level,
        active_task_id=active_task_id,
        show_select_matches=show_select_matches,
        selectable_matches=selectable_matches,
        selectable_error=selectable_error,
        top_picks=top_picks,
        top_picks_summary=top_picks_summary,
        app_version=APP_VERSION,
    )


@app.route("/sync", methods=["GET", "POST"])
def sync_view():
    _ensure_db_initialized()
    if request.method == "GET":
        return _index_redirect(
            message="`/sync` 是表单动作路由，请从首页按钮触发同步。",
            level="info",
        )

    return _run_action(lambda: sync_matches(return_details=True))


@app.post("/sync-issue")
def sync_issue_view():
    _ensure_db_initialized()
    issue = request.form.get("manual_issue", "").strip()
    return _run_action(
        lambda: sync_issue_matches(issue, return_details=True),
        issue=issue,
    )


@app.post("/select-matches")
def select_matches_view():
    _ensure_db_initialized()
    issue = request.form.get("issue", "").strip()
    selected_match_ids = request.form.getlist("selected_match_ids")
    result = add_selectable_matches(
        selected_match_ids,
        issue=issue,
        return_details=True,
    )
    matches = result.get("matches", []) if isinstance(result, dict) else []
    match_id = str(matches[0].get("match_id", "") if matches else request.form.get("match_id", "")).strip()
    return _index_redirect(
        match_id=match_id,
        issue=str(result.get("issue", issue) if isinstance(result, dict) else issue),
        message=str(result.get("status_message", "") if isinstance(result, dict) else ""),
        level=str(result.get("status_level", "info") if isinstance(result, dict) else "info"),
    )


@app.post("/select-matches/<match_id>/delete")
def delete_selectable_match_view(match_id: str):
    _ensure_db_initialized()
    issue = request.form.get("issue", "").strip()
    current_match_id = request.form.get("current_match_id", "").strip()
    result = remove_selectable_match(match_id, issue=issue, return_details=True)
    next_match_id = "" if current_match_id == match_id else current_match_id
    return _index_redirect(
        match_id=next_match_id,
        issue=str(result.get("issue", issue) if isinstance(result, dict) else issue),
        message=str(result.get("status_message", "") if isinstance(result, dict) else ""),
        level=str(result.get("status_level", "info") if isinstance(result, dict) else "info"),
    )


@app.get("/tasks/<task_id>")
def task_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(task)


@app.post("/collect/<match_id>")
def collect_single(match_id: str):
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="collect-single",
            title="采集当前对赛",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: collect_match(
                match_id,
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                ),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: collect_match(match_id),
        match_id=match_id,
        issue=issue,
    )


@app.post("/collect-all")
def collect_all():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    selected_match_ids = _selected_match_ids_from_form()
    if _wants_async_json():
        payload = _start_background_task(
            kind="collect-all",
            title="采集勾选对赛" if selected_match_ids else "采集全部对赛",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: collect_all_matches(
                issue or None,
                progress_callback=_task_progress_callback(task_id),
                return_details=True,
                match_ids=selected_match_ids or None,
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: collect_all_matches(
            issue or None,
            return_details=True,
            match_ids=selected_match_ids or None,
        ),
        match_id=match_id,
        issue=issue,
    )


@app.post("/predict/<match_id>")
def predict_single(match_id: str):
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="predict-single",
            title="预测当前对赛",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: predict_match(
                match_id,
                ensure_collected=False,
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                ),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: predict_match(match_id, ensure_collected=False),
        match_id=match_id,
        issue=issue,
    )


def _predict_issue_and_refresh_top_picks(
    issue: str | None,
    *,
    progress_callback=None,
    match_ids: list[str] | None = None,
):
    result = predict_issue(
        issue or None,
        ensure_collected=False,
        progress_callback=progress_callback,
        match_ids=match_ids,
    )
    try:
        effective_issue = str(issue or "").strip() or (list_issues()[0] if list_issues() else "")
        if effective_issue:
            compute_issue_top_picks(effective_issue)
    except Exception:
        pass
    return result


@app.post("/predict-all")
def predict_all():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    selected_match_ids = _selected_match_ids_from_form()
    if _wants_async_json():
        payload = _start_background_task(
            kind="predict-all",
            title="预测勾选对赛" if selected_match_ids else "预测当前期全部对赛",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: _predict_issue_and_refresh_top_picks(
                issue,
                progress_callback=_task_progress_callback(task_id),
                match_ids=selected_match_ids or None,
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: _predict_issue_and_refresh_top_picks(
            issue,
            match_ids=selected_match_ids or None,
        ),
        match_id=match_id,
        issue=issue,
    )


@app.post("/settle-all")
def settle_all():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    selected_match_ids = _selected_match_ids_from_form()
    if _wants_async_json():
        payload = _start_background_task(
            kind="settle-all",
            title="同步并结算勾选对赛" if selected_match_ids else "同步赛果并结算当前期",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: settle_issue_results(
                issue or None,
                progress_callback=_task_progress_callback(task_id),
                match_ids=selected_match_ids or None,
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: settle_issue_results(
            issue or None,
            match_ids=selected_match_ids or None,
        ),
        match_id=match_id,
        issue=issue,
    )


@app.post("/history/backtest")
def history_backtest():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="history-backtest",
            title="回测历史预测",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: _run_historical_predictions(
                issue,
                progress_callback=_task_progress_callback(task_id),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: _run_historical_predictions(issue),
        match_id=match_id,
        issue=issue,
    )


@app.post("/history/settle")
def history_settle():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="history-settle",
            title="结算历史赛果",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: _run_historical_settlement(
                issue,
                progress_callback=_task_progress_callback(task_id),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: _run_historical_settlement(issue),
        match_id=match_id,
        issue=issue,
    )


@app.post("/settle/<match_id>")
def settle_single(match_id: str):
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="settle-single",
            title="同步赛果并结算当前场次",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: settle_match_result(
                match_id,
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                ),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: settle_match_result(match_id),
        match_id=match_id,
        issue=issue,
    )


@app.post("/manual-review/<int:run_id>/resolve")
def manual_review_resolve(run_id: int):
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    return _run_action(
        lambda: resolve_manual_review(
            run_id,
            request.form.get("effective_recommendation", "").strip(),
            request.form.get("notes", "").strip(),
        ),
        match_id=match_id,
        issue=issue,
    )


@app.post("/learning/train")
def learning_train():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    window_issue_count = request.form.get("learning_window_issue_count", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="learning-train",
            title="学习训练",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: train_learning_profile(
                window_issue_count=window_issue_count,
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                ),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: train_learning_profile(window_issue_count=window_issue_count),
        match_id=match_id,
        issue=issue,
    )


@app.post("/learning/import-history")
def learning_import_history():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="learning-import-history",
            title="导入历史闭环",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: import_historical_learning_feedback(
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                ),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: import_historical_learning_feedback(),
        match_id=match_id,
        issue=issue,
    )


@app.post("/learning/replay-backfill")
def learning_replay_backfill():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="learning-replay-backfill",
            title="回放补齐历史期",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: replay_backfill_learning_feedback(
                progress_callback=_task_progress_callback(task_id),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: replay_backfill_learning_feedback(),
        match_id=match_id,
        issue=issue,
    )


@app.post("/learning/activate/<int:profile_id>")
def learning_activate(profile_id: int):
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    if _wants_async_json():
        payload = _start_background_task(
            kind="learning-activate",
            title="启用学习配置",
            match_id=match_id,
            issue=issue,
            action=lambda task_id: activate_learning_profile(
                profile_id,
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                ),
            ),
        )
        return jsonify(payload)
    return _run_action(
        lambda: activate_learning_profile(profile_id),
        match_id=match_id,
        issue=issue,
    )


@app.post("/learning/deactivate")
def learning_deactivate():
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    match_id = request.form.get("match_id", "")
    if _wants_async_json():
        def _deactivate_action(task_id: str):
            return deactivate_learning_profile(
                progress_callback=_task_progress_callback(
                    task_id,
                    total_items=1,
                    current_item_index=1,
                )
            )

        payload = _start_background_task(
            kind="learning-deactivate",
            title="停用学习配置",
            match_id=match_id,
            issue=issue,
            action=_deactivate_action,
        )
        return jsonify(payload)
    return _run_action(
        lambda: deactivate_learning_profile(),
        match_id=match_id,
        issue=issue,
    )


@app.post("/feedback/<match_id>")
def save_feedback(match_id: str):
    _ensure_db_initialized()
    issue = request.form.get("issue", "")
    run_id = int(request.form.get("run_id", "0") or 0)
    roi_text = request.form.get("roi_delta", "").strip()
    roi_delta = float(roi_text) if roi_text else None
    return _run_action(
        lambda: record_feedback(
            prediction_run_id=run_id,
            match_id=match_id,
            actual_result=request.form.get("actual_result", "").strip(),
            actual_score=request.form.get("actual_score", "").strip(),
            roi_delta=roi_delta,
            notes=request.form.get("notes", "").strip(),
        ),
        match_id=match_id,
        issue=issue,
    )


@app.route("/config")
def config_page():
    _ensure_db_initialized()
    config = load_env_config()
    return render_template("config.html", config=config)


@app.post("/config/save")
def config_save():
    config = load_env_config()
    for key in LEGACY_MODEL_KEYS:
        config.pop(key, None)
    for key in CONFIG_FORM_KEYS:
        if key in CONFIG_BOOLEAN_KEYS:
            config[key] = "true" if request.form.get(key, "").strip() else "false"
        elif key in request.form:
            config[key] = request.form[key].strip()
    save_env_config(config)
    reload_env_to_os()
    return redirect(url_for("config_page"))


@app.post("/config/test-openai")
def config_test_openai():
    base_url = request.form.get("base_url", "").strip()
    api_key = request.form.get("api_key", "").strip()
    model = request.form.get("model", "").strip()
    result = test_openai_api(base_url, api_key, model)
    return jsonify(result)


@app.post("/config/test-collection")
def config_test_collection():
    base_url = request.form.get("base_url", "").strip()
    api_key = request.form.get("api_key", "").strip()
    model = request.form.get("model", "").strip()
    max_tokens = request.form.get("max_tokens", "").strip()
    result = test_collection_api(base_url, api_key, model, max_tokens=max_tokens)
    return jsonify(result)


def main() -> None:
    try:
        init_db()
    except DatabaseWriteUnavailableError as exc:
        print(f"[warn] {exc}")
    reload_env_to_os()
    debug = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    try:
        port = int(os.getenv("FOOTBALL_APP_PORT", "5050"))
    except ValueError:
        port = 5050
    app.run(
        host="127.0.0.1",
        port=port,
        debug=debug,
        use_reloader=debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
