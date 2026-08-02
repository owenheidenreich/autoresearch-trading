"""Immutable scientific oracle for the Path-D Tier-S research build.

This file is intentionally frozen before the implementation modules exist.  Future
modules are imported only inside test bodies, so collection succeeds while missing
machinery fails loudly at execution.  The tests use fixed vectors and independent
reconstructions rather than accepting self-declared PASS fields.
"""
from __future__ import annotations
import ast
import copy
from dataclasses import fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import numpy as np
import pytest
from v4.research import pathd_entry_exit as prereg

def _api_names() -> tuple[str, ...]:
    return tuple((row['qualified_name'] for row in prereg.entry_future_api_contract()['functions']))

def _dummy_fit_authorization() -> prereg.FrozenFitAuthorization:
    sessions = ('2025-08-01', '2025-08-04')
    return prereg.FrozenFitAuthorization(role='outer_weights', outer_fold=1, inner_fold=None, sessions=sessions, sessions_sha256_newline=prereg.canonical_session_hash(sessions), preregistration_sha256='1' * 64, session_assignments_sha256='2' * 64, source_hash_policy_sha256='3' * 64, corpus_integrity_receipt_sha256='4' * 64, lineage_receipt_sha256='5' * 64, machinery_receipt_sha256='6' * 64, fit_environment_sha256='7' * 64, entry_pooled_acceptance_receipt_sha256=None)

def _assert_prereg_mutations_rejected(mutations: tuple[Callable[[dict[str, Any]], None], ...]) -> None:
    payload, assignments, lineage = prereg.preregistration_payload()
    prereg.validate_preregistration_payload(payload, assignments, lineage)
    for mutate in mutations:
        changed = copy.deepcopy(payload)
        mutate(changed)
        with pytest.raises((TypeError, ValueError)):
            prereg.validate_preregistration_payload(changed, assignments, lineage)

def _statistics_tuple(value: Any) -> tuple[np.ndarray, ...]:
    """Normalize the public statistic result without weakening its numeric oracle."""
    names = ('mean_lcb_dollars', 'mean_lcb_return', 'q10_dollars', 'q10_return')
    if type(value) is dict:
        rows = tuple((value[name] for name in names))
    elif all((hasattr(value, name) for name in names)):
        rows = tuple((getattr(value, name) for name in names))
    else:
        rows = tuple(value)
    assert len(rows) == 4
    return tuple((np.asarray(row, dtype=np.float64) for row in rows))

def _reseal_dataclass(value: Any, hash_field: str='artifact_sha256') -> Any:
    semantic = {field.name: getattr(value, field.name) for field in fields(value) if field.name != hash_field}
    return replace(value, **{hash_field: prereg.stable_hash(semantic)})

def _coverage_membership_fixture(ordered: list[str], dispositions: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    session = '2026-01-02'
    authorization_sha256 = 'f' * 64
    transitions: list[dict[str, Any]] = [{
        'event_schema_version': 'pathd.research_ledger_control_event.v1',
        'prior_authorization_sha256': None,
        'next_authorization_sha256': authorization_sha256,
        'event_payload': {'event_kind': 'GENESIS'},
    }]
    for digest, disposition in zip(ordered, dispositions, strict=True):
        transitions.append({'event_schema_version': 'pathd.entry_frame_coverage.v1', 'prior_authorization_sha256': authorization_sha256, 'next_authorization_sha256': authorization_sha256, 'event_payload': {'example_sha256': digest, 'disposition': disposition}})
        if disposition == 'ACTION_DECISION':
            transitions.append({'event_schema_version': 'pathd.entry_action_decision.v1', 'prior_authorization_sha256': authorization_sha256, 'next_authorization_sha256': authorization_sha256, 'event_payload': {'example_sha256': digest}})
            transitions.append({'event_schema_version': 'pathd.entry_action_observation.v1', 'prior_authorization_sha256': authorization_sha256, 'next_authorization_sha256': authorization_sha256, 'event_payload': {'example_sha256': digest}})
    receipt = {'sessions': [session], 'ordered_example_hashes': ordered, 'session_example_hashes': [{'session': session, 'ordered_example_hashes': ordered}]}
    return ({'authorization_sha256': authorization_sha256, 'terminal_journal': {'transitions': transitions}}, receipt)

def _seed_record(*, purpose: str, model_family: str, fold_scope: str, statistic: str) -> tuple[str, int]:
    key = {'campaign': 'pathd.tier_s.v1', 'plan_sha256': prereg.sha256_path(prereg.PLAN_PATH), 'purpose': purpose, 'model_family': model_family, 'fold_scope': fold_scope, 'statistic': statistic}
    raw = json.dumps(key, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    return (digest, int.from_bytes(bytes.fromhex(digest)[:4], 'big'))

def _matched_random_key(*, seed: int, outer_fold: int, proposal: dict[str, Any]) -> tuple[int, str]:
    key = {
        'attempt_seed_id': seed,
        'outer_fold': outer_fold,
        'session': proposal['session'],
        'decision_time_ns': proposal['decision_time_ns'],
        'source_neutral_contract_id': proposal['source_neutral_contract_id'],
        'expiry_yyyymmdd': proposal['expiry_yyyymmdd'],
        'strike_milli_points': proposal['strike_milli_points'],
        'right_code': proposal['right_code'],
    }
    raw = json.dumps(key, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    return (int.from_bytes(bytes.fromhex(digest)[:8], 'big'), digest)

def _journal_events(*, session: str, policy_id: str, pnl_micros: int, trade_count: int) -> tuple[dict[str, Any], ...]:
    cash = 10000000000
    events: list[dict[str, Any]] = []
    for trade_index in range(trade_count):
        trade_id = f'{policy_id}:{session}:{trade_index}'
        buy_delta = -100000000
        sell_delta = 100000000 + (pnl_micros if trade_index == trade_count - 1 else 0)
        for kind, delta, position_after in (('BUY_FILL', buy_delta, 1), ('SELL_FILL', sell_delta, 0)):
            event = {'sequence': len(events) + 1, 'kind': kind, 'trade_id': trade_id, 'cash_before_micros': cash, 'cash_delta_micros': delta, 'cash_after_micros': cash + delta, 'position_after': position_after}
            events.append(event)
            cash += delta
    return tuple(events)

def _reconstruct_fixture_journal(journal: dict[str, Any]) -> tuple[int, list[str], int]:
    events = journal.get('transitions')
    if type(events) not in (list, tuple) or not events:
        raise ValueError('journal transitions')
    cash = events[0].get('cash_before_micros')
    if type(cash) is not int:
        raise ValueError('journal cash')
    starting_cash = cash
    position = 0
    active_trade: str | None = None
    completed: list[str] = []
    for index, event in enumerate(events, 1):
        if type(event) is not dict or event.get('sequence') != index:
            raise ValueError('journal sequence')
        before = event.get('cash_before_micros')
        delta = event.get('cash_delta_micros')
        after = event.get('cash_after_micros')
        trade_id = event.get('trade_id')
        if any(type(value) is not int for value in (before, delta, after)) or type(trade_id) is not str:
            raise ValueError('journal types')
        if before != cash or before + delta != after:
            raise ValueError('journal cash chain')
        if event.get('kind') == 'BUY_FILL':
            if position != 0 or event.get('position_after') != 1:
                raise ValueError('journal buy state')
            position = 1
            active_trade = trade_id
        elif event.get('kind') == 'SELL_FILL':
            if position != 1 or event.get('position_after') != 0 or trade_id != active_trade:
                raise ValueError('journal sell state')
            position = 0
            active_trade = None
            completed.append(trade_id)
        else:
            raise ValueError('journal event kind')
        cash = after
    if position != 0 or active_trade is not None:
        raise ValueError('journal nonflat')
    return (cash - starting_cash, completed, cash)

def _seal_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(record)
    semantic.pop('record_sha256', None)
    record['record_sha256'] = prereg.stable_hash(semantic)
    return record

def _holdout_trace_dict(*, session: str, schema_version: str, guard_keys: tuple[str, ...], authorization_sha256: str, dataset_sha256: str, box_trade_count: int=2, box_pnl_micros: int=100, comparator_trade_count: int=1, comparator_pnl_micros: int=50) -> dict[str, Any]:
    box_events = _journal_events(session=session, policy_id='BOX_D', pnl_micros=box_pnl_micros, trade_count=box_trade_count)
    comparator_events = _journal_events(session=session, policy_id='COMPARATOR', pnl_micros=comparator_pnl_micros, trade_count=comparator_trade_count)
    box_ids = [f'BOX_D:{session}:{index}' for index in range(box_trade_count)]
    comparator_ids = [f'COMPARATOR:{session}:{index}' for index in range(comparator_trade_count)]
    return _seal_trace_record({
        'schema_version': schema_version,
        'holdout_caveat': prereg.HOLDOUT_CAVEAT, 'session': session,
        'authorization_sha256': authorization_sha256, 'dataset_sha256': dataset_sha256,
        'box_d_policy_id': 'BOX_D', 'comparator_policy_id': 'COMPARATOR',
        'box_d_terminal_journal': {'transitions': box_events, 'session_pnl_micros': box_pnl_micros, 'completed_trade_ids': box_ids},
        'comparator_terminal_journal': {'transitions': comparator_events, 'session_pnl_micros': comparator_pnl_micros, 'completed_trade_ids': comparator_ids},
        'box_d_session_pnl_micros': box_pnl_micros,
        'comparator_session_pnl_micros': comparator_pnl_micros,
        'box_d_completed_trade_ids': box_ids,
        'guard_results': {name: True for name in guard_keys},
        'survival_violations': [],
    })

def _validate_trace_record(record: dict[str, Any], expected: dict[str, Any]) -> None:
    required = {'schema_version', 'holdout_caveat', 'session', 'authorization_sha256', 'dataset_sha256', 'box_d_policy_id', 'comparator_policy_id', 'box_d_terminal_journal', 'comparator_terminal_journal', 'box_d_session_pnl_micros', 'comparator_session_pnl_micros', 'box_d_completed_trade_ids', 'guard_results', 'survival_violations', 'record_sha256'}
    if set(record) != required:
        raise ValueError('trace schema')
    if record['schema_version'] != 'pathd.protected_holdout_trace_record.v1' or record['holdout_caveat'] != prereg.HOLDOUT_CAVEAT:
        raise ValueError('trace classification')
    for key in ('authorization_sha256', 'dataset_sha256', 'box_d_policy_id', 'comparator_policy_id'):
        if record[key] != expected[key]:
            raise ValueError('trace identity')
    if record['session'] not in expected['sessions']:
        raise ValueError('trace partition')
    if type(record['guard_results']) is not dict or set(record['guard_results']) != set(expected['guard_keys']) or not all(type(value) is bool and value is True for value in record['guard_results'].values()):
        raise ValueError('trace guards')
    if record['survival_violations'] != []:
        raise ValueError('trace survival')
    box_journal = record['box_d_terminal_journal']
    comparator_journal = record['comparator_terminal_journal']
    box_pnl, box_ids, _box_cash = _reconstruct_fixture_journal(box_journal)
    comparator_pnl, comparator_ids, _comparator_cash = _reconstruct_fixture_journal(comparator_journal)
    if box_pnl != record['box_d_session_pnl_micros'] or box_journal.get('session_pnl_micros') != box_pnl or box_ids != record['box_d_completed_trade_ids'] or box_journal.get('completed_trade_ids') != box_ids or comparator_pnl != record['comparator_session_pnl_micros'] or comparator_journal.get('session_pnl_micros') != comparator_pnl or comparator_journal.get('completed_trade_ids') != comparator_ids:
        raise ValueError('trace journal reconstruction')
    semantic = dict(record)
    observed = semantic.pop('record_sha256')
    if observed != prereg.stable_hash(semantic):
        raise ValueError('trace hash')

def test_authorized_dependency_closure_rejects_unregistered_v4_and_test_imports() -> None:
    payload = prereg.preregistration_payload()[0]
    policy = payload['source_hash_policy']
    authorized = set(policy['authorized_in_repo_dependency_paths'])
    runtime = set(policy['authorized_runtime_dependency_paths'])
    assert 'v4/tests/test_pathd_fixed_science_contract.py' in authorized
    assert 'v4/tests/test_pathd_fixed_science_contract.py' not in runtime
    assert 'random' in prereg.FROZEN_OFFLINE_IMPORT_ROOTS
    fixed_path = Path('v4/tests/test_pathd_fixed_science_contract.py')
    fixed_tree = ast.parse(fixed_path.read_text(encoding='utf-8'))
    prereg._reject_dynamic_dependency_escape(fixed_path.as_posix(), fixed_tree)
    assert prereg._resolved_v4_import_paths(fixed_path.as_posix(), fixed_tree) <= authorized
    live_tree = ast.parse('from v4.live import protocol101_entry\n')
    live_paths = prereg._resolved_v4_import_paths('v4/research/probe.py', live_tree)
    assert 'v4/live/protocol101_entry.py' in live_paths
    assert live_paths - runtime
    test_tree = ast.parse('from v4.tests import test_pathd_entry_exit_gate_frozen\n')
    test_paths = prereg._resolved_v4_import_paths('v4/research/probe.py', test_tree)
    assert any((path.startswith('v4/tests/') for path in test_paths))
    assert test_paths - runtime
    for source in ("import importlib\nimportlib.import_module('v4.live.protocol101_entry')\n", 'import joblib\njoblib.Parallel(n_jobs=2)\n', 'import os\nos.fork()\n'):
        with pytest.raises(RuntimeError):
            prereg._reject_dynamic_dependency_escape('v4/research/pathd_entry_models.py', ast.parse(source))

def test_entry_family_selection_is_globally_frozen_before_any_outer_open() -> None:
    from v4.research import pathd_entry_models as models
    spec = prereg.entry_global_family_spec()
    assert spec['selectable_candidate'] == 'HGB'
    assert spec['selectable_candidates'] == ['HGB']
    assert spec['mandatory_challenger'] == 'NEURAL'
    assert spec['outer_folds'] == ['HGB'] * 5
    assert spec['entry_acceptance_candidate'] == 'HGB'
    assert spec['exit_oof_trajectory_entry_family'] == 'HGB'
    assert spec['neural_outcome_can_select_replace_rescue_or_reweight_hgb'] is False
    assert 'EntryFamilySelectionV1' not in prereg.entry_future_api_contract()['schema_versions']
    assert not hasattr(models, 'select_entry_family')
    _assert_prereg_mutations_rejected((lambda value: value['entry']['global_family_freeze'].update(selectable_candidate='NEURAL'), lambda value: value['entry']['global_family_freeze'].update(selectable_candidates=['HGB', 'NEURAL']), lambda value: value['entry']['global_family_freeze'].update(outer_folds=['HGB', 'HGB', 'NEURAL', 'HGB', 'HGB'])))

def test_neural_entry_fit_requires_frozen_hgb_baseline_for_same_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    from v4.research import pathd_entry_models as models
    authorization = _dummy_fit_authorization()
    dataset = SimpleNamespace(dataset_sha256='8' * 64)
    monkeypatch.setattr(models, 'load_authorized_entry_dataset', lambda value: dataset)
    monkeypatch.setattr(models, 'assert_fit_authorization_current', lambda value: value)
    calls: list[str] = []
    monkeypatch.setattr(models, '_fit_neural_entry_bundle_impl', lambda auth, loaded, hgb: calls.append(hgb.artifact_sha256) or 'NEURAL_OK')
    hgb = models.seal_entry_model_bundle(family='HGB', authorization=authorization, dataset_sha256=dataset.dataset_sha256, payload={'weights_sha256': '9' * 64})
    models.validate_entry_model_bundle(hgb)
    assert models.fit_neural_entry_bundle(authorization, hgb_bundle=hgb) == 'NEURAL_OK'
    assert calls == [hgb.artifact_sha256]
    bad = (replace(hgb, family='NEURAL'), replace(hgb, fit_role='nested_weights'), replace(hgb, inner_fold=1), replace(hgb, authorization_sha256='0' * 64), replace(hgb, dataset_sha256='0' * 64), replace(hgb, outer_fold=2), replace(hgb, sessions_sha256_newline='0' * 64), replace(hgb, artifact_sha256='0' * 64))
    for bundle in bad:
        before = list(calls)
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.fit_neural_entry_bundle(authorization, hgb_bundle=bundle)
        assert calls == before
    with pytest.raises(TypeError):
        models.fit_neural_entry_bundle(authorization)

def test_entry_composer_scores_wait_and_all_42_actions_under_physical_mask() -> None:
    from v4.research import pathd_entry_models as models
    assert 'v4.research.pathd_entry_models.select_entry_action_index' in _api_names()
    spec = prereg.entry_composer_spec()
    assert len(spec['required_gate_components']['dollars']) == 10
    assert len(spec['required_gate_components']['returns']) == 10
    assert 'q10 as gate' in spec['forbidden']
    signature = inspect.signature(models.compose_entry_action)
    assert 'prediction' not in signature.parameters
    assert 'mask' not in signature.parameters
    components = np.arange(420, dtype=np.float64).reshape(42, 10)
    observed = _statistics_tuple(models.compose_enter_statistics(mean_dollars=components, mean_returns=components / 100.0, q10_dollars=components - 20.0, q10_returns=(components - 20.0) / 100.0, available=np.asarray([True, False, True, False, True])))
    selected_columns = [0, 1, 4, 5, 8, 9]
    expected = components[:, selected_columns].mean(axis=1)
    np.testing.assert_allclose(observed[0], expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed[1], expected / 100.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed[2], expected - 20.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed[3], (expected - 20.0) / 100.0, rtol=0.0, atol=0.0)
    gate_dollars = np.full(42, -1.0)
    gate_returns = np.full(42, -0.01)
    gate_q10_dollars = np.full(42, -100.0)
    gate_q10_returns = np.full(42, -1.0)
    gate_dollars[[0, 2, 3, 4, 5, 6, 7]] = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, np.nan]
    gate_returns[[1, 2, 3, 4, 5, 6, 7]] = [0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01]
    gate_q10_dollars[[0, 1, 2, 3, 4, 5, 6, 7]] = [1000.0, 900.0, 800.0, 700.0, 1.0, 2.0, 9999.0, 9998.0]
    gate_q10_returns[[0, 1, 2, 3, 4, 5, 6, 7]] = [10.0, 9.0, 8.0, 7.0, 0.01, 0.02, 99.0, 98.0]
    gate_physical = np.ones(42, dtype=np.bool_)
    gate_dynamic = np.ones(42, dtype=np.bool_)
    gate_physical[6] = False
    canonical_ties = tuple((abs(index), index, 'C', 20260801, index, f'G{index:02d}') for index in range(42))
    strict_gate = models.select_entry_action_index(
        mean_lcb_dollars=gate_dollars, mean_lcb_returns=gate_returns,
        q10_dollars=gate_q10_dollars, q10_returns=gate_q10_returns,
        physical_action_mask=gate_physical, dynamic_account_mask=gate_dynamic,
        tie_break_keys=canonical_ties,
    )
    assert strict_gate['selected_action_index'] == 5
    gate_dynamic[5] = False
    assert models.select_entry_action_index(
        mean_lcb_dollars=gate_dollars, mean_lcb_returns=gate_returns,
        q10_dollars=gate_q10_dollars, q10_returns=gate_q10_returns,
        physical_action_mask=gate_physical, dynamic_account_mask=gate_dynamic,
        tie_break_keys=canonical_ties,
    )['selected_action_index'] == 4
    gate_dynamic[4] = False
    assert models.select_entry_action_index(
        mean_lcb_dollars=gate_dollars, mean_lcb_returns=gate_returns,
        q10_dollars=gate_q10_dollars, q10_returns=gate_q10_returns,
        physical_action_mask=gate_physical, dynamic_account_mask=gate_dynamic,
        tie_break_keys=canonical_ties,
    )['action'] == 'WAIT'
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        models.select_entry_action_index(
            mean_lcb_dollars=gate_dollars[:-1], mean_lcb_returns=gate_returns,
            q10_dollars=gate_q10_dollars, q10_returns=gate_q10_returns,
            physical_action_mask=gate_physical, dynamic_account_mask=gate_dynamic,
            tie_break_keys=canonical_ties,
        )
    mean_dollars = np.full(42, -1.0)
    mean_returns = np.full(42, -0.01)
    q10_dollars = np.zeros(42)
    q10_returns = np.zeros(42)
    mean_dollars[[1, 7, 8, 40]] = [1.0, 2.0, 2.0, 3.0]
    mean_returns[[1, 7, 8, 40]] = [0.01, 0.02, 0.02, 0.03]
    q10_dollars[[1, 7, 8, 40]] = [2.0, 4.0, 4.0, 100.0]
    q10_returns[[1, 7, 8, 40]] = [0.2, 0.4, 0.4, 1.0]
    physical = np.ones(42, dtype=np.bool_)
    dynamic = np.ones(42, dtype=np.bool_)
    physical[40] = False
    tie_break_keys = tuple((99, index, 'P', 20260801, index, f'C{index:02d}') for index in range(42))
    tie_break_keys = tuple((0, 0, 'C', 20260801, 1, 'TIE_WIN') if index == 8 else key for index, key in enumerate(tie_break_keys))
    def select() -> dict[str, Any]:
        return models.select_entry_action_index(
            mean_lcb_dollars=mean_dollars, mean_lcb_returns=mean_returns,
            q10_dollars=q10_dollars, q10_returns=q10_returns,
            physical_action_mask=physical, dynamic_account_mask=dynamic,
            tie_break_keys=tie_break_keys,
        )
    q10_returns[7] = 0.5
    assert select()['selected_action_index'] == 7
    q10_returns[7] = 0.4
    selected = select()
    assert set(selected) == {'action', 'selected_action_index', 'combined_action_mask'}
    assert selected == {
        'action': 'ENTER', 'selected_action_index': 8,
        'combined_action_mask': (physical & dynamic).tolist(),
    }
    dynamic[8] = False
    assert select()['selected_action_index'] == 7
    dynamic[7] = False
    assert select()['selected_action_index'] == 1
    dynamic[:] = False
    assert select() == {'action': 'WAIT', 'selected_action_index': None, 'combined_action_mask': [False] * 42}
    _assert_prereg_mutations_rejected((lambda value: value['entry']['composer'].update(gate='q10 > 0'), lambda value: value['entry']['composer']['tie_break'].reverse()))

def test_entry_negative_controls_transform_complete_bundles_and_reject_tamper() -> None:
    from v4.research import pathd_entry_models as models
    assert 'v4.research.pathd_entry_models.transform_entry_negative_control_bundle' in _api_names()
    assert 'v4.research.pathd_entry_models.reverse_entry_calibrated_statistics' in _api_names()
    spec = prereg.entry_negative_control_spec()
    expected = ['CONSTANT', 'SIGN_REVERSED', 'TIME_SHIFTED_FEATURES', *(f'SHUFFLED_TARGET_{index:02d}' for index in range(1, 9))]
    assert spec['required_control_ids_in_order'] == expected
    assert spec['required_per_base_family'] == ['HGB', 'NEURAL']
    assert spec['strong_shuffled_targets']['all_eight_required'] is True
    assert 'complete entry target bundle' in spec['strong_shuffled_targets']['bundle']
    assert spec['sign_reversed']['recalibration_or_refit'] == 'none'
    identities = tuple(
        {'outer_fold': 1, 'session': '2026-01-02', 'decision_time_ns': index,
         'source_neutral_contract_id': f'C{index}', 'expiry_yyyymmdd': 20260102,
         'strike_milli_points': 6000000 + index, 'right_code': 'C' if index % 2 == 0 else 'P'}
        for index in range(4)
    )
    targets = tuple({'mean': [index, index + 10], 'q10': [-index, index + 20]} for index in range(4))
    validity = tuple({'mean': [True, index % 2 == 0], 'q10': [True, True]} for index in range(4))
    histories = tuple({'current': [index, index + 1], 'lag30': [index - 30, index - 29]} for index in range(4))
    seed = 10121001
    transformed = models.transform_entry_negative_control_bundle(
        control_id='SHUFFLED_TARGET_01', target_bundles=targets,
        target_validity=validity, feature_histories=histories,
        row_identities=identities, seed=seed,
    )
    assert set(transformed) == {
        'control_id', 'seed', 'row_identities', 'source_row_identities',
        'target_bundles', 'target_validity', 'feature_histories', 'transform_sha256',
        'transform_stage', 'refit_required',
    }
    assert transformed['transform_stage'] == 'TARGET_PRE_FIT'
    assert transformed['refit_required'] is True
    keyed = []
    for identity in identities:
        key = {'attempt_seed_id': seed, **identity}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        keyed.append((digest, identity))
    expected_sources = tuple(identity for _digest, identity in sorted(keyed))
    source_indexes = tuple(identities.index(identity) for identity in expected_sources)
    assert tuple(transformed['source_row_identities']) == expected_sources
    assert tuple(transformed['target_bundles']) == tuple(targets[index] for index in source_indexes)
    assert tuple(transformed['target_validity']) == tuple(validity[index] for index in source_indexes)
    assert tuple(transformed['feature_histories']) == histories
    semantic = dict(transformed); observed_hash = semantic.pop('transform_sha256')
    assert observed_hash == prereg.stable_hash(semantic)
    constant = models.transform_entry_negative_control_bundle(
        control_id='CONSTANT', target_bundles=targets, target_validity=validity,
        feature_histories=histories, row_identities=identities, seed=seed,
    )
    assert constant['transform_stage'] == 'PREDICTION_BASELINE'
    assert constant['refit_required'] is False
    assert tuple(constant['target_bundles']) == ({'mean': [1.5, 11.0], 'q10': [-3.0, 20.0]},) * 4
    shifted = models.transform_entry_negative_control_bundle(
        control_id='TIME_SHIFTED_FEATURES', target_bundles=targets,
        target_validity=validity, feature_histories=histories,
        row_identities=identities, seed=seed,
    )
    assert shifted['transform_stage'] == 'FEATURE_PRE_FIT'
    assert shifted['refit_required'] is True
    assert tuple(shifted['target_bundles']) == targets
    assert tuple(row['current'] for row in shifted['feature_histories']) == tuple(row['lag30'] for row in histories)
    reversed_control = models.transform_entry_negative_control_bundle(
        control_id='SIGN_REVERSED', target_bundles=targets,
        target_validity=validity, feature_histories=histories,
        row_identities=identities, seed=seed,
    )
    assert reversed_control['transform_stage'] == 'COMPOSER_POST_CALIBRATION'
    assert reversed_control['refit_required'] is False
    assert tuple(reversed_control['target_bundles']) == targets
    assert tuple(reversed_control['target_validity']) == validity
    calibrated = tuple(np.asarray(row, dtype=np.float64) for row in (
        [[1.0, -2.0], [3.0, -4.0]],
        [[0.1, -0.2], [0.3, -0.4]],
        [[-5.0, 6.0], [-7.0, 8.0]],
        [[-0.5, 0.6], [-0.7, 0.8]],
    ))
    reversed_statistics = _statistics_tuple(models.reverse_entry_calibrated_statistics(
        mean_lcb_dollars=calibrated[0], mean_lcb_returns=calibrated[1],
        q10_dollars=calibrated[2], q10_returns=calibrated[3],
    ))
    for observed, source in zip(reversed_statistics, calibrated, strict=True):
        np.testing.assert_array_equal(observed, -source)
    for source, expected_source in zip(calibrated, (
        [[1.0, -2.0], [3.0, -4.0]],
        [[0.1, -0.2], [0.3, -0.4]],
        [[-5.0, 6.0], [-7.0, 8.0]],
        [[-0.5, 0.6], [-0.7, 0.8]],
    ), strict=True):
        np.testing.assert_array_equal(source, np.asarray(expected_source, dtype=np.float64))
    for bad_targets, bad_validity, bad_identities in (
        (targets[:-1], validity, identities),
        (targets, validity[:-1], identities),
        (targets, validity, (*identities[:-1], identities[0])),
        (tuple({**row, 'mean': row['mean'][:-1]} for row in targets), validity, identities),
    ):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.transform_entry_negative_control_bundle(
                control_id='SHUFFLED_TARGET_01', target_bundles=bad_targets,
                target_validity=bad_validity, feature_histories=histories,
                row_identities=bad_identities, seed=seed,
            )
    for control_id in ('SHUFFLED_TARGET_09', 'UNKNOWN'):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.transform_entry_negative_control_bundle(
                control_id=control_id, target_bundles=targets,
                target_validity=validity, feature_histories=histories,
                row_identities=identities, seed=seed,
            )
    contract = prereg.entry_future_api_contract()['dataclass_fields']
    assert [field.name for field in fields(models.EntryNegativeControlManifestV1)] == contract['EntryNegativeControlManifestV1']
    invalid = _reseal_dataclass(models.EntryNegativeControlManifestV1(schema_version=models.EntryNegativeControlManifestV1.SCHEMA_VERSION, outer_fold=1, required_control_ids=tuple(expected[:-1]), required_base_families=('HGB', 'NEURAL'), control_bundle_sha256s=(), control_calibration_sha256s=(), control_composer_sha256s=(), sign_reversed_base_composer_sha256s=(), seed_receipt_sha256s=(), replay_config_sha256='1' * 64, artifact_sha256='0' * 64))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        models.validate_entry_negative_control_manifest(invalid)
    _assert_prereg_mutations_rejected((lambda value: value['entry']['controls']['negative_controls'].update(required_per_base_family=['HGB']), lambda value: value['entry']['controls']['negative_controls']['strong_shuffled_targets'].update(all_eight_required=False), lambda value: value['entry']['controls']['negative_controls']['sign_reversed'].update(recalibration_or_refit='refit')))

def test_matched_random_schedules_enforce_hash_quota_collision_and_no_redraw() -> None:
    from v4.research import pathd_entry_models as models
    assert 'v4.research.pathd_entry_models.build_matched_random_schedule' in _api_names()
    assert 'v4.research.pathd_entry_models.realize_frozen_matched_random_schedule' in _api_names()
    spec = prereg.entry_matched_random_spec()
    assert spec['attempt_seeds_in_order'] == list(prereg.MATCHED_RANDOM_SEEDS)
    assert spec['all_eight_required'] is True
    assert spec['aggregate'].startswith('after replay')
    assert {'redraw', 'replacement', 'best-seed selection'} <= set(spec['forbidden'])
    proposals = [
        {'id': 'A0', 'cell': 'A', 'session': '2026-01-02', 'decision_time_ns': 10, 'source_neutral_contract_id': 'A0_0', 'expiry_yyyymmdd': 20260102, 'strike_milli_points': 6000000, 'right_code': 'C'},
        {'id': 'B0_COLLIDES', 'cell': 'B', 'session': '2026-01-02', 'decision_time_ns': 10, 'source_neutral_contract_id': 'B0_COLLIDES_9', 'expiry_yyyymmdd': 20260102, 'strike_milli_points': 6000009, 'right_code': 'P'},
        {'id': 'B1', 'cell': 'B', 'session': '2026-01-02', 'decision_time_ns': 20, 'source_neutral_contract_id': 'B1_3', 'expiry_yyyymmdd': 20260102, 'strike_milli_points': 6000003, 'right_code': 'P'},
        {'id': 'A1_UNUSED', 'cell': 'A', 'session': '2026-01-02', 'decision_time_ns': 30, 'source_neutral_contract_id': 'A1_UNUSED_4', 'expiry_yyyymmdd': 20260102, 'strike_milli_points': 6000004, 'right_code': 'C'},
    ]
    assert _matched_random_key(seed=10121001, outer_fold=1, proposal=proposals[0]) == (13381638799184710972, 'b9b52196c1fd4d3c649ce69a26bdc86412861923f8855651ac0d5d0b0a2a0f1c')
    schedule = models.build_matched_random_schedule(
        proposals=proposals, quotas={'A': 1, 'B': 1}, seed=10121001,
        policy_id='MATCHED_RANDOM_01', outer_fold=1,
    )
    assert set(schedule) == {
        'policy_id', 'outer_fold', 'seed', 'ordered_ids',
        'eligible_population_sha256', 'matching_budget_sha256', 'schedule_sha256',
    }
    assert schedule['policy_id'] == 'MATCHED_RANDOM_01'
    assert schedule['outer_fold'] == 1 and schedule['seed'] == 10121001
    assert schedule['ordered_ids'] == ['A0', 'B1']
    assert schedule['eligible_population_sha256'] == prereg.stable_hash(proposals)
    assert schedule['matching_budget_sha256'] == prereg.stable_hash({'A': 1, 'B': 1})
    semantic = dict(schedule); observed_hash = semantic.pop('schedule_sha256')
    assert observed_hash == prereg.stable_hash(semantic)
    second = models.build_matched_random_schedule(
        proposals=proposals, quotas={'A': 1, 'B': 1}, seed=10121002,
        policy_id='MATCHED_RANDOM_02', outer_fold=1,
    )
    def independently_expected(seed: int) -> list[str]:
        rows = sorted(proposals, key=lambda row: (*_matched_random_key(seed=seed, outer_fold=1, proposal=row), row['cell'], row['decision_time_ns'], row['expiry_yyyymmdd'], row['strike_milli_points'], row['right_code'], row['source_neutral_contract_id']))
        remaining = {'A': 1, 'B': 1}
        used_minutes: set[int] = set()
        selected_ids: list[str] = []
        for row in rows:
            if remaining[row['cell']] > 0 and row['decision_time_ns'] not in used_minutes:
                selected_ids.append(row['id'])
                remaining[row['cell']] -= 1
                used_minutes.add(row['decision_time_ns'])
        return selected_ids
    assert schedule['ordered_ids'] == independently_expected(10121001)
    assert second['ordered_ids'] == independently_expected(10121002)
    realized = models.realize_frozen_matched_random_schedule(
        schedule,
        proposal_population=proposals,
        outcomes={'A0': 'NO_FILL', 'B0_COLLIDES': 'FILLED', 'B1': 'FILLED', 'A1_UNUSED': 'FILLED'},
    )
    assert realized == {
        'attempted_ids': ['A0', 'B1'],
        'filled_ids': ['B1'],
        'no_fill_ids': ['A0'],
        'structural_skip_ids': [],
        'substituted_or_backfilled_ids': [],
    }
    assert 'A1_UNUSED' not in realized['attempted_ids']
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        models.build_matched_random_schedule(
            proposals=[{**proposals[0], 'rank64': 0, 'digest': '00'}, *proposals[1:]],
            quotas={'A': 1, 'B': 1}, seed=10121001,
            policy_id='MATCHED_RANDOM_01', outer_fold=1,
        )
    for bad_seed, bad_policy in ((10121009, 'MATCHED_RANDOM_09'), (10121001, 'MATCHED_RANDOM_02')):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.build_matched_random_schedule(
                proposals=proposals, quotas={'A': 1, 'B': 1}, seed=bad_seed,
                policy_id=bad_policy, outer_fold=1,
            )
    owners = tuple(prereg.entry_matched_random_owner_policy_ids())
    valid = _reseal_dataclass(models.EntryControlReplayConfigV1(schema_version=models.EntryControlReplayConfigV1.SCHEMA_VERSION, outer_fold=1, fill_law_hash='1' * 64, control_exit_sha256='2' * 64, required_policy_ids=owners, matched_random_seeds=tuple(prereg.MATCHED_RANDOM_SEEDS), fee_paths=(3, 4), sell_delay_rungs_ms=(0, 1000, 2000, 5000), headline_config_sha256='3' * 64, artifact_sha256='0' * 64))
    models.validate_entry_control_replay_config(valid)
    invalid = _reseal_dataclass(replace(valid, matched_random_seeds=tuple(reversed(prereg.MATCHED_RANDOM_SEEDS))))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        models.validate_entry_control_replay_config(invalid)
    for changed in (replace(valid, required_policy_ids=()), replace(valid, required_policy_ids=owners[:-1]), replace(valid, fee_paths=(3,)), replace(valid, sell_delay_rungs_ms=(1000,))):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            models.validate_entry_control_replay_config(_reseal_dataclass(changed))
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        models.build_matched_random_schedule(
            proposals=proposals[:2], quotas={'A': 1, 'B': 1}, seed=10121001,
            policy_id='MATCHED_RANDOM_01', outer_fold=1,
        )

def test_calibration_statistics_fixed_vectors_cover_seed_lcb_q10_and_action_deciles() -> None:
    from v4.research import pathd_entry_models as models
    assert 'v4.research.pathd_entry_models.session_bootstrap_mean_lcb_correction' in _api_names()
    assert 'v4.research.pathd_entry_models.reconstruct_entry_action_calibration_gate_inputs' in _api_names()
    mean, std = models.session_balanced_population_mean_std(np.asarray([1.0, 3.0, 10.0]), sessions=('A', 'A', 'B'), valid=np.asarray([True, True, True]))
    assert mean == 6.0
    assert std == math.sqrt(16.5)
    assert models.weighted_lower_conformal_correction(np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 3.0, 10.0]), sessions=('A', 'A', 'B'), identities=('A0', 'A1', 'B0'), alpha=0.1) == 1.0
    assert models.wait_raw_lower_statistic((-1.0, 2.0, 0.5)) == 2.0
    digest, seed = _seed_record(purpose='mean_lcb_bootstrap', model_family='HGB', fold_scope='OUTER_1', statistic='HEAD::h10_mfe_dollars')
    assert digest == '11a772ddc7ad411fd75d9ba5f09db2a8e94cc035d7ac6e292b0860522947f6a8'
    assert seed == 296186589
    residuals = np.asarray([1.0, 1.0, 3.0, 3.0], dtype=np.float64)
    residual_sessions = ('A', 'A', 'B', 'B')
    generator = np.random.Generator(np.random.PCG64(seed))
    session_means = np.asarray([1.0, 3.0], dtype=np.float64)
    bootstrap_means = session_means[generator.integers(0, 2, size=(2000, 2))].mean(axis=1)
    nearest_rank_index = min(2000, max(1, math.ceil((2000 + 1) * 0.10))) - 1
    expected_correction = float(np.sort(bootstrap_means, kind='stable')[nearest_rank_index])
    assert models.session_bootstrap_mean_lcb_correction(
        residuals, sessions=residual_sessions, seed=seed, resamples=2000,
    ) == expected_correction
    action_digest, action_seed = _seed_record(purpose='action_decile_bootstrap', model_family='GLOBAL_HGB_ENTRY', fold_scope='POOLED_OUTER_1_5', statistic='ACTION::ENTER')
    assert action_digest == 'b0c7e4869778d82ffc691448c6e01de4e2ded925a3de7516da8c7d350f709826'
    assert action_seed == 2965890182
    evidence_sessions = ('A', 'B', 'C')
    observations_list = []
    for decile in range(10):
        for session_index, session in enumerate(evidence_sessions):
            semantic = {
                'schema_version': models.EntryActionCalibrationObservationV1.SCHEMA_VERSION,
                'outer_fold': 1, 'action': 'ENTER', 'session': session,
                'trajectory_or_episode_id': f'E{decile:02d}:{session}',
                'predicted_mean_micros': decile * 100 + session_index,
                'predicted_lower_micros': decile * 10 - 1,
                'realized_micros': decile * 10,
                'source_policy_evaluation_sha256': '1' * 64,
                'source_transition_sha256': hashlib.sha256(f'{decile}:{session}'.encode()).hexdigest(),
            }
            observations_list.append(models.EntryActionCalibrationObservationV1(
                **semantic, observation_sha256=prereg.stable_hash(semantic)
            ))
    observations = tuple(observations_list)
    gate_inputs = models.reconstruct_entry_action_calibration_gate_inputs(
        observations, action='ENTER', evidence_sessions=evidence_sessions,
        seed=action_seed,
    )
    assert gate_inputs == {
        'distinct_trajectory_count': 30,
        'coverage_numerator': 30,
        'coverage_denominator': 30,
        'decile_trajectory_counts': [3] * 10,
        'decile_distinct_session_counts': [3] * 10,
        'adjacent_valid_bootstrap_replicates': [5000] * 9,
        'adjacent_total_draws': [5000] * 9,
        'adjacent_upper_bounds_micros': [-10.0] * 9,
    }
    underpowered = models.reconstruct_entry_action_calibration_gate_inputs(
        observations[:-1], action='ENTER', evidence_sessions=evidence_sessions,
        seed=action_seed,
    )
    assert underpowered['distinct_trajectory_count'] == 29
    assert min(underpowered['decile_trajectory_counts']) == 2
    assert min(underpowered['decile_distinct_session_counts']) == 2
    action = prereg.action_calibration_spec()
    assert action['minimum_distinct_trajectories_per_action'] == 30
    assert action['minimum_empirical_coverage'] == 0.85
    assert action['deciles']['minimum_per_decile'] == 3
    assert action['deciles']['minimum_distinct_sessions_per_decile'] == 3
    assert '5,000 valid' in action['deciles']['bootstrap']
    _assert_prereg_mutations_rejected((lambda value: value['calibration_and_statistics']['mean_lcb'].update(level=0.95), lambda value: value['metrics_and_gates']['action_calibration']['deciles'].update(minimum_per_decile=2)))

def test_nested_and_outer_evidence_scopes_open_and_seal_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v4.research import pathd_evidence_gate as gate
    from v4.scripts import run_pathd_entry_exit_research as runner
    roles = prereg.evidence_session_role_spec()
    assert tuple(roles) == ('nested_validation', 'outer_test_primary', 'outer_test_shortened_diagnostic')
    assignments = prereg.session_assignments()
    inner = assignments['folds'][0]['inner_forward_folds']['scored_forward_folds'][0]
    if inner['calibration_valid']:
        session = inner['validation'][0]
        assert runner.entry_evidence_role_for_session(assignments=assignments, outer_fold=1, inner_fold=1, session=session) == 'nested_validation'
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(gate, 'REPO_ROOT', root)
    monkeypatch.setattr(gate, 'ENTRY_FOLD_ARTIFACT_ROOT', root / 'folds')
    monkeypatch.setattr(gate, '_validate_claim_foundation', lambda *_a, **_k: None)
    with pytest.raises(gate.EntryEvidenceGateError):
        gate._assert_scope_order(gate._scope_paths(role='nested_validation', outer_fold=1, inner_fold=2))
    with pytest.raises(gate.EntryEvidenceGateError):
        gate._assert_scope_order(gate._scope_paths(role='outer_test_primary', outer_fold=1, inner_fold=None))
    sessions = ('2026-01-02',)
    claim = {'role': 'nested_validation', 'outer_fold': 1, 'inner_fold': 1, 'sessions': sessions, 'sessions_sha256_newline': prereg.canonical_session_hash(sessions), 'preregistration_sha256': '1' * 64, 'session_assignments_sha256': '2' * 64, 'source_hash_policy_sha256': '3' * 64, 'corpus_integrity_receipt_sha256': '4' * 64, 'lineage_receipt_sha256': '5' * 64, 'machinery_receipt_sha256': '6' * 64, 'fit_environment_sha256': '7' * 64, 'open_gate_receipts_sha256': ('8' * 64,)}
    monkeypatch.setattr(gate, '_prepare_entry_evidence_authorization_claim', lambda **_kw: claim)
    scope = 'NESTED_OUTER_1_INNER_1'
    nodes = prereg.entry_required_calibration_node_ids(scope)
    gate_semantic = {
        'schema_version': 'pathd.calibration_scope_gate_receipt.v1',
        'scope': scope,
        'status': 'VALID',
        'required_node_count': len(nodes),
        'required_node_ids_sha256': prereg.stable_hash(list(nodes)),
        'ordered_node_sha256s': [f'{index + 1:064x}' for index in range(len(nodes))],
        'node_vector_sha256': 'f' * 64,
        'failure_node_ids': [],
        'evidence_access_count': 0,
        'holdout_open_count': 0,
        'forbidden_rescue_applied': False,
    }
    gate_path = gate._calibration_scope_gate_path(
        gate._scope_paths(role='nested_validation', outer_fold=1, inner_fold=1)
    )
    gate._write_exclusive_json(
        gate_path,
        {**gate_semantic, 'receipt_sha256': prereg.stable_hash(gate_semantic)},
    )
    authorization = gate.begin_entry_evidence_once(role='nested_validation', outer_fold=1, inner_fold=1)
    assert gate.inspect_entry_evidence_state(role='nested_validation', outer_fold=1, inner_fold=1)['state'] == gate.OPEN_ACTIVE
    monkeypatch.setattr(gate, 'read_frozen_entry_evidence_authorization', lambda **_kwargs: authorization)
    assert gate.claim_entry_evidence_decode_once(authorization) is authorization
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.claim_entry_evidence_decode_once(authorization)
    gate._release_capability(id(authorization))
    burned = gate.recover_entry_evidence_after_crash(role='nested_validation', outer_fold=1, inner_fold=1)
    assert burned['state'] == gate.BURNED and burned['access_count'] == 1
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.begin_entry_evidence_once(role='nested_validation', outer_fold=1, inner_fold=1)
    source = inspect.getsource(gate.begin_entry_evidence_once)
    assert all((token in source for token in ('_assert_unopened', '_assert_scope_order', '_write_exclusive_json')))
    seal_source = inspect.getsource(gate.seal_entry_evidence_result)
    assert '_write_exclusive_json' in seal_source
    assert '_release_capability' in seal_source
    assert list(inspect.signature(gate.begin_entry_evidence_once).parameters) == ['role', 'outer_fold', 'inner_fold']
    assert list(inspect.signature(gate.claim_entry_evidence_decode_once).parameters) == ['authorization']
    assert list(inspect.signature(gate.seal_entry_evidence_result).parameters) == ['authorization', 'dataset', 'evaluation']

def test_nested_and_outer_results_bind_dataset_example_membership_authorization_and_ledger() -> None:
    from v4.scripts import run_pathd_entry_exit_research as runner
    assert 'v4.scripts.run_pathd_entry_exit_research.validate_entry_outer_selection_binding' in _api_names()
    ordered = ['a' * 64, 'b' * 64, 'c' * 64]
    evaluation, receipt = _coverage_membership_fixture(ordered, ['ACTION_DECISION', 'STATE_INELIGIBLE', 'ACTION_DECISION'])
    prereg._validate_policy_evaluation_dataset_membership(evaluation, receipt)
    for changed in ({**receipt, 'ordered_example_hashes': ordered[:-1]}, {**receipt, 'ordered_example_hashes': list(reversed(ordered))}):
        with pytest.raises(RuntimeError):
            prereg._validate_policy_evaluation_dataset_membership(evaluation, changed)
    for validator in (runner.validate_entry_nested_family_evaluation, runner.validate_entry_outer_primary_result):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            validator({}, authorization=object(), dataset=object())
    outer_semantic = {
        'schema_version': 'pathd.entry_outer_primary_result.v1',
        'outer_fold': 1, 'evidence_role': 'outer_test_primary',
        'authorization_sha256': '1' * 64, 'dataset_sha256': '2' * 64,
        'source_receipts_root_sha256': '3' * 64, 'access_receipt_sha256': '4' * 64,
        'hgb': {'family': 'HGB', 'result_sha256': '5' * 64},
        'neural': {'family': 'NEURAL', 'result_sha256': '6' * 64},
        'selected_family': 'HGB', 'selected_evaluation_sha256': '5' * 64,
        'negative_control_panel_sha256': '7' * 64,
        'control_replay_result_sha256': '8' * 64,
        'session_coverage_root_sha256': '9' * 64,
    }
    valid_outer = {**outer_semantic, 'result_sha256': prereg.stable_hash(outer_semantic)}
    binding = {
        'authorization_sha256': '1' * 64,
        'dataset_sha256': '2' * 64,
        'source_receipts_root_sha256': '3' * 64,
        'access_receipt_sha256': '4' * 64,
    }
    assert list(inspect.signature(runner.validate_entry_outer_selection_binding).parameters) == ['result', 'authorization_sha256', 'dataset_sha256', 'source_receipts_root_sha256', 'access_receipt_sha256']
    assert runner.validate_entry_outer_selection_binding(valid_outer, **binding) == valid_outer
    for mutate in (
        lambda row: row.update(selected_family='NEURAL', selected_evaluation_sha256='6' * 64),
        lambda row: row.update(selected_evaluation_sha256='6' * 64),
        lambda row: row['hgb'].update(family='NEURAL'),
        lambda row: row.update(authorization_sha256='0' * 64),
        lambda row: row.update(dataset_sha256='0' * 64),
        lambda row: row.update(source_receipts_root_sha256='0' * 64),
        lambda row: row.update(access_receipt_sha256='0' * 64),
        lambda row: row.update(result_sha256='0' * 64),
    ):
        changed = copy.deepcopy(valid_outer)
        mutate(changed)
        if changed['result_sha256'] != '0' * 64:
            semantic = dict(changed); semantic.pop('result_sha256')
            changed['result_sha256'] = prereg.stable_hash(semantic)
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            runner.validate_entry_outer_selection_binding(changed, **binding)
    nested = inspect.getsource(runner.validate_entry_nested_family_evaluation)
    outer = inspect.getsource(runner.validate_entry_outer_primary_result)
    required_tokens = ('authorization_sha256', 'dataset_sha256', 'source_receipts_root_sha256', 'access_receipt_sha256', 'validate_entry_policy_evaluation')
    for token in required_tokens:
        assert token in nested
    for token in ('authorization_sha256', 'dataset_sha256', 'session_coverage_root_sha256', 'validate_entry_outer_selection_binding'):
        assert token in outer
    fields_by_name = prereg.entry_future_api_contract()['dataclass_fields']
    assert fields_by_name['EntryNestedFamilyEvaluationV1'][-1] == 'result_sha256'
    assert fields_by_name['EntryOuterPrimaryResultV1'][-1] == 'result_sha256'
    for forbidden in ('caller_sessions', 'caller_pnl', 'caller_ledger'):
        assert forbidden not in nested + outer

def test_frame_coverage_visits_every_ordered_example_and_rejects_omitted_eligible_frame() -> None:
    from v4.path_d.execution import research_replay as replay
    contract = prereg.entry_dataset_seal_spec()
    assert contract['example_identity'].startswith('(model_input.session')
    assert 'strictly sort' in contract['example_identity']
    assert 'ordered_example_hashes' in contract['dataset_digest_object_fields']
    ordered = ['a' * 64, 'b' * 64, 'c' * 64]
    complete, receipt = _coverage_membership_fixture(ordered, ['ACTION_DECISION', 'STATE_INELIGIBLE', 'ACTION_DECISION'])
    prereg._validate_policy_evaluation_dataset_membership(complete, receipt)
    transitions = complete['terminal_journal']['transitions']
    first_decision = next(index for index, row in enumerate(transitions) if row['event_schema_version'] == 'pathd.entry_action_decision.v1')
    first_observation = next(index for index, row in enumerate(transitions) if row['event_schema_version'] == 'pathd.entry_action_observation.v1')
    ineligible_coverage = next(index for index, row in enumerate(transitions) if row['event_schema_version'] == 'pathd.entry_frame_coverage.v1' and row['event_payload']['example_sha256'] == ordered[1])
    action_on_ineligible = copy.deepcopy(transitions)
    action_on_ineligible.insert(ineligible_coverage + 1, {'event_schema_version': 'pathd.entry_action_decision.v1', 'event_payload': {'example_sha256': ordered[1]}})
    for changed in (
        transitions[1:],
        [transitions[1], transitions[0], *transitions[2:]],
        [copy.deepcopy(transitions[0]), *copy.deepcopy(transitions)],
        [row for index, row in enumerate(transitions) if index != first_decision],
        [row for index, row in enumerate(transitions) if index != first_observation],
        [*copy.deepcopy(transitions[:first_observation + 1]), copy.deepcopy(transitions[first_observation]), *copy.deepcopy(transitions[first_observation + 1:])],
        action_on_ineligible,
    ):
        forged = {'terminal_journal': {'transitions': changed}}
        with pytest.raises(RuntimeError):
            prereg._validate_policy_evaluation_dataset_membership(forged, receipt)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        replay.entry_frame_coverage_from_verified_example(object(), dataset=object(), authorization=object(), journal=object())
    source = inspect.getsource(replay.validate_research_ledger_journal_against_dataset)
    assert 'EntryFrameCoverageV1' in source
    assert 'examples' in source
    assert 'ACTION_DECISION' in source
    assert 'STATE_INELIGIBLE' in source

def test_entry_policy_evaluation_reconstructs_nonzero_economics_from_full_journal() -> None:
    from v4.path_d.execution import research_replay as replay
    from v4.scripts import run_pathd_entry_exit_research as runner
    assert 'v4.path_d.execution.research_replay.reconstruct_entry_policy_economics_from_validated_transitions' in _api_names()
    events = _journal_events(session='2026-01-02', policy_id='HGB', pnl_micros=87000000, trade_count=1)
    result = replay.reconstruct_entry_policy_economics_from_validated_transitions(
        transitions=events, starting_cash_micros=10000000000
    )
    assert set(result) == {'session_pnl_micros', 'completed_trade_ids', 'terminal_cash_micros'}
    assert result == {'session_pnl_micros': 87000000, 'completed_trade_ids': ['HGB:2026-01-02:0'], 'terminal_cash_micros': 10087000000}
    assert _reconstruct_fixture_journal({'transitions': events}) == (87000000, ['HGB:2026-01-02:0'], 10087000000)
    for index, field, delta in ((0, 'cash_after_micros', 1), (1, 'cash_before_micros', 1), (1, 'cash_delta_micros', 1)):
        changed = copy.deepcopy(events)
        changed[index][field] += delta
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.reconstruct_entry_policy_economics_from_validated_transitions(
                transitions=changed, starting_cash_micros=10000000000
            )
    semantic_tampers = []
    changed_kind = copy.deepcopy(events); changed_kind[0]['kind'] = 'SELL_FILL'; semantic_tampers.append(changed_kind)
    changed_trade = copy.deepcopy(events); changed_trade[1]['trade_id'] = 'OTHER'; semantic_tampers.append(changed_trade)
    changed_position = copy.deepcopy(events); changed_position[0]['position_after'] = 0; semantic_tampers.append(changed_position)
    semantic_tampers.extend((events[:1], tuple(reversed(events))))
    for changed in semantic_tampers:
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            replay.reconstruct_entry_policy_economics_from_validated_transitions(
                transitions=changed, starting_cash_micros=10000000000
            )
    assert 'terminal_journal' in inspect.getsource(replay.reconstruct_entry_policy_evaluation_from_journal)
    assert 'reconstruct_entry_policy_evaluation_from_journal' in inspect.getsource(runner.validate_entry_policy_evaluation)

def test_fixed_holdout_trace_reconstructs_synthetic_30_session_result() -> None:
    from v4.research import pathd_holdout_gate as holdout
    from v4.scripts import run_pathd_entry_exit_research as runner
    assert 'v4.scripts.run_pathd_entry_exit_research.reconstruct_protected_holdout_payload_from_validated_trace' in _api_names()
    assignments = prereg.session_assignments()
    sessions = tuple(assignments['protected_holdout_30'])
    assert len(sessions) == 30 and sessions[-1] == '2026-07-31'
    primary = tuple((session for session in sessions if session != '2026-07-31'))
    assert len(primary) == 29
    authorization_sha256, dataset_sha256 = '1' * 64, '2' * 64
    records = []
    for session in sessions:
        records.append(runner.ProtectedHoldoutTraceRecordV1(**_holdout_trace_dict(
            session=session,
            schema_version=runner.ProtectedHoldoutTraceRecordV1.SCHEMA_VERSION,
            guard_keys=holdout.HOLDOUT_GUARD_KEYS,
            authorization_sha256=authorization_sha256,
            dataset_sha256=dataset_sha256,
        )))
    validated = runner.validate_protected_holdout_trace_records(
        records=tuple(records), authorization_sha256=authorization_sha256,
        dataset_sha256=dataset_sha256, sessions=sessions,
        box_d_policy_id='BOX_D', comparator_policy_id='COMPARATOR',
    )
    assert tuple(validated) == tuple(records)
    payload = runner.reconstruct_protected_holdout_payload_from_validated_trace(
        records=tuple(validated), primary_sessions=primary,
        degradation_session='2026-07-31',
    )
    assert set(payload) == {'schema_version', 'verdict', 'protected_session_count', 'primary_non_degraded_session_count', 'completed_box_d_trades_on_primary_29', 'box_d_net_pnl_micros_on_primary_29', 'box_d_minus_comparator_paired_net_pnl_micros_on_primary_29', 'guards', 'survival_violation_counts', 'owner_facing_metrics', 'degraded_sensitivity', 'pass_criteria_recomputed'}
    assert payload['protected_session_count'] == 30
    assert payload['primary_non_degraded_session_count'] == 29
    assert payload['completed_box_d_trades_on_primary_29'] == 58
    assert payload['box_d_net_pnl_micros_on_primary_29'] == 2900
    assert payload['box_d_minus_comparator_paired_net_pnl_micros_on_primary_29'] == 1450
    assert payload['verdict'] == 'PASS'
    assert payload['degraded_sensitivity']['session'] == '2026-07-31'
    for wrong_primary, wrong_degraded in (
        (tuple(reversed(primary)), '2026-07-31'),
        (primary[:-1], '2026-07-31'),
        (primary, primary[-1]),
    ):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            runner.reconstruct_protected_holdout_payload_from_validated_trace(
                records=tuple(validated), primary_sessions=wrong_primary,
                degradation_session=wrong_degraded,
            )
    signature = inspect.signature(runner.reconstruct_protected_holdout_evaluation_from_trace)
    assert list(signature.parameters) == ['authorization', 'dataset']
    assert 'trace' not in signature.parameters

def test_fixed_holdout_trace_rejects_all_payload_identity_partition_and_hash_tamper() -> None:
    from v4.research import pathd_holdout_gate as holdout
    from v4.scripts import run_pathd_entry_exit_research as runner
    assert 'v4.scripts.run_pathd_entry_exit_research.validate_protected_holdout_trace_records' in _api_names()
    sessions = tuple(prereg.session_assignments()['protected_holdout_30'])
    expected = {'sessions': sessions, 'authorization_sha256': '1' * 64, 'dataset_sha256': '2' * 64, 'box_d_policy_id': 'BOX_D', 'comparator_policy_id': 'COMPARATOR', 'guard_keys': holdout.HOLDOUT_GUARD_KEYS}
    record_dicts = tuple(_holdout_trace_dict(
        session=session,
        schema_version=runner.ProtectedHoldoutTraceRecordV1.SCHEMA_VERSION,
        guard_keys=holdout.HOLDOUT_GUARD_KEYS,
        authorization_sha256=expected['authorization_sha256'],
        dataset_sha256=expected['dataset_sha256'],
        box_trade_count=1,
    ) for session in sessions)
    base = record_dicts[0]
    _validate_trace_record(base, expected)
    binding = {'authorization_sha256': expected['authorization_sha256'], 'dataset_sha256': expected['dataset_sha256'], 'sessions': sessions, 'box_d_policy_id': expected['box_d_policy_id'], 'comparator_policy_id': expected['comparator_policy_id']}
    typed_records = tuple(runner.ProtectedHoldoutTraceRecordV1(**row) for row in record_dicts)
    assert tuple(runner.validate_protected_holdout_trace_records(records=typed_records, **binding)) == typed_records
    for reordered in (typed_records[:-1], (*typed_records, typed_records[-1]), tuple(reversed(typed_records))):
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            runner.validate_protected_holdout_trace_records(records=reordered, **binding)
    semantic_mutations = (
        lambda row: row.update(session='1999-01-01'),
        lambda row: row.update(authorization_sha256='0' * 64),
        lambda row: row.update(dataset_sha256='0' * 64),
        lambda row: row.update(box_d_policy_id='OTHER'),
        lambda row: row.update(comparator_policy_id='OTHER'),
        lambda row: row['box_d_terminal_journal'].update(session_pnl_micros=101),
        lambda row: row['comparator_terminal_journal'].update(session_pnl_micros=51),
        lambda row: row.update(box_d_session_pnl_micros=101),
        lambda row: row.update(comparator_session_pnl_micros=51),
        lambda row: row.update(box_d_completed_trade_ids=['FORGED']),
        lambda row: row.update(guard_results={name: True for name in holdout.HOLDOUT_GUARD_KEYS[:-1]}),
        lambda row: row.update(guard_results={**{name: True for name in holdout.HOLDOUT_GUARD_KEYS}, 'EXTRA': True}),
        lambda row: row['guard_results'].update({holdout.HOLDOUT_GUARD_KEYS[0]: False}),
        lambda row: row.update(survival_violations=['FORGED']),
        lambda row: row['box_d_terminal_journal']['transitions'][0].update(cash_after_micros=row['box_d_terminal_journal']['transitions'][0]['cash_after_micros'] + 1),
        lambda row: row['box_d_terminal_journal']['transitions'][1].update(trade_id='FORGED'),
        lambda row: row['box_d_terminal_journal'].update(transitions=row['box_d_terminal_journal']['transitions'][:-1]),
        lambda row: row['comparator_terminal_journal'].update(transitions=tuple(reversed(row['comparator_terminal_journal']['transitions']))),
    )
    for mutate in semantic_mutations:
        changed = copy.deepcopy(base)
        mutate(changed)
        _seal_trace_record(changed)
        with pytest.raises(ValueError):
            _validate_trace_record(changed, expected)
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            runner.validate_protected_holdout_trace_records(records=(runner.ProtectedHoldoutTraceRecordV1(**changed), *typed_records[1:]), **binding)
    changed = copy.deepcopy(base)
    changed['record_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        _validate_trace_record(changed, expected)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        runner.validate_protected_holdout_trace_records(records=(runner.ProtectedHoldoutTraceRecordV1(**changed), *typed_records[1:]), **binding)

def test_fixed_holdout_evaluator_rejects_caller_authored_summary() -> None:
    from v4.research import pathd_holdout_gate as holdout
    from v4.scripts import run_pathd_entry_exit_research as runner
    evaluator = runner.run_fixed_protected_holdout_evaluator
    assert list(inspect.signature(evaluator).parameters) == ['authorization', 'dataset']
    with pytest.raises(TypeError):
        evaluator(object(), object(), {'status': 'PASS'})
    with pytest.raises(TypeError):
        holdout.ActiveProtectedHoldoutAuthorizationV1(object(), transaction_token_sha256='0' * 64, lock_fd=-1, semantic={})
    source = inspect.getsource(evaluator)
    for forbidden in ('caller_summary', 'caller_verdict', 'declared_pass'):
        assert forbidden not in source
    for required in ('validate_protected_holdout_trace', 'reconstruct_protected_holdout_evaluation_from_trace'):
        assert required in source
    assert "'PASS'" not in source and '"PASS"' not in source
    assert 'execute_protected_holdout_once' in holdout.__all__
    assert 'load_authorized_protected_holdout' not in holdout.__all__
    assert 'seal_protected_holdout_result' not in holdout.__all__

def test_preholdout_packet_runs_all_six_semantic_validators_and_rejects_declared_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v4.research import pathd_holdout_gate as holdout
    names = ('outer_entry_exit_artifacts', 'four_box_packet', 'guard_panel', 'outer_acceptance_packet', 'full_fit_entry_artifacts', 'full_fit_exit_artifacts')
    registered = holdout._registered_prepacket_artifact_validators()
    assert tuple(registered) == names
    assert tuple(value.__name__ for value in registered.values()) == (
        'validate_outer_entry_exit_artifacts_for_holdout',
        'validate_four_box_packet_for_holdout',
        'validate_guard_panel_for_holdout',
        'validate_outer_acceptance_packet_for_holdout',
        'validate_full_fit_entry_artifacts_for_holdout',
        'validate_full_fit_exit_artifacts_for_holdout',
    )
    statuses = {'outer_entry_exit_artifacts': 'FROZEN_COMPLETE', 'four_box_packet': 'FROZEN_COMPLETE', 'guard_panel': 'PASS', 'outer_acceptance_packet': 'PASS', 'full_fit_entry_artifacts': 'FROZEN_COMPLETE', 'full_fit_exit_artifacts': 'FROZEN_COMPLETE'}
    paths = {name: tmp_path / f'{name}.json' for name in names}
    calls: list[str] = []
    box_d, comparator = ('BOX_D', 'COMPARATOR')
    validators: dict[str, Callable[[Path], dict[str, Any]]] = {}
    packet: dict[str, Any] = {'selected_box_d_policy_id': box_d, 'selected_comparator_policy_id': comparator}
    monkeypatch.setattr(holdout, 'AUDIT_ROOT', tmp_path)
    monkeypatch.setattr(holdout, '_path_label', lambda path: str(path))
    for name in names:
        paths[name].write_text('{}\n', encoding='utf-8')
        digest = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        packet[name] = {'path': str(paths[name]), 'sha256': digest}

        def validator(path: Path, *, artifact_name: str=name, sha: str=digest) -> dict[str, Any]:
            calls.append(artifact_name)
            row = {'artifact_kind': artifact_name, 'status': statuses[artifact_name], 'artifact_sha256': sha}
            if artifact_name == 'four_box_packet':
                row['selected_box_d_policy_id'] = box_d
            if artifact_name == 'outer_acceptance_packet':
                row['selected_box_d_policy_id'] = box_d
                row['selected_comparator_policy_id'] = comparator
            return row
        validators[name] = validator
    monkeypatch.setattr(holdout, '_fixed_named_artifact_paths', lambda: paths)
    monkeypatch.setattr(holdout, '_registered_prepacket_artifact_validators', lambda: validators)
    rows, records = holdout._validate_named_prepacket_artifacts(packet)
    assert calls == list(names)
    assert len(rows) == len(records) == 6
    forged = dict(validators)
    forged['guard_panel'] = lambda path: {'artifact_kind': 'guard_panel', 'status': 'DECLARED_PASS', 'artifact_sha256': packet['guard_panel']['sha256']}
    monkeypatch.setattr(holdout, '_registered_prepacket_artifact_validators', lambda: forged)
    with pytest.raises(holdout.ProtectedHoldoutError):
        holdout._validate_named_prepacket_artifacts(packet)
    mismatched = dict(validators)
    mismatched['four_box_packet'] = lambda path: {'artifact_kind': 'four_box_packet', 'status': 'FROZEN_COMPLETE', 'artifact_sha256': packet['four_box_packet']['sha256'], 'selected_box_d_policy_id': 'OTHER'}
    monkeypatch.setattr(holdout, '_registered_prepacket_artifact_validators', lambda: mismatched)
    with pytest.raises(holdout.ProtectedHoldoutError):
        holdout._validate_named_prepacket_artifacts(packet)

def test_aref_decision_critical_support_closure_and_gradient_boundaries() -> None:
    topology = prereg.aref_decision_critical_topology_spec()
    spec = prereg.exit_diagnostic_model_isolation_spec()
    action = spec['action_model']
    assert action['targets'] == ['A_ref_mean', 'A_ref_q10']
    assert action['action_consumers'] == ['A_ref_mean', 'A_ref_q10']
    assert topology['direct_action_inputs'] == ['A_ref_mean', 'A_ref_q10']
    assert topology['decision_critical_calibration_support'] == ['A_ref_q10', 'A_ref_q50', 'A_ref_q90']
    assert topology['complete_required_aref_outputs'] == ['A_ref_mean', 'A_ref_q10', 'A_ref_q50', 'A_ref_q90']
    assert topology['diagnostic_only_families'] == ['downside_300', 'recovery_300', 'giveback_300', 'remaining_tail_300']
    assert not set(topology['complete_required_aref_outputs']) & set(topology['diagnostic_only_families'])
    assert action['shared_parameters_with_diagnostics'] is False
    assert action['shared_preprocessor_or_target_scaler_with_diagnostics'] is False
    assert action['shared_optimizer_or_gradient_graph_with_diagnostics'] is False
    assert spec['diagnostic_family_count'] == 4
    assert tuple(spec['diagnostic_models']) == ('downside_300', 'recovery_300', 'giveback_300', 'remaining_tail_300')
    support = spec['aref_decision_critical_support']
    assert support['targets'] == ['A_ref_q10', 'A_ref_q50', 'A_ref_q90']
    assert support['q50_support_can_backpropagate_into_action'] is False
    assert support['q90_support_can_backpropagate_into_action_or_q50'] is False
    assert support['q50_calibration_edge_is_binding'] is True
    assert support['q90_exit_reliability_edge_is_binding'] is True
    assert support['missing_or_nonfinite_effect'] == 'invalid_result'
    assert all(row['separate_model_bundle'] and row['separate_parameters'] and row['separate_feature_preprocessor'] and row['separate_target_scaler'] and row['separate_calibration'] and row['separate_optimizer_and_gradient_graph'] for row in spec['diagnostic_models'].values())
    payload = prereg.preregistration_payload()[0]
    targets = payload['exit']['distributional_targets']
    assert targets['A_ref_action_inputs'] == ['mean', 'q10']
    assert targets['A_ref_decision_critical_calibration_support'] == ['q10', 'q50', 'q90']
    assert targets['A_ref_complete_required_outputs'] == ['mean', 'q10', 'q50', 'q90']
    assert 'A_ref_diagnostic_only' not in targets
    power = payload['exit']['power_count_semantics']
    assert power['minimum_model_fit_sessions'] == 60
    assert power['known_exit_weight_session_counts_by_fold'] == [0, 0, 19, 48, 56]
    assert max(power['known_exit_weight_session_counts_by_fold']) < power['minimum_model_fit_sessions']
    assert power['consequence'].startswith('if entry clears, exit fitting must stop insufficient_evidence before weights')
    forbidden_current_run_apis = tuple(name for name in _api_names() if '.pathd_exit_models.' in name or name.rsplit('.', 1)[-1].startswith(('fit_exit', 'build_exit_action', 'build_exit_diagnostic')))
    assert forbidden_current_run_apis == ()
    _assert_prereg_mutations_rejected((
        lambda value: value['exit']['diagnostic_target_isolation']['action_model'].update(shared_parameters_with_diagnostics=True),
        lambda value: value['exit']['diagnostic_target_isolation']['diagnostic_models'].update(A_ref_q90={}),
        lambda value: value['exit']['distributional_targets']['A_ref_decision_critical_calibration_support'].remove('q50'),
        lambda value: value['exit']['composer']['action_inputs'].append('A_ref_q90'),
        lambda value: value['exit']['power_count_semantics'].update(minimum_model_fit_sessions=50),
    ))

def test_aref_hgb_and_neural_topologies_are_exact() -> None:
    topology = prereg.aref_decision_critical_topology_spec()
    hgb = topology['hgb']
    assert hgb['ensemble_seeds'] == [301, 302, 303]
    assert hgb['estimator_fits_per_seed'] == 16
    assert hgb['estimator_fits_total'] == 48
    assert len(hgb['decision_critical_estimators_per_seed']) == 4
    assert len(hgb['diagnostic_estimators_per_seed']) == 12
    assert hgb['constructor_kwargs'] == {
        'learning_rate': 0.05, 'max_iter': 100, 'max_leaf_nodes': 31,
        'max_depth': 3, 'min_samples_leaf': 50, 'l2_regularization': 1.0,
        'max_features': 1.0, 'max_bins': 255, 'categorical_features': None,
        'monotonic_cst': None, 'interaction_cst': None, 'warm_start': False,
        'early_stopping': False, 'scoring': 'loss', 'validation_fraction': 0.1,
        'n_iter_no_change': 10, 'tol': 1e-7, 'verbose': 0,
        'random_state': 'exact ensemble seed',
    }
    assert hgb['assembly'].startswith('q10=r10_frozen;')
    assert 'support fitting cannot change the frozen r10 estimator' in hgb['loss_gradient_rule']
    assert hgb['serialization_manifest_order'][2:6] == ['A_ref_mean', 'A_ref_q10', 'A_ref_q50_support', 'A_ref_q90_support']
    neural = topology['neural']
    assert neural['ensemble_seeds'] == [311, 312, 313]
    assert neural['independent_modules_per_seed'] == 7
    assert tuple(neural['modules']) == ('A_REF_ACTION_CORE', 'A_REF_Q50_SUPPORT', 'A_REF_Q90_SUPPORT', 'DOWNSIDE_300', 'RECOVERY_300', 'GIVEBACK_300', 'REMAINING_TAIL_300')
    assert neural['modules']['A_REF_ACTION_CORE']['loss'] == '0.5*MSE(mean)+0.5*pinball_tau_0.10(q10)'
    assert neural['modules']['A_REF_Q50_SUPPORT']['detached_inputs'] == ['A_ref_q10']
    assert neural['modules']['A_REF_Q90_SUPPORT']['detached_inputs'] == ['A_ref_q10', 'A_ref_q50']
    assert neural['optimizer_state_parameter_gradient_or_mutable_scaler_sharing'] is False
    assert neural['graph']['causal_conv1d_channels'] == [32, 32, 32]
    assert neural['graph']['explicit_left_pads'] == [2, 8, 32]
    assert topology['training_order'] == ['A_REF_ACTION_CORE', 'A_REF_Q50_SUPPORT', 'A_REF_Q90_SUPPORT', 'A_REF_JOINT_Q10_Q50_Q90_CALIBRATOR']
    assert topology['shared_read_only_contract']['mutable_object_sharing'] is False

def test_aref_exit_coverage_identity_and_dependency_tamper_fail_closed() -> None:
    topology = prereg.aref_decision_critical_topology_spec()
    assert topology['exit_coverage_identity'] == 'A_ref<=calibrated_q90_upper iff -A_ref>=-calibrated_q90_upper'
    for aref in (-2.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0):
        for q90_upper in (-1.0, -1.0, 0.0, 1.0, 1.0):
            assert (aref <= q90_upper) == (-aref >= -q90_upper)
    raw_q10 = -2.0
    s_low = 0.5
    calibrated_q10_a = 0.0 - s_low * (0.0 - raw_q10)
    calibrated_q10_b = 1.0 - s_low * (1.0 - raw_q10)
    assert calibrated_q10_a != calibrated_q10_b
    def utility(mean_lcb: float, calibrated_q10: float) -> float:
        return mean_lcb + 0.25 * min(calibrated_q10, 0.0)
    composer_before = utility(0.2, -0.4)
    composer_after_q90_perturbation = utility(0.2, -0.4)
    exit_lower_before, exit_lower_after = -1.0, -2.0
    assert composer_before == composer_after_q90_perturbation
    assert exit_lower_before != exit_lower_after
    _assert_prereg_mutations_rejected((
        lambda value: value['exit']['aref_decision_critical_topology']['complete_required_aref_outputs'].remove('A_ref_q90'),
        lambda value: value['exit']['aref_decision_critical_topology']['diagnostic_only_families'].append('A_ref_q50'),
        lambda value: value['exit']['aref_decision_critical_topology']['q50_support_bundle'].update(backpropagation_into_action_bundle=True),
        lambda value: value['exit']['aref_decision_critical_topology']['calibration'].update(missing_or_nonfinite_required_output='diagnostic_incomplete_only'),
        lambda value: value['metrics_and_gates']['action_calibration']['actions']['EXIT'].pop('coverage_identity'),
    ))

def test_context_diagnostics_are_fixed_one_dimensional_post_primary_nonalpha_tables() -> None:
    from v4.scripts import run_pathd_entry_exit_research as runner
    spec = prereg.context_diagnostics_spec()
    assert spec['role'] == 'diagnostics_and_strata_only'
    assert spec['alpha_or_gate'] is False
    authority = spec['execution_authority']
    assert authority['current_preregistration'] == 'AUTHORIZED_POST_ENTRY_PRIMARY_READ_ONLY'
    assert authority['four_box_and_protected_holdout'].startswith('UNAUTHORIZED_AND_UNREACHABLE')
    assert 'five' in authority['blocking_precondition']
    assert 'pooled' in authority['blocking_precondition']
    sources = spec['sources']
    assert sources['VIX']['path_template'] == 'vendor/thetadata/index/vix_1m/{session}.parquet'
    assert sources['VIX']['required_fields'] == ['event_time', 'symbol', 'open', 'high', 'low', 'close', 'context_source', 'is_derived', 'is_proxy', 'is_official_index_data']
    assert sources['VIX']['row_invariants'] == {'symbol': 'VIX', 'context_source': 'thetadata_index_history_ohlc', 'is_derived': False, 'is_proxy': False, 'is_official_index_data': True}
    assert sources['VIX']['clock'] == {'kind': 'HISTORICAL_BAR_END_CLOCK_ONLY', 'timestamp_semantics': 'BAR_OPEN_TIMESTAMP', 'available_at': 'event_time+60 seconds', 'eligible': 'latest available_at<=decision_time and age_seconds from available_at in [0,90]', 'live_parity_claim': False}
    assert sources['ES']['path_template'] == 'raw/databento/glbx_es_ohlcv_1m/{session}.es_c_0.ohlcv-1m.parquet'
    assert sources['ES']['empty_source_sessions'] == ['2025-09-19', '2025-12-19', '2026-03-20', '2026-06-18']
    assert sources['ES']['degraded_sessions'] == ['2025-09-17', '2025-09-24', '2025-11-28', '2026-03-16', '2026-04-10']
    assert sources['VX']['availability_boundary_session'] == '2026-04-01'
    assert sources['VX']['known_publishers'] == [105, 106]
    assert sources['SPX_REFERENCE']['path_template'] == 'vendor/thetadata/index/spx_1m/{session}.parquet'
    assert sources['SPX_REFERENCE']['role'] == 'ES_MINUS_SPX_BASIS_REFERENCE_ONLY_NONALPHA'
    assert sources['SPX_REFERENCE']['receipt_required'] is True
    assert sources['SPX_REFERENCE']['required_fields'] == ['event_time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'context_source', 'is_derived', 'is_proxy', 'is_official_index_data']
    assert sources['SPX_REFERENCE']['row_invariants'] == {'symbol': 'SPX', 'context_source': 'thetadata_index_history_ohlc', 'is_derived': False, 'is_proxy': False, 'is_official_index_data': True}
    assert sources['SPX_REFERENCE']['clock'] == {'kind': 'HISTORICAL_BAR_END_CLOCK_ONLY', 'timestamp_semantics': 'BAR_OPEN_TIMESTAMP', 'available_at': 'event_time+60 seconds', 'eligible': 'latest available_at<=anchor and age_seconds from available_at in [0,90]', 'live_parity_claim': False}
    payload = spec['artifact_contract']['payload_spec']
    payload_spec = payload['source_receipt_row']
    assert payload['source_order'] == ['VIX', 'ES', 'VX', 'SPX_REFERENCE']
    assert 'SPX_REFERENCE' in payload_spec['types_and_nulls']
    assert 'SPX_REFERENCE' in payload_spec['ordering']
    assert payload['source_selection']['schema'] == 'EntryContextSourceSelectionV1'
    assert payload['anchor_context_rows']['schema'] == 'EntryContextAnchorRowV1'
    assert payload['anchor_context_rows']['anchor_kinds_in_order'] == ['SESSION_1000', 'ENTRY_DECISION']
    assert 'including ENTER, WAIT, fill, and no-fill' in payload['anchor_context_rows']['coverage']
    assert 'fold-1-through-fold-5 concatenation' in payload['anchor_context_rows']['root']
    assert payload['age_samples']['sources_in_order'] == ['VIX', 'ES', 'VX', 'SPX_REFERENCE']
    assert 'never average fold quantiles' in payload['age_samples']['pooled']
    assert prereg.context_age_quantile_type7([], 0.5) is None
    assert prereg.context_age_quantile_type7([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], 0.5) == 1.5
    assert prereg.context_age_quantile_type7([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], 0.95) == 2.85
    for samples, quantile in (([True], 0.5), ([-1], 0.5), ([0], 0.9)):
        with pytest.raises(ValueError):
            prereg.context_age_quantile_type7(samples, quantile)
    consolidate = spec['publisher_consolidation']
    assert consolidate['identity'] == ['symbol', 'instrument_id', 'ts_event']
    assert consolidate['within_publisher_duplicate'] == 'invalid:DUPLICATE_WITHIN_PUBLISHER'
    assert consolidate['publisher_order'] == 'ascending numeric publisher_id'
    assert consolidate['high'] == 'maximum across publishers'
    assert consolidate['low'] == 'minimum across publishers'
    assert consolidate['volume'] == 'sum across publishers'
    assert spec['lag_15m']['forward_fill'] is False
    derived = spec['derived_values']
    assert derived['es_15m_bps'] == '10000.0*(ES_close_current/ES_close_lag15-1.0); nonpositive or nonfinite denominator => SCHEMA_INVALID'
    assert derived['vx_minus_vix_points'] == 'VX_close_current - VIX_close_current in index points at the same anchor clock'
    assert spec['missing_status_priority'] == ['NOT_REQUESTED_PRE_VX_BOUNDARY', 'MISSING_FILE', 'EMPTY_SOURCE', 'NO_CAUSAL_BAR', 'STALE_GT90S', 'NONFINITE_CLOSE', 'DUPLICATE_WITHIN_PUBLISHER', 'SCHEMA_INVALID']
    assert spec['source_status_vocabulary'] == ['FRESH', *spec['missing_status_priority']]
    bins = spec['bins']
    assert bins['interval_rule'] == 'left-closed/right-open except the unbounded edge bins'
    assert bins['vix_level'] == ['LT15', '15_20', '20_25', '25_30', 'GE30']
    assert bins['es_15m_bps'] == ['LT_NEG25', 'NEG25_NEG10', 'NEG10_POS10', 'POS10_POS25', 'GE25']
    assert bins['es_minus_spx_basis_bps'] == ['LT_NEG5', 'NEG5_POS5', 'GE5']
    assert bins['vx_minus_vix_points'] == ['LT_NEG2', 'NEG2_0', '0_2', '2_4', 'GE4']
    assert bins['status'] == spec['source_status_vocabulary']
    expected_edges = {
        'vix_level': ['NEG_INF', 15.0.hex(), 20.0.hex(), 25.0.hex(), 30.0.hex(), 'POS_INF'],
        'vx_level': ['NEG_INF', 15.0.hex(), 20.0.hex(), 25.0.hex(), 30.0.hex(), 'POS_INF'],
        'vix_15m_point_change': ['NEG_INF', (-1.0).hex(), (-0.25).hex(), 0.25.hex(), 1.0.hex(), 'POS_INF'],
        'vx_15m_point_change': ['NEG_INF', (-1.0).hex(), (-0.25).hex(), 0.25.hex(), 1.0.hex(), 'POS_INF'],
        'es_15m_bps': ['NEG_INF', (-25.0).hex(), (-10.0).hex(), 10.0.hex(), 25.0.hex(), 'POS_INF'],
        'es_minus_spx_basis_bps': ['NEG_INF', (-5.0).hex(), 5.0.hex(), 'POS_INF'],
        'vx_minus_vix_points': ['NEG_INF', (-2.0).hex(), 0.0.hex(), 2.0.hex(), 4.0.hex(), 'POS_INF'],
    }
    assert bins['edges_float64_hex'] == expected_edges
    for stratifier, edges in expected_edges.items():
        assert len(bins[stratifier]) == len(edges) - 1
        assert len(bins[stratifier]) == len(set(bins[stratifier]))
    assert 'every registered one-dimensional bin' in bins['zero_retention']
    assert spec['anchors']['joint_or_multidimensional_strata'] is False
    assert set(spec['forbidden_consumers']) >= {'model feature', 'loss', 'action mask', 'family selection', 'comparator selection', 'minimum power', 'feasibility verdict'}
    contract = prereg.entry_future_api_contract()
    assert 'EntryPooledAcceptanceResultV1' in contract['schema_versions']
    assert contract['dataclass_fields']['EntryContextSourceSelectionV1'] == ['schema_version', 'source', 'status', 'requested', 'source_degraded', 'manifest_relative_path', 'source_file_sha256', 'current_component_locators', 'lag15_component_locators', 'current_available_at_ns', 'lag15_available_at_ns', 'current_age_ns', 'lag15_age_ns', 'current_close_float64_hex', 'lag15_close_float64_hex', 'selection_sha256']
    assert contract['dataclass_fields']['EntryContextAnchorRowV1'] == ['schema_version', 'outer_fold', 'policy_id', 'policy_order_index', 'policy_evaluation_sha256', 'terminal_journal_sha256', 'anchor_kind', 'anchor_ordinal', 'anchor_id', 'session', 'anchor_time_ns', 'source_transition_sha256', 'action', 'filled', 'trade_id', 'trade_pnl_micros', 'session_pnl_micros', 'source_selections', 'source_selections_root_sha256', 'derived_values_float64_hex', 'bin_ids', 'row_sha256']
    outer_fields = contract['dataclass_fields']['EntryOuterContextDiagnosticsV1']
    pooled_fields = contract['dataclass_fields']['EntryPooledContextDiagnosticsV1']
    for field in ('policy_journal_bindings_root_sha256', 'source_receipts_root_sha256', 'anchor_context_rows', 'anchor_context_rows_root_sha256', 'age_samples_ns_by_source', 'age_samples_root_sha256', 'tables_sha256'):
        assert field in outer_fields
    for field in ('policy_journal_bindings_root_sha256', 'anchor_context_rows', 'anchor_context_rows_root_sha256', 'age_samples_ns_by_source', 'age_samples_root_sha256', 'tables_sha256'):
        assert field in pooled_fields
    transaction = spec['artifact_contract']['transaction']
    assert transaction['access_receipt_fields_in_order'][-1] == 'receipt_sha256'
    assert 'anchor_context_rows_root_sha256' in transaction['outer_receipt_fields_in_order']
    assert 'anchor_context_rows_root_sha256' in transaction['pooled_receipt_fields_in_order']
    assert {'reason_code', 'state_before_abort', 'partial_artifacts', 'no_redecode', 'no_reauthorization', 'receipt_sha256'} <= set(transaction['abort_receipt_fields_in_order'])
    names = _api_names()
    required = ('v4.research.pathd_entry_exit.validate_entry_pooled_acceptance_result', 'v4.scripts.run_pathd_entry_exit_research.run_fixed_entry_context_diagnostics', 'v4.scripts.run_pathd_entry_exit_research.validate_entry_outer_context_diagnostics', 'v4.scripts.run_pathd_entry_exit_research.validate_entry_pooled_context_diagnostics')
    assert set(required) <= set(names)
    assert not any(('holdout_context_diagnostics' in name for name in names))
    ordering = authority['blocking_precondition']
    assert 'pooled entry-acceptance result receipt' in ordering
    assert list(inspect.signature(runner.run_fixed_entry_context_diagnostics).parameters) == ['authorization']
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        runner.run_fixed_entry_context_diagnostics(object())
    for validator, kwargs in ((runner.validate_entry_outer_context_diagnostics, {'authorization': object(), 'outer_fold': 1}), (runner.validate_entry_pooled_context_diagnostics, {'authorization': object()})):
        missing_spx = {'source_receipts': [{'source': source} for source in ('VIX', 'ES', 'VX')], 'source_receipts_root_sha256': '0' * 64}
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            validator(missing_spx, **kwargs)
    from v4.research import pathd_entry_execution_v32 as execution
    producer_source = inspect.getsource(execution.build_spx_reference_transaction)
    writer_source = inspect.getsource(runner.run_fixed_entry_context_diagnostics)
    assert 'SPX_REFERENCE' in producer_source and 'source_receipts_root_sha256' in producer_source
    assert 'assert_context_diagnostics_authorization_current' in writer_source
    assert 'EntryContextAnchorRowV1' in writer_source and 'context_age_quantile_type7' in writer_source
    pooled_gate_json = json.dumps(prereg.entry_pooled_gate_input_spec(), sort_keys=True).lower()
    assert not any(token in pooled_gate_json for token in ('vix', 'spx_reference', 'context_diagnostics', 'anchor_context_rows', 'diagnostic_inventory_receipt'))
