"""Offline two-arm normalization, reusing ID-clustered paired analysis; no inference."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from aggregate_factorial import contrast
from aggregate_factorial import summarize as paired_summary


def summarize(document):
    # The existing analyzer's early bit is only an internal two-arm index here.
    # Actual EARLY_FIRST_QUERY remains false in both measured configurations.
    normalized = copy.deepcopy(document)
    for phase in normalized.get('phases', []):
        if phase.get('block') in (1, 2):
            tokens = phase['max_num_batched_tokens']
            if tokens not in (8192, 32768):
                raise ValueError('Unexpected model max batch')
            phase['early_first_query'] = tokens == 8192
            phase['rerank_priority'] = 0
    result = paired_summary(normalized)
    measured = [phase for phase in document.get('phases', []) if phase.get('block') in (1, 2)]
    ragu = [(phase, row) for phase in measured for row in phase.get('cases', [])]
    menon = [row for phase in measured for row in phase.get('menon', [])]
    result['measured_denominators'] = {
        'ragu_requested': sum(phase.get('requested', 0) for phase in measured),
        'ragu_attempted': len(ragu),
        'ragu_status_counts': dict(Counter(row.get('status', 'missing') for phase, row in ragu)),
        'menon_offered': len(menon),
        'menon_status_counts': dict(Counter(row.get('status', 'missing') for row in menon)),
        'latency_scope': 'Successful cases and successful matched pairs only; failures remain in the overall attempt denominator',
    }
    result['failed_measured_ragu_cases'] = [
        {'phase': phase.get('name'), 'model_block': phase.get('model_block'),
         'max_num_batched_tokens': phase.get('max_num_batched_tokens'), 'concurrency': phase.get('concurrency'),
         **{name: row[name] for name in ('id', 'position', 'status', 'error_type', 'total_ms', 'http_requests') if name in row}}
        for phase, row in ragu if row.get('status') != 'ok']
    for cell in result['cells']:
        cell['max_num_batched_tokens'] = 8192 if cell.pop('early') else 32768
        cell.pop('priority', None)
    result['effects'] = [row for row in result['effects'] if row['factor'] == 'early']
    for effect in result['effects']:
        effect['factor'] = 'max_num_batched_tokens_8192_minus_32768'
        for metric in effect['metrics'].values():
            metric.pop('conditioned_on_other_factor', None)
    result['stability_8192_against_32768'] = [row for row in result.pop('stability_against_e0p0')
                                               if row['candidate_early'] == 1 and row['candidate_priority'] == 0]
    for row in result['stability_8192_against_32768']:
        for key in ('reference_early', 'reference_priority', 'candidate_early', 'candidate_priority'):
            row.pop(key)
        row['limit'] = 'First changed observable stage only; complete raw material remains server-local for factual review'
    for row in result['repeatability_same_arm']:
        row['max_num_batched_tokens'] = 8192 if row.pop('early') else 32768
        row.pop('priority', None)
    result.pop('menon_alone_drift_after_minus_before')
    result['menon_alone_drift_by_model_block'] = {}
    alone = {row['name']: row.get('cases', []) for row in document.get('menon_alone', [])}
    for tag in ('A1', 'B1', 'B2', 'A2'):
        before = {row['id']: row for row in alone.get(tag + '_before', []) if row['status'] == 'ok'}
        after = {row['id']: row for row in alone.get(tag + '_after', []) if row['status'] == 'ok'}
        result['menon_alone_drift_by_model_block'][tag] = {
            metric: contrast([(case_id, after[case_id][metric] - before[case_id][metric], 0)
                              for case_id in sorted(before.keys() & after.keys())
                              if before[case_id].get(metric) is not None and after[case_id].get(metric) is not None])
            for metric in ('ttft_ms', 'total_ms')}
    result['model_blocks'] = [{key: block[key] for key in ('tag', 'ordinal', 'max_num_batched_tokens', 'pair_block', 'status', 'receipt_sha256') if key in block}
                              for block in document.get('model_blocks', [])]
    result['model_block_status_counts'] = dict(Counter(block.get('status') for block in document.get('model_blocks', [])))
    result['design'] = {'max_num_batched_tokens_A': 32768, 'max_num_batched_tokens_B': 8192,
                        'fresh_model_restarts': 4, 'graph_loads': 1, 'clock_frozen': True,
                        **document.get('constants', {})}
    result['warnings'] += ['Every block uses a fresh model and identical warmup; residual temporal load remains possible',
                            'Eight independent question IDs; C8 repetitions and two time blocks do not establish release-wide noninferiority',
                            'Hash changes need server-local factual review before any same-quality claim']
    if result['failed_measured_ragu_cases']:
        result['warnings'].append('Phase completion is not request success; measured failures are listed separately and remain in the denominator')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
